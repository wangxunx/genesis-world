"""Scripted pick-and-place, parameterized via TaskSpec.

Reuses build_scene() from build_scene.py. The robot performs a scripted
pick-and-place state machine using IK + motion planning + force-controlled grasp.

A task is described by a `TaskSpec` (what to pick, where to place, success
tolerance). Per-object grasp behavior is described by a `GraspProfile`.

Examples:
    # pick the banana and place it into the bowl (default)
    uv run python grasp_demo.py --cpu --save-frames

    # pick the mug and place it onto a tabletop coordinate
    uv run python grasp_demo.py --cpu --pick 025_mug --place 0.5,0.2
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import genesis as gs
from genesis.utils.geom import euler_to_quat, quat_to_xyz

from build_scene import build_scene
from scene_config import FRANKA_QPOS, TABLE_TOP_Z

# A place target is either an object name (place onto/into it) or a tabletop (x, y).
PlaceTarget = Union[str, tuple[float, float]]

# End-effector control constants shared across objects (world frame, meters).
# The hand-link origin sits ~0.095 m above the fingertips, so to place the
# fingertips near the tabletop (z=TABLE_TOP_Z) the hand link must be ~0.105 above it.
PREGRASP_CLEARANCE = 0.18  # hand-link height above object centroid before descending
LIFT_HAND_Z = TABLE_TOP_Z + 0.30  # absolute world z to lift the hand to after grasping
RETREAT_HAND_Z = TABLE_TOP_Z + 0.35  # absolute world z to retreat to after releasing
PLACE_HAND_Z_ABOVE_TARGET = 0.16  # hand-link height above the place reference z when releasing

GRIPPER_OPEN = 0.04

MOTORS_DOF = np.arange(7)
FINGERS_DOF = np.arange(7, 9)


@dataclass(frozen=True)
class GraspProfile:
    """Per-object top-down grasp parameters."""

    yaw_offset: float = 90.0  # deg added to the object yaw to orient the jaws
    grasp_hand_z: float = TABLE_TOP_Z + 0.105  # absolute hand-link z at grasp
    close_force: float = -10.0  # N, finger force-control while holding the grasp


# Defaults are tuned for the banana; other objects fall back to DEFAULT_PROFILE
# and can be tuned here as needed.
DEFAULT_PROFILE = GraspProfile()
GRASP_PROFILES: dict[str, GraspProfile] = {
    "011_banana": GraspProfile(yaw_offset=90.0, grasp_hand_z=TABLE_TOP_Z + 0.105, close_force=-10.0),
    # apple/orange: large smooth spheres (~7.5 cm, original scale) -- only marginally
    # graspable (excluded from the reliable pickable pool); profiles kept for completeness.
    "013_apple": GraspProfile(yaw_offset=0.0, grasp_hand_z=TABLE_TOP_Z + 0.10, close_force=-12.0),
    "017_orange": GraspProfile(yaw_offset=0.0, grasp_hand_z=TABLE_TOP_Z + 0.12, close_force=-12.0),
    # lemon: small oblate ellipsoid, grasped near its equator -- reliable (verified 6/6).
    "014_lemon": GraspProfile(yaw_offset=0.0, grasp_hand_z=TABLE_TOP_Z + 0.10, close_force=-12.0),
    # plum: small near-sphere, grasped near its equator -- reliable (verified 5/5).
    "018_plum": GraspProfile(yaw_offset=0.0, grasp_hand_z=TABLE_TOP_Z + 0.10, close_force=-12.0),
    # pear: profile kept for completeness, but its round cross-section slips on lift
    # (not reliably graspable -- excluded from the pickable pool). Jaws close across short axis.
    "016_pear": GraspProfile(yaw_offset=90.0, grasp_hand_z=TABLE_TOP_Z + 0.13, close_force=-12.0),
    "025_mug": GraspProfile(yaw_offset=0.0, grasp_hand_z=TABLE_TOP_Z + 0.085, close_force=-12.0),
    "006_mustard_bottle": GraspProfile(yaw_offset=0.0, grasp_hand_z=TABLE_TOP_Z + 0.11, close_force=-12.0),
}


@dataclass
class TaskSpec:
    """Describes a single pick-and-place task."""

    pick_object: str
    place_target: PlaceTarget
    success_tol: float = 0.08

    def grasp_profile(self) -> GraspProfile:
        return GRASP_PROFILES.get(self.pick_object, DEFAULT_PROFILE)


def _topdown_quat(yaw_deg: float) -> np.ndarray:
    """Top-down grasp orientation with an extra yaw about world z."""
    return euler_to_quat(np.array([180.0, 0.0, yaw_deg]))


def _settle(bundle, steps: int) -> None:
    hold = np.array(FRANKA_QPOS)
    for _ in range(steps):
        bundle.franka.control_dofs_position(hold)
        bundle.scene.step()
        bundle.update_wrist_cam()


def _obj_xy_yaw(entity) -> tuple[np.ndarray, float]:
    pos = entity.get_pos().cpu().numpy().reshape(-1)
    quat = entity.get_quat().cpu().numpy().reshape(-1)
    yaw = float(quat_to_xyz(quat, degrees=True)[2])
    return pos, yaw


def _ik(bundle, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    hand = bundle.franka.get_link("hand")
    return bundle.franka.inverse_kinematics(link=hand, pos=pos, quat=quat)


def _goto_plan(bundle, pos, quat, *, finger, num_waypoints=150, settle=20):
    """Plan a collision-free path to (pos, quat) and execute it."""
    qpos = _ik(bundle, pos, quat)
    qpos[-2:] = finger
    path = bundle.franka.plan_path(qpos_goal=qpos, num_waypoints=num_waypoints)
    for wp in path:
        bundle.franka.control_dofs_position(wp)
        bundle.scene.step()
        bundle.update_wrist_cam()
    for _ in range(settle):
        bundle.franka.control_dofs_position(qpos)
        bundle.scene.step()
        bundle.update_wrist_cam()
    return qpos


def _goto_direct(bundle, pos, quat, *, finger_cmd, steps=120, close_force=None):
    """Move arm via direct position control (no planning), holding gripper command.

    If `close_force` is given, the fingers are force-controlled (grasping); otherwise
    they are position-controlled to `finger_cmd`.
    """
    qpos = _ik(bundle, pos, quat)
    for _ in range(steps):
        bundle.franka.control_dofs_position(qpos[:-2], MOTORS_DOF)
        if close_force is not None:
            bundle.franka.control_dofs_force(np.array([close_force, close_force]), FINGERS_DOF)
        else:
            bundle.franka.control_dofs_position(np.array([finger_cmd, finger_cmd]), FINGERS_DOF)
        bundle.scene.step()
        bundle.update_wrist_cam()
    return qpos


def _descend_vertical(bundle, xy, z_from, z_to, quat, *, finger, steps=80, settle=15):
    """Descend straight down along a fixed xy by interpolating z and re-solving IK.

    A single IK snap can swing the hand laterally mid-descent; for tight clearances
    (e.g. a sphere nearly as wide as the gripper) that sideways sweep grazes and rolls
    the object away. Stepping z keeps the hand on a vertical line.
    """
    qpos = None
    for z in np.linspace(z_from, z_to, steps):
        qpos = _ik(bundle, np.array([xy[0], xy[1], z]), quat)
        qpos[-2:] = finger
        bundle.franka.control_dofs_position(qpos)
        bundle.scene.step()
        bundle.update_wrist_cam()
    for _ in range(settle):
        bundle.franka.control_dofs_position(qpos)
        bundle.scene.step()
        bundle.update_wrist_cam()
    return qpos


def _resolve_place(bundle, place_target: PlaceTarget) -> tuple[np.ndarray, float, object]:
    """Return (target_xy, reference_z, target_entity_or_None) for a place target."""
    if isinstance(place_target, str):
        ent = bundle.ycb[place_target]
        p = ent.get_pos().cpu().numpy().reshape(-1)
        return np.array([p[0], p[1]]), float(p[2]), ent
    x, y = place_target
    return np.array([float(x), float(y)]), TABLE_TOP_Z, None


def run_pick_place(bundle, task: TaskSpec, *, save_frames: bool = False):
    pick_entity = bundle.ycb[task.pick_object]
    profile = task.grasp_profile()

    frames = []

    def snap(tag):
        if save_frames and bundle.world_cam is not None:
            frames.append((tag, bundle.world_cam.render(rgb=True)[0]))

    # Let objects settle on the table.
    _settle(bundle, 60)
    snap("00_start")

    obj_pos, obj_yaw = _obj_xy_yaw(pick_entity)
    # Orient the jaws relative to the object (e.g. across a banana's short axis).
    grasp_quat = _topdown_quat(obj_yaw + profile.yaw_offset)

    # 1) Pre-grasp above the object, gripper open.
    pregrasp = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + PREGRASP_CLEARANCE])
    _goto_plan(bundle, pregrasp, grasp_quat, finger=GRIPPER_OPEN)
    snap("01_pregrasp")

    # 2) Descend straight down to grasp height (vertical path avoids grazing the object).
    _descend_vertical(
        bundle, (obj_pos[0], obj_pos[1]), pregrasp[2], profile.grasp_hand_z, grasp_quat, finger=GRIPPER_OPEN
    )
    grasp = np.array([obj_pos[0], obj_pos[1], profile.grasp_hand_z])
    snap("02_reach")

    # 3) Close the gripper with force control.
    _goto_direct(bundle, grasp, grasp_quat, finger_cmd=0.0, steps=100, close_force=profile.close_force)
    snap("03_grasp")

    # 4) Lift straight up from the grasp xy.
    lift = np.array([grasp[0], grasp[1], LIFT_HAND_Z])
    _goto_direct(bundle, lift, grasp_quat, finger_cmd=0.0, steps=100, close_force=profile.close_force)
    snap("04_lift")

    # 5) Move above the place target.
    place_xy, place_ref_z, _ = _resolve_place(bundle, task.place_target)
    above = np.array([place_xy[0], place_xy[1], place_ref_z + PLACE_HAND_Z_ABOVE_TARGET])
    _goto_direct(bundle, above, grasp_quat, finger_cmd=0.0, steps=120, close_force=profile.close_force)
    snap("05_above_target")

    # 6) Release the object.
    _goto_direct(bundle, above, grasp_quat, finger_cmd=GRIPPER_OPEN, steps=80)
    snap("06_release")

    # 7) Retreat upward and let the object settle.
    retreat = np.array([place_xy[0], place_xy[1], RETREAT_HAND_Z])
    _goto_direct(bundle, retreat, grasp_quat, finger_cmd=GRIPPER_OPEN, steps=80)
    _settle(bundle, 60)
    snap("07_done")

    success = check_success(bundle, task)
    return success, frames


def check_success(bundle, task: TaskSpec) -> bool:
    pick_pos = bundle.ycb[task.pick_object].get_pos().cpu().numpy().reshape(-1)
    place_xy, place_ref_z, _ = _resolve_place(bundle, task.place_target)
    horizontal = float(np.linalg.norm(pick_pos[:2] - place_xy))
    # Object should be within tolerance of the target xy and resting at/above the reference.
    return bool(horizontal < task.success_tol and pick_pos[2] > place_ref_z - 0.02)


def _parse_place(text: str) -> PlaceTarget:
    """Parse a --place argument: either an object name or 'x,y' tabletop coords."""
    if "," in text:
        x_str, y_str = text.split(",")
        return (float(x_str), float(y_str))
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Scripted pick-and-place (parameterized via TaskSpec).")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("--pick", default="011_banana", help="Object name to pick.")
    parser.add_argument(
        "--place",
        default="024_bowl",
        help="Place target: an object name (e.g. 024_bowl) or tabletop coords 'x,y'.",
    )
    parser.add_argument("--tol", type=float, default=0.08, help="Success tolerance (m).")
    parser.add_argument("--save-frames", action="store_true", help="Save world-cam frames per stage.")
    args = parser.parse_args()

    task = TaskSpec(pick_object=args.pick, place_target=_parse_place(args.place), success_tol=args.tol)

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)
    bundle = build_scene(show_viewer=args.vis, n_envs=1, add_world_cam=True, add_wrist_cam=True)

    success, frames = run_pick_place(bundle, task, save_frames=args.save_frames)
    print(f"[grasp_demo] task: pick={task.pick_object} place={task.place_target} -> success = {success}")

    if args.save_frames:
        import imageio.v2 as imageio

        out_dir = _ROOT / "grasp_demo_frames"
        out_dir.mkdir(exist_ok=True)
        for tag, img in frames:
            imageio.imwrite(out_dir / f"{tag}.png", img)
        print(f"[grasp_demo] saved {len(frames)} frames to {out_dir}")


if __name__ == "__main__":
    main()
