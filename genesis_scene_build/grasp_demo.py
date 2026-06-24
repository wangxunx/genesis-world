"""M1 demo: hard-coded pick of 011_banana and place into 024_bowl.

Reuses build_scene() from build_scene.py. The robot performs a scripted
pick-and-place state machine using IK + motion planning + force-controlled grasp.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import genesis as gs
from genesis.utils.geom import euler_to_quat, quat_to_xyz

from build_scene import build_scene
from scene_config import FRANKA_QPOS, TABLE_TOP_Z

# Object / target names.
PICK_OBJECT = "011_banana"
PLACE_TARGET = "024_bowl"

# End-effector control constants (world frame, meters).
# The hand-link origin sits ~0.095 m above the fingertips, so to place the
# fingertips near the tabletop (z=TABLE_TOP_Z) the hand link must be ~0.105 above it.
PREGRASP_CLEARANCE = 0.18  # hand-link height above object centroid before descending
GRASP_HAND_Z = TABLE_TOP_Z + 0.105  # hand-link world z at grasp (fingertips ~1cm above table)
LIFT_HAND_Z = TABLE_TOP_Z + 0.30  # absolute world z to lift the hand to after grasping
RETREAT_HAND_Z = TABLE_TOP_Z + 0.35  # absolute world z to retreat to after releasing
PLACE_HAND_Z_ABOVE_BOWL = 0.16  # hand-link height above bowl centroid when releasing

GRIPPER_OPEN = 0.04
GRIPPER_CLOSE_FORCE = -10.0  # N, finger force-control for a secure grasp

MOTORS_DOF = np.arange(7)
FINGERS_DOF = np.arange(7, 9)


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


def _goto_direct(bundle, pos, quat, *, finger_cmd, steps=120, force_close=False):
    """Move arm via direct position control (no planning), holding gripper command."""
    qpos = _ik(bundle, pos, quat)
    for _ in range(steps):
        bundle.franka.control_dofs_position(qpos[:-2], MOTORS_DOF)
        if force_close:
            bundle.franka.control_dofs_force(
                np.array([GRIPPER_CLOSE_FORCE, GRIPPER_CLOSE_FORCE]), FINGERS_DOF
            )
        else:
            bundle.franka.control_dofs_position(np.array([finger_cmd, finger_cmd]), FINGERS_DOF)
        bundle.scene.step()
        bundle.update_wrist_cam()
    return qpos


def run_pick_place(bundle, *, save_frames: bool = False):
    franka = bundle.franka
    banana = bundle.ycb[PICK_OBJECT]
    bowl = bundle.ycb[PLACE_TARGET]

    frames = []

    def snap(tag):
        if save_frames and bundle.world_cam is not None:
            frames.append((tag, bundle.world_cam.render(rgb=True)[0]))

    # Let objects settle on the table.
    _settle(bundle, 60)
    snap("00_start")

    banana_pos, banana_yaw = _obj_xy_yaw(banana)
    # Close the jaws across the banana's short axis (perpendicular to its long axis).
    grasp_quat = _topdown_quat(banana_yaw + 90.0)

    # 1) Pre-grasp above the banana, gripper open.
    pregrasp = np.array([banana_pos[0], banana_pos[1], banana_pos[2] + PREGRASP_CLEARANCE])
    _goto_plan(bundle, pregrasp, grasp_quat, finger=GRIPPER_OPEN)
    snap("01_pregrasp")

    # 2) Descend to grasp height (fingertips just above the tabletop).
    grasp = np.array([banana_pos[0], banana_pos[1], GRASP_HAND_Z])
    _goto_direct(bundle, grasp, grasp_quat, finger_cmd=GRIPPER_OPEN, steps=100)
    snap("02_reach")

    # 3) Close the gripper with force control.
    _goto_direct(bundle, grasp, grasp_quat, finger_cmd=0.0, steps=100, force_close=True)
    snap("03_grasp")

    # 4) Lift.
    lift = np.array([banana_pos[0], banana_pos[1], LIFT_HAND_Z])
    _goto_direct(bundle, lift, grasp_quat, finger_cmd=0.0, steps=100, force_close=True)
    snap("04_lift")

    # 5) Move above the bowl (re-read bowl pose in case it moved).
    bowl_pos = bowl.get_pos().cpu().numpy().reshape(-1)
    above_bowl = np.array([bowl_pos[0], bowl_pos[1], bowl_pos[2] + PLACE_HAND_Z_ABOVE_BOWL])
    _goto_direct(bundle, above_bowl, grasp_quat, finger_cmd=0.0, steps=120, force_close=True)
    snap("05_above_bowl")

    # 6) Release the banana into the bowl.
    _goto_direct(bundle, above_bowl, grasp_quat, finger_cmd=GRIPPER_OPEN, steps=80)
    snap("06_release")

    # 7) Retreat upward and let the banana settle.
    retreat = np.array([bowl_pos[0], bowl_pos[1], RETREAT_HAND_Z])
    _goto_direct(bundle, retreat, grasp_quat, finger_cmd=GRIPPER_OPEN, steps=80)
    _settle(bundle, 60)
    snap("07_done")

    success = check_success(banana, bowl)
    return success, frames


def check_success(banana, bowl) -> bool:
    bp = banana.get_pos().cpu().numpy().reshape(-1)
    wp = bowl.get_pos().cpu().numpy().reshape(-1)
    horizontal = np.linalg.norm(bp[:2] - wp[:2])
    # Banana centroid should be within the bowl radius and above the bowl base.
    return bool(horizontal < 0.08 and bp[2] > wp[2] - 0.02)


def main() -> None:
    parser = argparse.ArgumentParser(description="M1: pick banana, place into bowl.")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("--save-frames", action="store_true", help="Save world-cam frames per stage.")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)
    bundle = build_scene(show_viewer=args.vis, n_envs=1, add_world_cam=True, add_wrist_cam=True)

    success, frames = run_pick_place(bundle, save_frames=args.save_frames)
    print(f"[grasp_demo] success = {success}")

    if args.save_frames:
        import imageio.v2 as imageio

        out_dir = _ROOT / "grasp_demo_frames"
        out_dir.mkdir(exist_ok=True)
        for tag, img in frames:
            imageio.imwrite(out_dir / f"{tag}.png", img)
        print(f"[grasp_demo] saved {len(frames)} frames to {out_dir}")


if __name__ == "__main__":
    main()
