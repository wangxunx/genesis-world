"""Build a Franka manipulation scene with a table and YCB objects."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import genesis as gs
from genesis.utils.geom import trans_R_to_T, euler_to_R

from scene_config import (
    ASSETS,
    FRANKA_EULER,
    FRANKA_FORCE_MAX,
    FRANKA_FORCE_MIN,
    FRANKA_KP,
    FRANKA_KV,
    FRANKA_POS,
    FRANKA_QPOS,
    TABLE_CENTER,
    TABLE_COLOR,
    TABLE_LEG_COLOR,
    TABLE_LEG_INSET,
    TABLE_LEG_SIZE,
    TABLE_TOP_SIZE,
    TABLE_TOP_Z,
    WORLD_CAM_FOV,
    WORLD_CAM_LOOKAT,
    WORLD_CAM_POS,
    WORLD_CAM_RES,
    WRIST_CAM_FOV,
    WRIST_CAM_LINK,
    WRIST_CAM_OFFSET_EULER,
    WRIST_CAM_OFFSET_POS,
    WRIST_CAM_RES,
    YCB_LAYOUT,
    get_ycb_assets,
)
from setup_assets import setup_assets


@dataclass
class SceneBundle:
    """Everything a caller needs to drive and observe the scene."""

    scene: gs.Scene
    franka: gs.RigidEntity
    ycb: dict[str, gs.RigidEntity]
    world_cam: "gs.vis.camera.Camera | None" = None
    wrist_cam: "gs.vis.camera.Camera | None" = None
    _wrist_link: "gs.RigidLink | None" = None

    def update_wrist_cam(self) -> None:
        """Sync the wrist camera to the current hand-link pose. Call each step."""
        if self.wrist_cam is not None:
            self.wrist_cam.move_to_attach()

    def render(self, *, rgb: bool = True, depth: bool = False):
        """Render both cameras (if present). Returns a dict keyed by camera name."""
        out = {}
        if self.world_cam is not None:
            out["world"] = self.world_cam.render(rgb=rgb, depth=depth)
        if self.wrist_cam is not None:
            out["wrist"] = self.wrist_cam.render(rgb=rgb, depth=depth)
        return out


def _ensure_assets() -> Path:
    if not (ASSETS / "robots" / "franka" / "panda.xml").exists():
        setup_assets()
    return ASSETS


def _add_table(scene: gs.Scene) -> None:
    cx, cy = TABLE_CENTER
    top_lx, top_ly, top_lz = TABLE_TOP_SIZE
    leg_lx, leg_ly = TABLE_LEG_SIZE
    leg_lz = TABLE_TOP_Z - top_lz

    # Tabletop: its top surface sits exactly at TABLE_TOP_Z.
    scene.add_entity(
        morph=gs.morphs.Box(
            size=TABLE_TOP_SIZE,
            pos=(cx, cy, TABLE_TOP_Z - top_lz / 2),
            fixed=True,
        ),
        surface=gs.surfaces.Default(color=TABLE_COLOR),
    )

    # Four legs from the floor up to the underside of the tabletop.
    dx = top_lx / 2 - TABLE_LEG_INSET
    dy = top_ly / 2 - TABLE_LEG_INSET
    for sx in (-1, 1):
        for sy in (-1, 1):
            scene.add_entity(
                morph=gs.morphs.Box(
                    size=(leg_lx, leg_ly, leg_lz),
                    pos=(cx + sx * dx, cy + sy * dy, leg_lz / 2),
                    fixed=True,
                ),
                surface=gs.surfaces.Default(color=TABLE_LEG_COLOR),
            )


def build_scene(
    *,
    show_viewer: bool = False,
    n_envs: int = 1,
    add_world_cam: bool = True,
    add_wrist_cam: bool = True,
    draw_world_frame: bool = False,
) -> SceneBundle:
    assets = _ensure_assets()
    ycb_assets = get_ycb_assets()

    cx, cy = TABLE_CENTER
    camera_lookat = (cx, cy, TABLE_TOP_Z)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=2),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 960),
            camera_pos=(cx + 1.0, -1.2, 1.5),
            camera_lookat=camera_lookat,
            camera_fov=45,
        ),
        show_viewer=show_viewer,
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
    )

    scene.add_entity(gs.morphs.Plane())
    _add_table(scene)

    ycb_entities: dict[str, gs.RigidEntity] = {}
    for name, layout in YCB_LAYOUT.items():
        asset = ycb_assets[name]
        x, y, _ = layout["pos"]
        z = TABLE_TOP_Z + asset.rest_z_offset
        ycb_entities[name] = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(asset.mesh_path),
                pos=(x, y, z),
                euler=layout["euler"],
                align=False,
                convexify=True,
                decimate_face_num=500,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=layout.get("friction")),
        )

    franka = scene.add_entity(
        gs.morphs.MJCF(
            file=str(assets / "robots" / "franka" / "panda.xml"),
            pos=FRANKA_POS,
            euler=FRANKA_EULER,
        ),
    )

    # Cameras must be added before scene.build().
    world_cam = None
    if add_world_cam:
        world_cam = scene.add_camera(
            res=WORLD_CAM_RES,
            pos=WORLD_CAM_POS,
            lookat=WORLD_CAM_LOOKAT,
            fov=WORLD_CAM_FOV,
            GUI=False,
        )

    wrist_cam = None
    wrist_link = None
    if add_wrist_cam:
        wrist_cam = scene.add_camera(
            res=WRIST_CAM_RES,
            fov=WRIST_CAM_FOV,
            GUI=False,
        )
        wrist_link = franka.get_link(WRIST_CAM_LINK)

    if n_envs > 1:
        scene.build(n_envs=n_envs, env_spacing=(1.5, 1.5))
    else:
        scene.build()

    configure_franka(franka, n_envs=n_envs)

    # Attaching needs the link's runtime pose, so it happens after build().
    if wrist_cam is not None:
        offset_T = trans_R_to_T(
            np.asarray(WRIST_CAM_OFFSET_POS, dtype=np.float64),
            euler_to_R(np.asarray(WRIST_CAM_OFFSET_EULER, dtype=np.float64)),
        )
        wrist_cam.attach(wrist_link, offset_T)
        wrist_cam.move_to_attach()

    # Debug world frame at the origin (X=red, Y=green, Z=blue) to help tune layout.
    if draw_world_frame:
        scene.draw_debug_frame(T=np.eye(4), axis_length=0.3, origin_size=0.02, axis_radius=0.01)

    return SceneBundle(
        scene=scene,
        franka=franka,
        ycb=ycb_entities,
        world_cam=world_cam,
        wrist_cam=wrist_cam,
        _wrist_link=wrist_link,
    )


def configure_franka(franka: gs.RigidEntity, *, n_envs: int) -> None:
    qpos = np.array(FRANKA_QPOS)
    if n_envs > 1:
        qpos = np.tile(qpos, (n_envs, 1))
    franka.set_qpos(qpos)
    franka.set_dofs_kp(np.array(FRANKA_KP))
    franka.set_dofs_kv(np.array(FRANKA_KV))
    franka.set_dofs_force_range(np.array(FRANKA_FORCE_MIN), np.array(FRANKA_FORCE_MAX))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and run the Franka manipulation scene.")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-n", "--n-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--setup-assets",
        action="store_true",
        help="Refresh YCB and robot symlinks before building.",
    )
    parser.add_argument("--no-world-cam", action="store_true", help="Disable the fixed world camera.")
    parser.add_argument("--no-wrist-cam", action="store_true", help="Disable the wrist camera.")
    parser.add_argument(
        "--debug-frame",
        action="store_true",
        help="Draw a world coordinate frame at the origin for layout debugging.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Render and save one frame from each camera at the end of the run.",
    )
    args = parser.parse_args()

    if args.setup_assets:
        setup_assets()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)
    bundle = build_scene(
        show_viewer=args.vis,
        n_envs=args.n_envs,
        add_world_cam=not args.no_world_cam,
        add_wrist_cam=not args.no_wrist_cam,
        draw_world_frame=args.debug_frame,
    )

    # Keep the arm at its initial pose so it does not droop under gravity.
    hold_qpos = np.array(FRANKA_QPOS)
    if args.n_envs > 1:
        hold_qpos = np.tile(hold_qpos, (args.n_envs, 1))

    for _ in range(args.steps):
        bundle.franka.control_dofs_position(hold_qpos)
        bundle.scene.step()
        bundle.update_wrist_cam()

    if args.save_frames:
        import imageio.v2 as imageio

        if bundle.world_cam is not None:
            imageio.imwrite("world_cam.png", bundle.world_cam.render(rgb=True)[0])
            print("Saved world_cam.png")
        if bundle.wrist_cam is not None:
            imageio.imwrite("wrist_cam.png", bundle.wrist_cam.render(rgb=True)[0])
            print("Saved wrist_cam.png")


if __name__ == "__main__":
    main()
