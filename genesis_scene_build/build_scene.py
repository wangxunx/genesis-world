"""Build a Franka manipulation scene with a table and YCB objects."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import genesis as gs

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
    YCB_LAYOUT,
    get_ycb_assets,
)
from setup_assets import setup_assets


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
    add_camera: bool = False,
) -> tuple[gs.Scene, gs.RigidEntity, dict[str, gs.RigidEntity]]:
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
            material=gs.materials.Rigid(rho=300.0),
        )

    franka = scene.add_entity(
        gs.morphs.MJCF(
            file=str(assets / "robots" / "franka" / "panda.xml"),
            pos=FRANKA_POS,
            euler=FRANKA_EULER,
        ),
    )

    camera = None
    if add_camera:
        camera = scene.add_camera(
            res=(960, 720),
            pos=(cx + 1.0, -1.2, 1.5),
            lookat=camera_lookat,
            fov=45,
            GUI=False,
        )

    if n_envs > 1:
        scene.build(n_envs=n_envs, env_spacing=(1.5, 1.5))
    else:
        scene.build()

    configure_franka(franka, n_envs=n_envs)
    if add_camera:
        return scene, franka, ycb_entities, camera
    return scene, franka, ycb_entities


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
    args = parser.parse_args()

    if args.setup_assets:
        setup_assets()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)
    scene, franka, _ = build_scene(show_viewer=args.vis, n_envs=args.n_envs)

    # Keep the arm at its initial pose so it does not droop under gravity.
    hold_qpos = np.array(FRANKA_QPOS)
    if args.n_envs > 1:
        hold_qpos = np.tile(hold_qpos, (args.n_envs, 1))

    for _ in range(args.steps):
        franka.control_dofs_position(hold_qpos)
        scene.step()


if __name__ == "__main__":
    main()
