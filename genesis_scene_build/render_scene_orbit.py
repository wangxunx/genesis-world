#!/usr/bin/env python3
"""M1 showcase: render a 360° turntable orbit of the built scene.

Builds the Franka + table + YCB scene, lets the objects settle, then sweeps a
third-person camera in a full circle around the tabletop center while the arm
holds its initial pose -- an at-a-glance "here is the scene we built" clip.

Reuses the cosmetic `video_cam` from `build_scene` (1280x720). Output loops
seamlessly (last frame ~= first) so it can be trimmed/looped freely.

Example
-------
    python -m franka_fruit_pick.tools.render_scene_orbit \
        --out outputs/m1_scene_orbit.mp4 --duration 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Package dir on sys.path so flat sibling imports (build_scene, scene_config) resolve.
_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import genesis as gs  # noqa: E402

from build_scene import build_scene, SceneDomainRandomizationConfig  # noqa: E402
from scene_config import FRANKA_QPOS, TABLE_CENTER, TABLE_TOP_Z  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/m1_scene_orbit.mp4")
    ap.add_argument("--duration", type=float, default=5.0, help="clip length in seconds")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--radius", type=float, default=1.3, help="orbit radius around table center (m)")
    ap.add_argument("--cam-height", type=float, default=0.95, help="camera height above the tabletop (m)")
    ap.add_argument("--lookat-z", type=float, default=0.02, help="lookat height above the tabletop (m)")
    ap.add_argument("--start-deg", type=float, default=-90.0,
                    help="starting azimuth (deg); -90 = front (-y) edge, looking down at the tabletop")
    ap.add_argument("--sweep-deg", type=float, default=210.0,
                    help="signed arc length in degrees (+ = CCW). ~210 = 'big half circle' away from the "
                         "robot; use +/-360 for a full seamless turntable. Eased in/out unless it is a full loop.")
    ap.add_argument("--settle-steps", type=int, default=60, help="physics steps to settle objects before orbit")
    ap.add_argument("--cpu", action="store_true")
    # optional M4 Layer-A appearance DR (off by default -> baseline look)
    ap.add_argument("--dr-appearance", action="store_true")
    ap.add_argument("--dr-table-jitter", type=float, default=0.15)
    ap.add_argument("--dr-object-color", action="store_true")
    ap.add_argument("--dr-fov-jitter", type=float, default=0.0)
    ap.add_argument("--dr-seed", type=int, default=None)
    args = ap.parse_args()

    import imageio.v2 as imageio

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    scene_dr = SceneDomainRandomizationConfig(
        enabled=args.dr_appearance,
        table_color_jitter=args.dr_table_jitter,
        randomize_object_color=args.dr_object_color,
        fov_jitter_deg=args.dr_fov_jitter,
        seed=args.dr_seed,
    )

    bundle = build_scene(
        show_viewer=False,
        add_world_cam=False,
        add_wrist_cam=False,
        add_video_cam=True,
        scene_dr=scene_dr,
    )
    cam = bundle.video_cam
    assert cam is not None, "video_cam was not created"

    cx, cy = TABLE_CENTER
    lookat = (cx, cy, TABLE_TOP_Z + args.lookat_z)
    cam_z = TABLE_TOP_Z + args.cam_height

    # Hold the arm at its initial pose and let the objects settle on the table.
    hold_qpos = np.array(FRANKA_QPOS)
    for _ in range(args.settle_steps):
        bundle.franka.control_dofs_position(hold_qpos)
        bundle.scene.step()

    n_frames = max(1, int(round(args.duration * args.fps)))

    # Sweep an arc of `sweep_deg` starting at `start_deg`. A world-up of +z is pinned on
    # every frame so the horizon never rolls/flips as the camera goes around. A full-circle
    # sweep (|sweep| >= 360) runs at constant speed and loops seamlessly; a partial arc is
    # eased in/out (smootherstep) so the pan starts and stops gently.
    start = np.deg2rad(args.start_deg)
    span = np.deg2rad(args.sweep_deg)
    full_loop = abs(args.sweep_deg) >= 360.0 - 1e-6

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "full turntable (seamless)" if full_loop else f"eased {args.sweep_deg:.0f} deg arc"
    print(f"orbit: {n_frames} frames | {mode} over {args.duration}s @ {args.fps}fps")
    print(f"radius={args.radius} m, cam_z={cam_z:.2f} m, lookat={lookat}")

    writer = imageio.get_writer(str(out_path), fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=16)
    try:
        for i in range(n_frames):
            if full_loop:
                t = i / n_frames  # exclusive end -> seamless loop
            else:
                t = i / max(1, n_frames - 1)  # inclusive end -> full arc
                t = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)  # smootherstep ease-in/out
            ang = start + span * t
            pos = (cx + args.radius * np.cos(ang), cy + args.radius * np.sin(ang), cam_z)
            cam.set_pose(pos=pos, lookat=lookat, up=(0.0, 0.0, 1.0))
            rgb = cam.render(rgb=True)[0]
            writer.append_data(np.asarray(rgb))
    finally:
        writer.close()

    print("saved", out_path)


if __name__ == "__main__":
    main()
