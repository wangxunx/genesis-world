"""Scene layout and asset configuration for the Franka manipulation scene."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import trimesh

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

# == Table (built from box primitives) ==
TABLE_TOP_Z = 0.75  # height of the tabletop surface (m)
TABLE_CENTER = (0.35, 0.0)  # (x, y) center of the tabletop
TABLE_TOP_SIZE = (1.20, 0.80, 0.05)  # tabletop (length_x, width_y, thickness_z)
TABLE_LEG_SIZE = (0.06, 0.06)  # leg cross-section (x, y)
TABLE_LEG_INSET = 0.10  # how far legs are inset from the tabletop edges
TABLE_COLOR = (0.62, 0.47, 0.35, 1.0)
TABLE_LEG_COLOR = (0.47, 0.35, 0.24, 1.0)

# == Franka base, mounted on the tabletop near the rear (-x) edge, facing +x ==
FRANKA_POS = (-0.10, 0.0, TABLE_TOP_Z)
FRANKA_EULER = (0.0, 0.0, 0.0)

# == YCB object xy positions on the tabletop (z filled in at build time) ==
YCB_LAYOUT = {
    "003_cracker_box": {"pos": (0.28, -0.18, 0.0), "euler": (0.0, 0.0, 15.0)},
    # "006_mustard_bottle": {"pos": (0.42, 0.04, 0.0), "euler": (0.0, 0.0, -20.0)},
    "011_banana": {"pos": (0.30, 0.20, 0.0), "euler": (0.0, 0.0, 35.0)},
    "024_bowl": {"pos": (0.52, -0.08, 0.0), "euler": (0.0, 0.0, 0.0)},
    "025_mug": {"pos": (0.22, 0.10, 0.0), "euler": (0.0, 0.0, 10.0)},
}

FRANKA_QPOS = (0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.79, 0.04, 0.04)
FRANKA_KP = (4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100)
FRANKA_KV = (450, 450, 350, 350, 200, 200, 200, 10, 10)
FRANKA_FORCE_MIN = (-87, -87, -87, -87, -12, -12, -12, -100, -100)
FRANKA_FORCE_MAX = (87, 87, 87, 87, 12, 12, 12, 100, 100)

# == Cameras ==
# World camera: a fixed third-person view of the tabletop. Intrinsics are matched to
# the Intel RealSense D435i RGB (color) module, same as the wrist camera: FOV
# 69 deg (H) x 42 deg (V) at native 16:9. Using vfov = 42 deg at 1280x720 gives
# H-FOV ~= 68.6 deg and intrinsics fx = fy ~= 938 px, cx = 640, cy = 360. Only the
# extrinsics (pos / lookat) differ from the wrist camera.
WORLD_CAM_RES = (1280, 720)
# Placed 1 m directly above the table corner closest to the viewer (+x, -y corner).
#   x = 0.35 + 1.20/2 = 0.95,  y = 0.0 - 0.80/2 = -0.40,  z = 0.75 + 1.0 = 1.75
WORLD_CAM_POS = (
    TABLE_CENTER[0] + TABLE_TOP_SIZE[0] / 2,
    TABLE_CENTER[1] - TABLE_TOP_SIZE[1] / 2,
    TABLE_TOP_Z + 1.0,
)
WORLD_CAM_LOOKAT = (TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP_Z)
WORLD_CAM_FOV = 42  # vertical FOV in degrees (D435i RGB module)

# Wrist camera: mounted on the Franka hand link (eye-in-hand), looking toward the
# grasp area. Parameters are matched to a real Intel RealSense D435i.
#
# Genesis derives a simple pinhole intrinsic purely from (resolution, vertical FOV):
#     fx = fy = 0.5 * height / tan(vfov / 2),  cx = width / 2,  cy = height / 2
# so matching D435i means picking the right resolution aspect ratio + vertical FOV.
#
# Used as an RGB-only camera (depth channel ignored), matched to the D435i *color*
# module spec: FOV 69 deg (H) x 42 deg (V), native 16:9 sensor.
# Using vfov = 42 deg at 1280x720 reproduces the horizontal FOV automatically:
#     H-FOV = 2 * atan((1280/720) * tan(21 deg)) ~= 68.6 deg  (~= 69 deg).
# Resulting intrinsics @ 1280x720: fx = fy ~= 938 px, cx = 640, cy = 360,
# which closely matches a real D435i color stream at this resolution.
WRIST_CAM_RES = (1280, 720)
WRIST_CAM_FOV = 42  # vertical FOV in degrees (D435i RGB module)
WRIST_CAM_LINK = "hand"
# Camera pose relative to the hand link frame. On the Franka, the hand link +z
# points along the gripper approach direction (toward the fingertips). The camera's
# optical axis is its local -z, so a 180 deg rotation about x makes -z align with
# the hand +z, i.e. the camera looks forward along the approach direction. The
# position offset sits the camera slightly behind the hand origin so the fingers
# stay in view.
WRIST_CAM_OFFSET_POS = (0.05, 0.0, -0.03)
WRIST_CAM_OFFSET_EULER = (180.0, 0.0, 0.0)


@dataclass(frozen=True)
class YCBAsset:
    name: str
    mesh_path: Path
    collision_path: Path
    rest_z_offset: float


def _mesh_rest_z_offset(mesh_path: Path) -> float:
    mesh = trimesh.load(mesh_path, force="mesh")
    return float(-mesh.bounds[0][2])


def get_ycb_assets() -> dict[str, YCBAsset]:
    assets: dict[str, YCBAsset] = {}
    for name in YCB_LAYOUT:
        mesh_path = ASSETS / "ycb" / name / "textured.obj"
        collision_path = ASSETS / "ycb" / name / "collision.ply"
        assets[name] = YCBAsset(
            name=name,
            mesh_path=mesh_path,
            collision_path=collision_path,
            rest_z_offset=_mesh_rest_z_offset(mesh_path),
        )
    return assets
