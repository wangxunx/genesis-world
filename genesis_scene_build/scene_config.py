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
    "006_mustard_bottle": {"pos": (0.42, 0.04, 0.0), "euler": (0.0, 0.0, -20.0)},
    "011_banana": {"pos": (0.30, 0.20, 0.0), "euler": (0.0, 0.0, 35.0)},
    "024_bowl": {"pos": (0.52, -0.08, 0.0), "euler": (0.0, 0.0, 0.0)},
    "025_mug": {"pos": (0.22, 0.10, 0.0), "euler": (0.0, 0.0, 10.0)},
}

FRANKA_QPOS = (0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.79, 0.04, 0.04)
FRANKA_KP = (4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100)
FRANKA_KV = (450, 450, 350, 350, 200, 200, 200, 10, 10)
FRANKA_FORCE_MIN = (-87, -87, -87, -87, -12, -12, -12, -100, -100)
FRANKA_FORCE_MAX = (87, 87, 87, 87, 12, 12, 12, 100, 100)


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
