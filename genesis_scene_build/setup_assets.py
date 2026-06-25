"""Prepare local assets (symlinks) for the manipulation scene."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
YCB_SOURCE = ROOT.parent / "mani_skill_dataset"
FRANKA_SOURCE = ROOT.parent / "genesis" / "assets" / "xml" / "franka_emika_panda"

YCB_OBJECTS = (
    "003_cracker_box",
    # "006_mustard_bottle",
    "011_banana",
    "024_bowl",
    "025_mug",
)


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        if dst.is_symlink():
            dst.unlink()
        else:
            raise FileExistsError(f"Asset path already exists and is not a symlink: {dst}")
    dst.symlink_to(src.resolve())


def setup_assets() -> Path:
    ycb_dir = ASSETS / "ycb"
    robot_dir = ASSETS / "robots" / "franka"

    for name in YCB_OBJECTS:
        src = YCB_SOURCE / name
        if not src.is_dir():
            raise FileNotFoundError(f"Missing YCB asset directory: {src}")
        _symlink(src, ycb_dir / name)

    _symlink(FRANKA_SOURCE, robot_dir)
    return ASSETS


def main() -> None:
    assets_dir = setup_assets()
    print(f"Assets ready at: {assets_dir}")


if __name__ == "__main__":
    main()
