#!/usr/bin/env python3
"""Build a grid showcase video from recorded LeRobot datasets.

For each dataset (one grid row) it picks N episodes and, per episode, places the
`world` and `wrist` camera clips side by side (1x2). With N=3 episodes that is a
1x6 row; stacking several datasets gives an ROWSx(N*2) grid video.

The clips are read straight from the concatenated per-camera mp4 files using the
episode boundaries stored in `meta/episodes/*.parquet`, trimmed to a common
duration so every tile stays frame-aligned.

Reusable: point `--datasets` at the non-DR datasets or the `*_dr` ones to get the
matching showcase. Requires `ffmpeg`/`ffprobe` on PATH and `pandas`/`pyarrow`.

Example
-------
    python -m franka_fruit_pick.tools.make_dataset_showcase \
        --datasets .../banana_pick_v3 .../lemon_pick_v3 .../plum_pick_v3 \
        --out outputs/dataset_showcase_no_dr_3x6.mp4
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
from pathlib import Path

import pandas as pd

WORLD = "videos/observation.images.world"
WRIST = "videos/observation.images.wrist"


def _find_font() -> str | None:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(p):
            return p
    return None


def load_meta(dataset: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(dataset / "meta/episodes/**/*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no episode metadata under {dataset}/meta/episodes")
    df = pd.concat([pd.read_parquet(f) for f in files])
    return df.sort_values("episode_index").reset_index(drop=True)


def cam_clip_path(dataset: Path, cam: str, row) -> Path:
    ci = int(row[f"{cam}/chunk_index"])
    fi = int(row[f"{cam}/file_index"])
    return dataset / cam / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"


def pick_episodes(n: int, k: int) -> list[int]:
    if k >= n:
        return list(range(n))
    # evenly spread across the dataset for visual variety
    return [round(i * (n - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]


def clean_label(dataset: Path) -> str:
    return dataset.name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True, help="dataset directories (one grid row each)")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--episodes", nargs="+", default=None,
                    help="episode indices to use. Provide ONE comma-list per dataset for distinct picks per row "
                         "(e.g. '0,33,66' '11,44,77' '22,55,88'), or a single comma-list applied to every "
                         "dataset. Default: auto evenly-spaced per dataset.")
    ap.add_argument("--num-episodes", type=int, default=3, help="episodes per dataset when auto-picking")
    ap.add_argument("--cell-width", type=int, default=480)
    ap.add_argument("--cell-height", type=int, default=270)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=None, help="override per-clip seconds (default: min episode len)")
    ap.add_argument("--no-labels", action="store_true", help="disable drawtext labels")
    args = ap.parse_args()

    datasets = [Path(d) for d in args.datasets]

    # --episodes: one comma-group per dataset (distinct picks per row), a single group applied
    # to all, or None (auto). Parsed into `ep_groups` aligned with `datasets`.
    ep_groups: list[list[int]] | None = None
    if args.episodes:
        parsed = [[int(x) for x in grp.split(",")] for grp in args.episodes]
        if len(parsed) == 1:
            ep_groups = parsed * len(datasets)
        elif len(parsed) == len(datasets):
            ep_groups = parsed
        else:
            raise SystemExit(
                f"--episodes: give 1 group or one per dataset ({len(datasets)}), got {len(parsed)}"
            )

    # Gather per-cell clip specs: rows x (k episodes) x {world, wrist}
    rows: list[dict] = []
    min_dur = float("inf")
    for di, ds in enumerate(datasets):
        meta = load_meta(ds)
        n = len(meta)
        eps = ep_groups[di] if ep_groups is not None else pick_episodes(n, args.num_episodes)
        eps = [e for e in eps if e < n]
        cells = []
        for ep in eps:
            r = meta[meta.episode_index == ep].iloc[0]
            w_start = float(r[f"{WORLD}/from_timestamp"])
            w_dur = float(r[f"{WORLD}/to_timestamp"]) - w_start
            min_dur = min(min_dur, w_dur)
            cells.append({
                "ep": ep,
                "world": (cam_clip_path(ds, WORLD, r), w_start),
                "wrist": (cam_clip_path(ds, WRIST, r), float(r[f"{WRIST}/from_timestamp"])),
            })
        rows.append({"label": clean_label(ds), "eps": eps, "cells": cells})

    ncols = len(rows[0]["cells"]) * 2
    for row in rows:
        if len(row["cells"]) * 2 != ncols:
            raise ValueError("all datasets must yield the same number of episodes")

    dur = args.duration if args.duration is not None else round(min_dur, 3)
    cw, ch = args.cell_width, args.cell_height
    font = None if args.no_labels else _find_font()

    # Build ffmpeg inputs (world then wrist for each cell, row-major).
    inputs: list[list[str]] = []
    tiles: list[tuple[str, str]] = []  # (input_label_suffix, drawtext)
    for row in rows:
        for cell in row["cells"]:
            for cam in ("world", "wrist"):
                path, start = cell[cam]
                inputs.append(["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(path)])
                tiles.append((f"{row['label']} ep{cell['ep']} {cam}", cam))

    filt = []
    row_len = ncols  # tiles per row
    for i, (label_txt, cam) in enumerate(tiles):
        chain = f"[{i}:v]scale={cw}:{ch},setsar=1,fps={args.fps}"
        if font:
            txt = label_txt.replace(":", r"\:")
            chain += (
                f",drawtext=fontfile='{font}':text='{txt}':x=6:y=6:fontsize=14:"
                f"fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=4"
            )
        chain += f"[v{i}]"
        filt.append(chain)

    row_labels = []
    for r in range(len(rows)):
        ins = "".join(f"[v{r * row_len + c}]" for c in range(row_len))
        filt.append(f"{ins}hstack=inputs={row_len}[row{r}]")
        row_labels.append(f"[row{r}]")
    filt.append(f"{''.join(row_labels)}vstack=inputs={len(rows)}[out]")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y"]
    for inp in inputs:
        cmd += inp
    cmd += [
        "-filter_complex", ";".join(filt),
        "-map", "[out]",
        "-r", str(args.fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
        args.out,
    ]

    print(f"grid: {len(rows)} rows x {ncols} cols | cell {cw}x{ch} | clip {dur:.2f}s @ {args.fps}fps")
    for row in rows:
        print(f"  {row['label']}: episodes {row['eps']}")
    subprocess.run(cmd, check=True)
    print("saved", args.out)


if __name__ == "__main__":
    main()
