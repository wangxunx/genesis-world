#!/usr/bin/env python3
"""Stitch several clips into one showcase reel with title cards + transitions.

Each stage is rendered as `title card -> clip`, all normalized onto a common
canvas (aspect-preserving scale + letterbox pad) so clips of different sizes mix
cleanly. Consecutive pieces are joined with `xfade` cross-transitions and the
whole reel gets a fade in/out.

Reusable: pass any number of `--titles` / `--videos` (and optional
`--subtitles`) pairs. Requires `ffmpeg`/`ffprobe` on PATH.

Example
-------
    python -m franka_fruit_pick.tools.make_showcase_reel \
        --titles "① Record data" "② Domain randomization" "③ Policy eval" \
        --videos a.mp4 b.mp4 c.mp4 \
        --out outputs/pipeline_reel.mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

DEF_ROOT = "genesis_scene_build/eval_videos"
DEFAULT_TITLES = [
    "① Build scene",
    "② Scripted pick-and-place",
    "③ Record data",
    "④ Domain randomization",
    "⑤ Policy eval",
]
DEFAULT_SUBTITLES = [
    "Franka + table + YCB fruits · scripted pick-and-place scene",
    "IK-based reach → grasp → place · the scripted M2 policy",
    "world + wrist cameras · 3 fruits × 3 episodes",
    "table / object colors & camera FOV randomized per episode",
    "trained policy rollout · banana / lemon / plum",
]
DEFAULT_VIDEOS = [
    f"{DEF_ROOT}/dataset_showcase/m1_scene_orbit.mp4",
    f"{DEF_ROOT}/dataset_showcase/m2_scripted_pick.mp4",
    f"{DEF_ROOT}/dataset_showcase/no_dr_3x6.mp4",
    f"{DEF_ROOT}/dataset_showcase/dr_3x6.mp4",
    f"{DEF_ROOT}/fruit_pick_v2_aggregated/combined_3x3_banana_lemon_plum.mp4",
]


def find_font(bold: bool = True) -> str:
    names = (
        ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"] if bold else ["DejaVuSans.ttf"]
    )
    dirs = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"]
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    raise FileNotFoundError("DejaVu font not found; install fonts-dejavu")


def probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(out)


def esc(text: str) -> str:
    return text.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--titles", nargs="+", default=DEFAULT_TITLES)
    ap.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS)
    ap.add_argument("--subtitles", nargs="*", default=DEFAULT_SUBTITLES,
                    help="optional second line per title card ('' to skip a card)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--title-dur", type=float, default=1.8, help="title card seconds")
    ap.add_argument("--transition", type=float, default=0.6, help="xfade seconds")
    ap.add_argument("--transition-type", default="fade", help="xfade type (fade, dissolve, wipeleft, ...)")
    ap.add_argument("--bg", default="0x101418", help="background/letterbox color")
    ap.add_argument("--intro-title", default="Franka Fruit-Pick Demo", help="opening cover title")
    ap.add_argument("--intro-subtitle", default="scripted → learned fruit pick-and-place on Genesis",
                    help="opening cover subtitle ('' to omit)")
    ap.add_argument("--intro-dur", type=float, default=2.8, help="opening-card seconds")
    ap.add_argument("--no-intro", action="store_true", help="disable the opening cover card")
    ap.add_argument("--outro-url", default="https://github.com/wangxunx/franka_fruit_pick_demo",
                    help="repo/link shown on the closing card")
    ap.add_argument("--outro-heading", default="Reference Code on GitHub", help="closing-card heading")
    ap.add_argument("--outro-dur", type=float, default=2.8, help="closing-card seconds")
    ap.add_argument("--no-outro", action="store_true", help="disable the closing repo-link card")
    args = ap.parse_args()

    if len(args.titles) != len(args.videos):
        raise SystemExit(f"--titles ({len(args.titles)}) and --videos ({len(args.videos)}) must match")
    subs = list(args.subtitles) + [""] * (len(args.titles) - len(args.subtitles))

    W, H, fps, T = args.width, args.height, args.fps, args.transition
    font_bold, font_reg = find_font(True), find_font(False)

    # Build interleaved clip list: [intro,] title, video, title, video, ... [, outro]
    inputs: list[list[str]] = []
    durations: list[float] = []
    kinds: list[str] = []  # "intro" | "title" | "video" | "outro"
    metas: list[dict] = []

    # Opening cover card.
    if not args.no_intro:
        inputs.append(["-f", "lavfi", "-i",
                       f"color=c={args.bg}:s={W}x{H}:d={args.intro_dur}:r={fps}"])
        durations.append(args.intro_dur)
        kinds.append("intro")
        metas.append({"title": args.intro_title, "sub": args.intro_subtitle})

    for title, sub, vid in zip(args.titles, subs, args.videos):
        if not os.path.exists(vid):
            raise SystemExit(f"missing video: {vid}")
        # title card (lavfi color source)
        inputs.append(["-f", "lavfi", "-i",
                       f"color=c={args.bg}:s={W}x{H}:d={args.title_dur}:r={fps}"])
        durations.append(args.title_dur)
        kinds.append("title")
        metas.append({"title": title, "sub": sub})
        # video clip
        inputs.append(["-i", vid])
        durations.append(probe_duration(vid))
        kinds.append("video")
        metas.append({})

    # Closing card with the repo link.
    if not args.no_outro:
        inputs.append(["-f", "lavfi", "-i",
                       f"color=c={args.bg}:s={W}x{H}:d={args.outro_dur}:r={fps}"])
        durations.append(args.outro_dur)
        kinds.append("outro")
        metas.append({"heading": args.outro_heading, "url": args.outro_url})

    filt: list[str] = []
    for i, kind in enumerate(kinds):
        if kind == "intro":
            title, sub = metas[i]["title"], metas[i]["sub"]
            ty = "(h/2)-100" if sub else "(h-text_h)/2"
            parts = [
                f"drawtext=fontfile='{font_bold}':text='{esc(title)}':fontcolor=white:"
                f"fontsize=112:x=(w-text_w)/2:y={ty}"
            ]
            if sub:
                parts.append(
                    f"drawtext=fontfile='{font_reg}':text='{esc(sub)}':fontcolor=0xB8C0CC:"
                    f"fontsize=44:x=(w-text_w)/2:y=(h/2)+50"
                )
            parts.append(f"setsar=1,fps={fps},format=yuv420p")
            filt.append(f"[{i}:v]" + ",".join(parts) + f"[c{i}]")
        elif kind == "title":
            title, sub = metas[i]["title"], metas[i]["sub"]
            ty = "(h/2)-90" if sub else "(h-text_h)/2"
            parts = [
                f"drawtext=fontfile='{font_bold}':text='{esc(title)}':fontcolor=white:"
                f"fontsize=88:x=(w-text_w)/2:y={ty}"
            ]
            if sub:
                parts.append(
                    f"drawtext=fontfile='{font_reg}':text='{esc(sub)}':fontcolor=0xB8C0CC:"
                    f"fontsize=40:x=(w-text_w)/2:y=(h/2)+40"
                )
            parts.append(f"setsar=1,fps={fps},format=yuv420p")
            filt.append(f"[{i}:v]" + ",".join(parts) + f"[c{i}]")
        elif kind == "outro":
            heading, url = metas[i]["heading"], metas[i]["url"]
            parts = [
                f"drawtext=fontfile='{font_bold}':text='{esc(heading)}':fontcolor=white:"
                f"fontsize=64:x=(w-text_w)/2:y=(h/2)-70",
                # link-style URL in a rounded dark box
                f"drawtext=fontfile='{font_reg}':text='{esc(url)}':fontcolor=0x6AB0FF:"
                f"fontsize=46:x=(w-text_w)/2:y=(h/2)+20:box=1:boxcolor=0x0D1117@0.9:boxborderw=22",
                f"setsar=1,fps={fps},format=yuv420p",
            ]
            filt.append(f"[{i}:v]" + ",".join(parts) + f"[c{i}]")
        else:
            filt.append(
                f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:{args.bg},"
                f"setsar=1,fps={fps},format=yuv420p[c{i}]"
            )

    # Chain xfade transitions across all clips.
    prev = "c0"
    running = durations[0]
    for k in range(1, len(kinds)):
        off = running - T
        out = f"x{k}"
        filt.append(
            f"[{prev}][c{k}]xfade=transition={args.transition_type}:duration={T}:offset={off:.3f}[{out}]"
        )
        running = running + durations[k] - T
        prev = out

    total = running
    filt.append(f"[{prev}]fade=t=in:st=0:d=0.4,fade=t=out:st={total - 0.4:.3f}:d=0.4[out]")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y"]
    for inp in inputs:
        cmd += inp
    cmd += [
        "-filter_complex", ";".join(filt),
        "-map", "[out]",
        "-r", str(fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
        "-movflags", "+faststart",
        args.out,
    ]

    print(f"reel: {len(args.titles)} stages | canvas {W}x{H}@{fps} | transition {T}s ({args.transition_type})")
    print(f"total duration ~= {total:.2f}s")
    subprocess.run(cmd, check=True)
    print("saved", args.out)


if __name__ == "__main__":
    main()
