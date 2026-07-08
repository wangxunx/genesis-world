"""M5: sweep every checkpoint of a training run and plot a success-rate curve.

Builds the Genesis scene *once*, then loads each checkpoint in turn and evaluates
it with an identical, fixed protocol (same seeds -> same initial conditions), so
the resulting curve isolates the effect of training progress. Outputs:

  * ``<out-dir>/sweep.json``  -- full structured results (per checkpoint + per episode)
  * ``<out-dir>/sweep.csv``   -- one row per checkpoint (step, success_rate, ...)
  * ``<out-dir>/success_curve.png`` -- success rate vs training step

This is the base building block for the train->eval comparisons: fine-tuning
learning curves now, and later DR ablations (run the same sweep on each training
condition and overlay the curves).

Usage:
    uv run python genesis_scene_build/eval_sweep.py \
        --run-dir outputs/train/act_banana_pick_50ep \
        --repo-id genesis/banana_pick \
        --dataset-root genesis_scene_build/datasets/banana_pick_50ep \
        --episodes 20 --pick 011_banana
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import genesis as gs

from build_scene import build_scene
from eval_policy import evaluate_policy, load_policy


def discover_checkpoints(run_dir: Path, only_steps: list[int] | None = None) -> list[tuple[int, Path]]:
    """Return sorted (step, pretrained_model_dir) for a training run.

    Looks under ``<run_dir>/checkpoints/<step>/pretrained_model``. The ``last``
    symlink is skipped (it duplicates a numbered step). Non-numeric step dirs are
    kept with step=-1 so they still get evaluated (sorted first).
    """
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.is_dir():
        raise SystemExit(f"[sweep] no checkpoints dir at {ckpt_root}")

    found: dict[int, Path] = {}
    for child in sorted(ckpt_root.iterdir()):
        if child.is_symlink() or not child.is_dir():
            continue  # skip 'last' symlink and stray files
        pm = child / "pretrained_model"
        if not (pm / "config.json").exists():
            continue
        step = int(child.name) if child.name.isdigit() else -1
        if only_steps and step not in only_steps:
            continue
        found[step] = pm
    return sorted(found.items())


def plot_curve(rows: list[dict], out_png: Path, title: str) -> None:
    steps = [r["step"] for r in rows]
    rates = [r["success_rate"] * 100.0 for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, rates, marker="o", linewidth=2)
    for x, y in zip(steps, rates):
        plt.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    plt.xlabel("training step")
    plt.ylabel("success rate (%)")
    plt.title(title)
    plt.ylim(-2, 102)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep checkpoints -> success-rate curve (M5).")
    parser.add_argument("--run-dir", required=True, help="Training run dir (contains checkpoints/).")
    parser.add_argument("--repo-id", required=True, help="Dataset repo id the policy trained on.")
    parser.add_argument("--dataset-root", default=None, help="Local dataset dir (feature shapes/stats/fps).")
    parser.add_argument("--device", default="cuda", help="Inference device (cuda | cpu | mps).")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run the Genesis sim on CPU backend.")
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("--episodes", type=int, default=20, help="Eval episodes per checkpoint.")
    parser.add_argument("--seed", type=int, default=1000, help="Base eval seed (fixed across checkpoints).")
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument("--pick", nargs="+", default=["011_banana"], help="Object(s) to evaluate on.")
    parser.add_argument("--place", default="024_bowl")
    parser.add_argument("--tol", type=float, default=0.08)
    parser.add_argument("--no-task", action="store_true")
    parser.add_argument("--jitter", type=float, default=0.03)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Only evaluate these checkpoint steps (default: all found).",
    )
    parser.add_argument("--out-dir", default=None, help="Output dir (default: eval_results/sweep_<run-name>).")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoints = discover_checkpoints(run_dir, args.steps)
    if not checkpoints:
        raise SystemExit(f"[sweep] no usable checkpoints found under {run_dir}/checkpoints")
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "eval_results" / f"sweep_{run_dir.name}"

    print(f"[sweep] {len(checkpoints)} checkpoint(s): {[s for s, _ in checkpoints]}")

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)
    bundle = build_scene(show_viewer=args.vis, n_envs=1, add_world_cam=True, add_wrist_cam=True)

    rows: list[dict] = []
    per_ckpt_full: list[dict] = []
    for step, pm in checkpoints:
        print(f"\n[sweep] === checkpoint step={step} ({pm}) ===")
        pb = load_policy(str(pm), args.repo_id, args.dataset_root, args.device, use_amp=args.use_amp)
        res = evaluate_policy(
            bundle, pb,
            episodes=args.episodes,
            seed=args.seed,
            max_seconds=args.max_seconds,
            pick=args.pick,
            place=args.place,
            tol=args.tol,
            no_task=args.no_task,
            jitter=args.jitter,
            label=str(step),
        )
        rows.append(
            {
                "step": step,
                "success_rate": res["success_rate"],
                "n_success": res["n_success"],
                "episodes": res["episodes"],
                "policy_type": pb.policy_type,
            }
        )
        per_ckpt_full.append({"step": step, "checkpoint": str(pm), **res})

    out_dir.mkdir(parents=True, exist_ok=True)
    sweep = {
        "meta": {
            "run_dir": str(run_dir),
            "repo_id": args.repo_id,
            "episodes": args.episodes,
            "seed": args.seed,
            "pick": args.pick,
            "max_seconds": args.max_seconds,
            "jitter": args.jitter,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "checkpoints": per_ckpt_full,
    }
    with open(out_dir / "sweep.json", "w") as f:
        json.dump(sweep, f, indent=2)
    with open(out_dir / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "success_rate", "n_success", "episodes", "policy_type"])
        w.writeheader()
        w.writerows(rows)
    plot_curve(rows, out_dir / "success_curve.png", title=f"{run_dir.name}: success vs step")

    print("\n[sweep] summary:")
    for r in rows:
        print(f"  step {r['step']:>7}: {r['n_success']}/{r['episodes']} = {r['success_rate']:.1%}")
    print(f"\n[sweep] wrote: {out_dir}/sweep.json, sweep.csv, success_curve.png")


if __name__ == "__main__":
    main()
