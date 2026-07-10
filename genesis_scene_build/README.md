# genesis_scene_build

A small Franka pick-and-place pipeline built on [Genesis](https://github.com/Genesis-Embodied-AI/Genesis). It builds a tabletop scene with a Franka arm, YCB objects, and two RealSense-style cameras, runs a scripted pick-and-place, randomizes the task per episode, and records successful episodes into a [LeRobot](https://github.com/huggingface/lerobot) dataset for imitation learning.

## Contents

| File | Role |
|------|------|
| `scene_config.py` | Central config: table geometry, object layout, camera intrinsics/extrinsics, Franka gains, reachable workspace, and per-object appearance priors (`DR_APPEARANCE_PRIORS`) for domain randomization. |
| `build_scene.py` | Builds the Genesis scene (table + Franka + YCB objects + cameras) and returns a `SceneBundle`. Applies M4 Layer-A domain randomization (`SceneDomainRandomizationConfig`). |
| `grasp_demo.py` | Scripted pick-and-place state machine (IK + motion planning + force-controlled grasp), parameterized by `TaskSpec` / `GraspProfile`. |
| `randomize.py` | Per-episode environment randomization (`EnvRandomizer.reset()`): object poses + task sampling. |
| `record_dataset.py` | Records scripted episodes into a LeRobot dataset (success-filtered); can rebuild per appearance domain for a DR dataset. |
| `train_policy.py` | Thin wrapper around `lerobot-train` (ACT / SmolVLA presets) wired to this project's conventions. |
| `eval_policy.py` | Policy-agnostic closed-loop evaluation of a trained lerobot policy; reports success rate. |
| `eval_sweep.py` | Sweeps every checkpoint of a run into a success-vs-step curve; optionally across M4 appearance domains. |
| `dr_preview.py` | Renders the scene under several Layer-A appearance domains and stitches a labeled montage for visual inspection. |
| `setup_assets.py` | Verifies/populates local assets (YCB meshes + Franka model). Runs automatically on first build. |
| `scale_ycb.py` | Utility to bake scaled-down copies of YCB meshes (geometry only; textures preserved). |

## Setup

From the repo root:

```bash
uv sync
```

Install PyTorch for your hardware. This project runs on an **AMD Radeon (ROCm)** GPU, using the prebuilt ROCm wheels:

```bash
# Download the ROCm 7.2.1 wheels (Python 3.12)
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl

# Install into the project venv (replace any existing torch stack)
uv pip uninstall torch torchvision triton torchaudio
uv pip install \
    torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl \
    torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl \
    torchaudio-2.9.0+rocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl \
    triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl
```

> For other hardware, install the matching PyTorch build instead (e.g. `uv pip install torch --index-url https://download.pytorch.org/whl/cu126` for CUDA, or the default CPU wheel).

Assets are populated automatically the first time a scene is built (via `setup_assets.py`). To do it explicitly:

```bash
uv run python genesis_scene_build/setup_assets.py
```

> **Tip:** `uv run <script>` re-syncs the environment on every call. To skip that, call the venv Python directly: `./.venv/bin/python genesis_scene_build/<script>.py ...`

## The scene

- **Table**: built from box primitives; top surface at `z = 0.75 m`.
- **Franka**: mounted near the rear edge, facing `+x`. Held at its home pose with PD control so it doesn't droop under gravity.
- **Objects** (active set): `011_banana`, `014_lemon`, `018_plum` (pickable) + `024_bowl` (place container). Other YCB objects are disabled in `scene_config.py` (uncomment to re-enable as distractors).
- **Cameras**: a fixed `world` camera and a wrist-mounted `wrist` camera, both matched to an Intel RealSense D435i RGB module (1280x720, 42° vertical FOV).

## Usage

### 1. Build / preview the scene

```bash
# Headless smoke test (GPU by default; add --cpu for CPU)
uv run python genesis_scene_build/build_scene.py --steps 200

# With viewer, and save one frame from each camera
uv run python genesis_scene_build/build_scene.py --vis --save-frames
```

Key flags: `--cpu`, `--vis`, `--n-envs N`, `--steps N`, `--no-world-cam`, `--no-wrist-cam`, `--debug-frame`, `--setup-assets`.

To build under M4 Layer-A domain randomization (baked at build time), add `--dr-appearance` plus any of `--dr-table-jitter <amp>`, `--dr-object-color`, `--dr-fov-jitter <deg>`, `--dr-seed <n>` — see [Domain randomization](#domain-randomization-m4) below.

### 2. Scripted pick-and-place (`grasp_demo.py`)

Runs a single hardcoded-strategy pick-and-place and prints whether it succeeded.

```bash
# Pick the banana, place it into the bowl (defaults)
uv run python genesis_scene_build/grasp_demo.py --cpu

# Pick a different object; place onto tabletop coordinates instead of the bowl
uv run python genesis_scene_build/grasp_demo.py --cpu --pick 018_plum --place 0.5,0.1

# Save a world-camera frame at each stage into grasp_demo_frames/
uv run python genesis_scene_build/grasp_demo.py --cpu --save-frames
```

Flags: `--pick <object>`, `--place <object|x,y>`, `--tol <m>`, `--vis`, `--cpu`, `--save-frames`.

A task is a `TaskSpec(pick_object, place_target, success_tol)`; per-object grasp behavior lives in `GRASP_PROFILES`.

### 3. Randomized episodes (`randomize.py`)

Runs several episodes, each with randomized object poses (small position jitter + yaw jitter around home slots) and a sampled task.

```bash
uv run python genesis_scene_build/randomize.py --cpu -n 8 --seed 10
```

Flags: `-n/--episodes`, `--seed`, `--jitter <m>`, `--vis`, `--cpu`.

Randomization is intentionally small-amplitude ("start small"): non-overlap is guaranteed by a slot method (each object keeps its home slot; jitter is clamped to a per-object safe radius). Pick objects are drawn from `RELIABLE_PICK_POOL`.

### 4. Record a LeRobot dataset (`record_dataset.py`)

Runs randomized scripted episodes and writes **only successful** ones into a LeRobot dataset.

```bash
# Banana (most reliable)
uv run python genesis_scene_build/record_dataset.py --episodes 50 \
    --root genesis_scene_build/datasets/banana_pick_50ep

# Lemon or plum (per-object dataset). The task string is set automatically per object.
uv run python genesis_scene_build/record_dataset.py --pick 014_lemon --episodes 50 \
    --repo-id genesis/lemon_pick --root genesis_scene_build/datasets/lemon_pick_50ep
uv run python genesis_scene_build/record_dataset.py --pick 018_plum --episodes 50 \
    --repo-id genesis/plum_pick --root genesis_scene_build/datasets/plum_pick_50ep

# Mixed multi-object dataset: one object sampled per episode, with a matching task label
uv run python genesis_scene_build/record_dataset.py \
    --pick 011_banana 014_lemon 018_plum --episodes 90 \
    --repo-id genesis/fruit_pick --root genesis_scene_build/datasets/fruit_pick_90ep
```

`--pick` accepts one or more object names. With several, each episode randomly samples one of them, and the natural-language `task` field is set accordingly (e.g. `"pick the lemon and place it in the bowl"`). Since `lemon`/`plum` grasps succeed only ~50% of the time, expect roughly 2x as many attempts as episodes; raise `--max-attempts` if needed (default is 5x episodes).

Common flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--episodes` | 10 | Number of **successful** episodes to record. |
| `--max-attempts` | 5x episodes | Cap on total attempts. |
| `--pick` | `011_banana` | Object(s) to pick; one or more names (one sampled per episode). |
| `--fps` | 30 | Recording rate (decimated from the 100 Hz sim). |
| `--img-width/--img-height` | 640 / 360 | Recorded image size. |
| `--repo-id` | `genesis/banana_pick` | LeRobot repo id. |
| `--root` | `datasets/<repo-name>` | Output directory. |
| `--vcodec` | `libsvtav1` | Video codec. |
| `--overwrite` | off | Delete an existing output dir first. |
| `--keep-failures` | off | Debug: also save failed episodes (see below). |
| `--dr-appearance` | off | Enable M4 Layer-A DR: rebuild the scene with a new appearance domain every `--dr-rebuild-every` successful episodes. |
| `--dr-rebuild-every` | 5 | Successful episodes per appearance domain before rebuilding. |
| `--dr-table-jitter` / `--dr-object-color` / `--dr-fov-jitter` | 0.15 / off / 0.0 | Layer-A knobs (see [Domain randomization](#domain-randomization-m4)). |
| `--dr-seed` | `--seed` | Base seed for the appearance-domain sequence (domain `d` uses `base + d`). |

**Debugging failures** (`--keep-failures`): normally only successful episodes are written. With this flag, failed attempts are *also* saved, with their `task` label prefixed by `FAILED: ` (e.g. `"FAILED: pick the lemon and place it in the bowl"`). This lets you watch the failure videos to understand what went wrong. Filter them out when training, e.g. keep only episodes whose task does not start with `FAILED:`.

```bash
uv run python genesis_scene_build/record_dataset.py --pick 014_lemon --episodes 5 \
    --keep-failures --root genesis_scene_build/datasets/lemon_debug
```

**Appearance-randomized dataset** (`--dr-appearance`): the scene appearance is baked at build time, so a new appearance domain requires a rebuild. `--dr-rebuild-every N` advances to a new domain every `N` successful episodes; within a domain the M2 per-episode object-pose jitter still varies. So 50 episodes with `--dr-rebuild-every 5` yields ~10 appearance domains. See [Domain randomization](#domain-randomization-m4).

```bash
uv run python genesis_scene_build/record_dataset.py --episodes 50 \
    --dr-appearance --dr-object-color --dr-table-jitter 0.15 --dr-rebuild-every 5 \
    --repo-id genesis/banana_pick_dr --root genesis_scene_build/datasets/banana_pick_dr
```

**Recorded schema** (joint-position action space):

- `observation.state` — 9-D Franka joint positions (7 arm + 2 gripper)
- `action` — 9-D commanded joint targets
- `observation.images.world`, `observation.images.wrist` — RGB video
- plus the standard LeRobot fields (`timestamp`, `frame_index`, `episode_index`, `index`, `task_index`, `task`)

## Reading the recorded dataset

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    repo_id="genesis/banana_pick",
    root="genesis_scene_build/datasets/banana_pick_50ep",
    video_backend="pyav",   # see note below
)
print(ds.num_episodes, ds.num_frames, ds.fps)
sample = ds[0]              # dict of tensors for frame 0
```

Notes:

- The dataset uses the **LeRobot v3.0** on-disk format: multiple episodes are concatenated into shared `data/.../file-000.parquet` and per-camera `videos/.../file-000.mp4` files. Episode boundaries are stored in `meta/episodes/`. One mp4 therefore contains several grasps back-to-back — this is expected; `ds[i]` still returns discrete per-episode frames.
- Use `video_backend="pyav"` when reading. The default `torchcodec` backend requires system FFmpeg shared libraries (`libavutil.so.*`); if those are missing, decoding fails while `pyav` (bundled with `av`) works out of the box.

## Train a policy (`train_policy.py`)

`train_policy.py` is a thin wrapper around lerobot's own `lerobot-train`, wired to this project's directory conventions (datasets under `datasets/<name>`, runs under `outputs/train/<job-name>`). It does **not** reimplement training — it just assembles the right CLI. Built-in presets: `act` (trained from scratch) and `smolvla` (fine-tuned from `lerobot/smolvla_base`).

```bash
# ACT on the 50-episode banana dataset
uv run python genesis_scene_build/train_policy.py act \
    --repo-id genesis/banana_pick \
    --dataset-root genesis_scene_build/datasets/banana_pick_50ep \
    --steps 20000

# SmolVLA (fine-tune from the pretrained base)
uv run python genesis_scene_build/train_policy.py smolvla \
    --repo-id genesis/banana_pick \
    --dataset-root genesis_scene_build/datasets/banana_pick_50ep \
    --steps 20000
```

The run is written to `outputs/train/<policy>_<dataset-dir>/` (e.g. `outputs/train/act_banana_pick_50ep/`); the checkpoint at `checkpoints/last/pretrained_model` is exactly what `eval_policy.py --policy-path` expects.

Useful flags: `--steps`, `--batch-size` (defaults per preset: ACT 8, SmolVLA 4), `--save-freq`, `--device`, `--wandb`, `--dry-run` (print the command without running). The dataset decoder defaults to `--video-backend pyav`: lerobot's default `torchcodec` needs system FFmpeg shared libraries (`libavutil.so.*`) and is ABI-tied to the torch build, which is unreliable on this ROCm setup; `pyav` works out of the box. (Note: the pip `ffmpeg` / `imageio-ffmpeg` packages do **not** provide those shared libraries.) Any lerobot flags after `--` are forwarded verbatim, e.g.:

```bash
uv run python genesis_scene_build/train_policy.py act --repo-id genesis/banana_pick -- \
    --policy.optimizer_lr=1e-4 --policy.chunk_size=50
```

To train an arbitrary lerobot policy instead of a preset, use `--policy-type <t>` (from scratch) or `--policy-path <ckpt>` (fine-tune).

## Evaluate a trained policy (`eval_policy.py`)

`eval_policy.py` runs a **trained lerobot policy** closed-loop in the same Genesis scene and reports a success rate. It is **policy-agnostic**: ACT, SmolVLA (or any other lerobot policy) are loaded through the same generic path, so switching policy only means pointing `--policy-path` at a different checkpoint.

```bash
# ACT
uv run python genesis_scene_build/eval_policy.py \
    --policy-path outputs/train/act_banana_pick_50ep/checkpoints/last/pretrained_model \
    --repo-id genesis/banana_pick \
    --dataset-root genesis_scene_build/datasets/banana_pick_50ep \
    --episodes 20 --pick 011_banana

# SmolVLA — identical invocation, only the checkpoint changes
uv run python genesis_scene_build/eval_policy.py \
    --policy-path outputs/train/smolvla_banana_pick_50ep/checkpoints/last/pretrained_model \
    --repo-id genesis/banana_pick \
    --dataset-root genesis_scene_build/datasets/banana_pick_50ep \
    --episodes 20 --save-video
```

How it works (thin layers, no per-policy branching):

1. **Policy** — `PreTrainedConfig.from_pretrained` + `make_policy` + `make_pre_post_processors` load the checkpoint. The dataset metadata (`--repo-id`/`--dataset-root`) supplies feature shapes, normalization stats and fps; no episodes are read.
2. **Observation** — each step reads `observation.state` and the camera images into the exact dataset feature layout (raw numpy); lerobot's `predict_action` tensorizes/normalizes them.
3. **Action** — the policy emits the dataset's 9-D joint-target vector, which position-controls the arm + gripper.
4. **Rollout** — the policy is queried at its fps while the sim runs at 100 Hz (fractional decimation); each episode uses the same M2 randomizer, and success reuses `check_success` from `grasp_demo.py`.

The language `task` string (e.g. `"pick the banana and place it in the bowl"`) is always passed; ACT ignores it, VLAs use it. Pass `--no-task` to force an empty string.

Common flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--policy-path` | (required) | Checkpoint dir (`config.json` + `model.safetensors`). |
| `--repo-id` | (required) | Repo id of the dataset the policy trained on. |
| `--dataset-root` | HF cache | Local dataset dir (feature shapes / stats / fps). |
| `--episodes` | 10 | Number of eval rollouts. |
| `--pick` | `011_banana` | Object(s) to evaluate; one sampled per episode. |
| `--max-seconds` | 15 | Max sim time per episode before giving up. |
| `--device` | `cuda` | Inference device (auto-falls back). |
| `--save-video` | off | Write each rollout's world-cam video to `eval_videos/<repo>/`. |
| `--seed` | 1000 | Base RNG seed (offset from training seeds so eval poses differ). |
| `--results-out` | `eval_results/<repo>/<checkpoint>.json` | Structured results JSON (`none` to skip). |

Every eval writes a structured results JSON (`meta` + aggregate `success_rate` + `per_object` + per-episode detail), so runs can be compared later rather than only read off the console.

## Sweep checkpoints into a success curve (`eval_sweep.py`)

To see the **fine-tuning learning curve** (and later, to compare training conditions such as domain-randomization ablations), `eval_sweep.py` evaluates *every checkpoint* of a run under an identical, fixed protocol and plots success rate vs training step. By default it builds the Genesis scene **once** and reloads each checkpoint in turn; because the eval seeds are fixed, every checkpoint faces the *same* initial conditions — so the curve isolates the effect of training progress. (With `--dr-appearance` it instead sweeps several appearance domains while keeping the comparison fair — see below.)

```bash
uv run python genesis_scene_build/eval_sweep.py \
    --run-dir outputs/train/act_banana_pick_50ep \
    --repo-id genesis/banana_pick \
    --dataset-root genesis_scene_build/datasets/banana_pick_50ep \
    --episodes 20 --pick 011_banana
```

Outputs (under `eval_results/sweep_<run-name>/`):

- `sweep.json` — full results: `meta`, `aggregate` (per-checkpoint mean over domains), `per_domain` (per-domain curve rows), `evaluations` (one entry per domain×checkpoint)
- `sweep.csv` — one row per checkpoint (`step, success_rate, n_success, episodes, policy_type`)
- `success_curve.png` — success rate vs training step (with faint per-domain overlays when sweeping >1 domain)

It scans `<run-dir>/checkpoints/<step>/pretrained_model` and skips the `last` symlink. Use `--steps 10000 20000 …` to evaluate only specific checkpoints. **To get a real curve you need multiple checkpoints** — train with a smaller `--save-freq` (e.g. `train_policy.py … --save-freq 2000`) so intermediate steps are saved, not just the final one.

**Appearance-OOD sweep** (`--dr-appearance`): evaluate every checkpoint under `--dr-domains N` M4 appearance domains. The loop is **domain-outer / checkpoint-inner** — the scene is rebuilt once per domain (only `N` rebuilds total), and every checkpoint sees the *identical* set of domains, so the comparison stays fair. Each checkpoint's reported rate is the mean over all domains. Total rollouts = `domains × checkpoints × episodes`, and each domain adds one ~20 s rebuild, so keep `N`/`--episodes` sane for large checkpoint sets.

```bash
uv run python genesis_scene_build/eval_sweep.py \
    --run-dir outputs/train/act_banana_pick_50ep_step30000 \
    --repo-id genesis/banana_pick \
    --dataset-root genesis_scene_build/datasets/banana_pick_50ep \
    --episodes 20 --dr-appearance --dr-domains 5 --dr-object-color
```

Without `--dr-appearance` the sweep behaves exactly as before (single build, one domain). DR flags mirror the other scripts: `--dr-domains`, `--dr-table-jitter`, `--dr-object-color`, `--dr-fov-jitter`, `--dr-seed`.

## Domain randomization (M4)

Domain randomization (DR) is organized into three layers by *where* and *when* the variation is applied:

| Layer | When | What | Status |
|-------|------|------|--------|
| **A** | build time (per built scene) | object color, table color, camera FOV (intrinsics) | **implemented** |
| **B** | runtime (per episode) | friction, mass, camera extrinsics | planned |
| **C** | training time (per frame) | photometric jitter (brightness/contrast/hue/…) via lerobot's built-in `--dataset.image_transforms` | via lerobot config |

The split exists because the default Genesis rasterizer bakes colors/textures/intrinsics at `scene.build()` and cannot change them at runtime — so appearance/intrinsics DR must happen at build time (Layer A), while dynamics and extrinsics can be jittered per episode (Layer B).

### Layer A knobs

Configured by `SceneDomainRandomizationConfig` (in `build_scene.py`) and exposed as `--dr-*` flags on `build_scene.py`, `record_dataset.py`, and `eval_sweep.py`:

| Knob | Flag | Notes |
|------|------|-------|
| Table/leg color | `--dr-table-jitter <amp>` | Uniform ±`amp` per-RGB-channel jitter. Task-irrelevant *nuisance* — free to randomize widely. |
| Object color | `--dr-object-color` | **Opt-in.** Recolors objects **within per-object plausibility priors** (see below). Replaces the mesh texture with a flat plausible color. |
| Camera FOV | `--dr-fov-jitter <deg>` | ±`deg` on both cameras' vertical FOV (intrinsics; can only be set at build time). |
| Domain seed | `--dr-seed <n>` | Base seed; appearance domain `d` uses `n + d`, so runs are reproducible. |

Lighting is intentionally **not** a Layer-A knob: `scene.add_light` requires Genesis's `BatchRenderer`, whereas this project uses the default rasterizer. Do lighting-like variation at Layer C (photometric jitter) instead.

### Plausible object recolor (per-object priors)

Object recolor is constrained to each object's *real-world* appearance distribution via `DR_APPEARANCE_PRIORS` in `scene_config.py`, so recolored objects stay physically sensible (a banana drifts yellow↔yellow-green for ripeness but never turns black or blue). Sampling is done in **HSV** — a narrow hue band plus saturation/value ranges with a value floor (prevents implausibly dark colors). Objects without an entry keep their original texture; the bowl is a container (not a grasp target), so its color is a nuisance and its band is left wide.

### Preview the appearance domains (`dr_preview.py`)

Renders the scene under a baseline (DR off) plus one tile per seed and stitches a labeled montage into `eval_results/dr_preview/` (instead of dumping PNGs into the repo root). Each domain is rendered in its own subprocess (fresh `gs.init` + build), so the montage is reproducible given the same seeds.

```bash
# baseline + 4 seeds, recolor objects, jitter table + FOV
uv run python genesis_scene_build/dr_preview.py \
    --seeds 0 1 2 3 --object-color --table-jitter 0.15 --fov-jitter 2.0

# nuisance-only (keep original object textures), also montage the wrist cam
uv run python genesis_scene_build/dr_preview.py --seeds 0 1 2 --table-jitter 0.2 --wrist
```

Outputs: `eval_results/dr_preview/world_montage.png` (+ `wrist_montage.png` with `--wrist`) and the per-domain frames under `frames/`.

### Where Layer A is wired in

- **`build_scene.py`** — one scene under a chosen domain (for inspection / `dr_preview`).
- **`record_dataset.py`** — rebuilds every `--dr-rebuild-every N` successful episodes to spread multiple appearance domains across one dataset (variety is purely beneficial when generating data).
- **`eval_sweep.py`** — sweeps `--dr-domains N` domains **domain-outer / checkpoint-inner**, so every checkpoint is scored on the identical set of domains (fair comparison); reports the per-checkpoint mean plus per-domain curves.

Recommendation: `--dr-object-color` is the most aggressive knob (it discards the realistic YCB texture), so consider leaving it off — randomize the nuisances (table + FOV) at Layer A and get object-appearance variety from Layer C photometric jitter at training time.

## Object grasp reliability

The scripted parallel-jaw grasp is reliable for the banana (~100%) and moderately reliable (~50%) for the near-spherical `lemon`/`plum` at realistic friction. Round objects (`apple`, `orange`, `pear`) and the `mug` are not reliably graspable and are disabled by default. For clean data generation, keep to the banana (or rely on the success filter).

## Helper: scale down a YCB object

```bash
uv run python genesis_scene_build/scale_ycb.py --scale 0.8 013_apple 017_orange
```

Scales mesh geometry (and collision mesh) while copying materials/textures verbatim, into a parallel dataset that `setup_assets.py` can pull from.
