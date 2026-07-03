"""M2 environment randomization: per-episode reset() for the Franka scene.

This is the first (small-amplitude) stage of M2. It randomizes only the *task*
variety that pick-and-place inherently needs -- object poses and task selection --
and deliberately leaves domain randomization (friction, mass, lighting, camera
jitter, textures) to M4.

Design choices (per user):

1. Non-overlap via a **slot method** rather than rejection sampling. Each object
   owns a fixed "home slot" (its nominal position in ``YCB_LAYOUT``). The tuned
   layout is already collision-free and reachable, so keeping every object in its
   own slot guarantees non-overlap *by construction*. To keep that guarantee even
   with jitter, each object's jitter is clamped to a per-object safe radius derived
   from the gap to its nearest neighbor, so no rejection loop is ever needed.

2. **Small perturbation to start.** Positions jitter within a few centimeters of
   the home slot and yaw is fully randomized. Amplitude can be scaled up later
   (larger jitter, slot reassignment) without changing the interface.

Usage:
    randomizer = EnvRandomizer(bundle, RandomizationConfig(seed=0))
    task = randomizer.reset()          # new episode, returns a TaskSpec
    success, _ = run_pick_place(bundle, task)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from genesis.utils.geom import euler_to_quat

from grasp_demo import PlaceTarget, TaskSpec
from scene_config import (
    FRANKA_QPOS,
    REACH_X,
    REACH_Y,
    TABLE_TOP_Z,
    YCB_LAYOUT,
    get_ycb_assets,
)

# Objects used as pick targets during M2 randomization development. Kept intentionally
# small (banana, lemon, plum) while the reset pipeline is being validated; the rest stay
# in the scene as distractors. Expand this pool once reliability is confirmed.
RELIABLE_PICK_POOL = ("011_banana", "014_lemon", "018_plum")

# Default container to place the picked object into.
DEFAULT_PLACE_CONTAINER = "024_bowl"

# Extra clearance kept between object footprints when clamping jitter (m).
OVERLAP_MARGIN = 0.02


@dataclass
class RandomizationConfig:
    """Knobs for one episode's randomization. Defaults are the small-start regime."""

    pos_jitter: float = 0.03  # +/- m, uniform box around each object's home slot (x, y)
    # Yaw is jittered by a small amount around each object's home orientation rather than
    # fully randomized: the tuned layout packs objects only ~8 cm apart, so a large yaw
    # swings the gripper's finger sweep into neighbors and explodes the contact solver.
    yaw_jitter: float = 30.0  # +/- deg around each object's home yaw (0 disables)
    randomize_pick: bool = True  # sample the pick object from RELIABLE_PICK_POOL
    randomize_place: bool = False  # if True, sometimes place onto a random tabletop xy
    place_tabletop_prob: float = 0.0  # P(tabletop target) when randomize_place is True
    settle_steps: int = 80  # physics steps to let objects/arm settle after teleport
    success_tol: float = 0.08  # forwarded into the sampled TaskSpec
    seed: int | None = None


class EnvRandomizer:
    """Resets a built ``SceneBundle`` to a fresh, randomized pick-and-place episode."""

    def __init__(self, bundle, config: RandomizationConfig | None = None):
        self.bundle = bundle
        self.cfg = config or RandomizationConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        self._assets = get_ycb_assets()
        self.names = list(YCB_LAYOUT.keys())
        self.home = {n: np.asarray(YCB_LAYOUT[n]["pos"][:2], dtype=float) for n in self.names}
        self.home_yaw = {n: float(YCB_LAYOUT[n]["euler"][2]) for n in self.names}
        self.radius = {n: self._assets[n].radius_xy for n in self.names}
        self.rest_z = {n: self._assets[n].rest_z_offset for n in self.names}

        # Per-object safe jitter: half the clearance to the nearest neighbor slot, so
        # that even if two neighbors jitter toward each other they cannot overlap.
        self.safe_jitter = self._compute_safe_jitter()

    def _compute_safe_jitter(self) -> dict[str, float]:
        safe: dict[str, float] = {}
        for a in self.names:
            gap = np.inf
            for b in self.names:
                if a == b:
                    continue
                center_dist = float(np.linalg.norm(self.home[a] - self.home[b]))
                clearance = center_dist - self.radius[a] - self.radius[b] - OVERLAP_MARGIN
                gap = min(gap, clearance)
            # Both neighbors may move by J toward each other -> require 2J <= gap.
            safe[a] = max(0.0, 0.5 * gap) if np.isfinite(gap) else self.cfg.pos_jitter
        return safe

    # -- public API ---------------------------------------------------------

    def reset(self, seed: int | None = None) -> TaskSpec:
        """Randomize object poses + task and settle physics. Returns the TaskSpec."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._reset_robot()
        self._place_objects()
        self._settle()
        return self._sample_task()

    # -- internals ----------------------------------------------------------

    def _reset_robot(self) -> None:
        q = np.asarray(FRANKA_QPOS, dtype=float)
        self.bundle.franka.set_qpos(q, zero_velocity=True)

    def _place_objects(self) -> None:
        for name in self.names:
            jitter = min(self.cfg.pos_jitter, self.safe_jitter[name])
            dx, dy = self.rng.uniform(-jitter, jitter, size=2)
            x = float(np.clip(self.home[name][0] + dx, *REACH_X))
            y = float(np.clip(self.home[name][1] + dy, *REACH_Y))
            z = TABLE_TOP_Z + self.rest_z[name] + 0.002  # tiny clearance; settles down

            dyaw = float(self.rng.uniform(-self.cfg.yaw_jitter, self.cfg.yaw_jitter))
            yaw = self.home_yaw[name] + dyaw
            quat = euler_to_quat(np.array([0.0, 0.0, yaw]))

            entity = self.bundle.ycb[name]
            entity.set_pos(np.array([x, y, z]), relative=False, zero_velocity=True, skip_forward=True)
            entity.set_quat(quat, relative=False, zero_velocity=True, skip_forward=False)

    def _settle(self) -> None:
        hold = np.asarray(FRANKA_QPOS, dtype=float)
        for _ in range(self.cfg.settle_steps):
            self.bundle.franka.control_dofs_position(hold)
            self.bundle.scene.step()
            self.bundle.update_wrist_cam()

    def _sample_task(self) -> TaskSpec:
        pool = [n for n in RELIABLE_PICK_POOL if n in self.bundle.ycb]
        if not pool:
            raise RuntimeError("No reliable pick objects present in the scene.")
        pick = str(self.rng.choice(pool)) if self.cfg.randomize_pick else pool[0]

        place = self._sample_place(exclude=pick)
        return TaskSpec(pick_object=pick, place_target=place, success_tol=self.cfg.success_tol)

    def _sample_place(self, *, exclude: str) -> PlaceTarget:
        want_tabletop = (
            self.cfg.randomize_place and self.rng.uniform() < self.cfg.place_tabletop_prob
        )
        container_ok = DEFAULT_PLACE_CONTAINER in self.bundle.ycb
        if not want_tabletop and container_ok:
            return DEFAULT_PLACE_CONTAINER
        return self._sample_free_xy(exclude=exclude)

    def _sample_free_xy(self, *, exclude: str, min_clear: float = 0.10, tries: int = 50) -> tuple[float, float]:
        """Sample a tabletop xy that is clear of all objects (rejection, bounded tries)."""
        others = [n for n in self.names if n != exclude]
        for _ in range(tries):
            x = float(self.rng.uniform(*REACH_X))
            y = float(self.rng.uniform(*REACH_Y))
            p = np.array([x, y])
            if all(np.linalg.norm(p - self._current_xy(n)) > self.radius[n] + min_clear for n in others):
                return x, y
        # Fallback: reachable-zone center.
        return float(np.mean(REACH_X)), float(np.mean(REACH_Y))

    def _current_xy(self, name: str) -> np.ndarray:
        return self.bundle.ycb[name].get_pos().cpu().numpy().reshape(-1)[:2]


def main() -> None:
    import argparse

    import genesis as gs

    from build_scene import build_scene
    from grasp_demo import run_pick_place

    parser = argparse.ArgumentParser(description="M2 randomized pick-and-place episodes.")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-n", "--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument("--jitter", type=float, default=0.03, help="Position jitter (m).")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)
    bundle = build_scene(show_viewer=args.vis, n_envs=1, add_world_cam=True, add_wrist_cam=True)

    cfg = RandomizationConfig(pos_jitter=args.jitter, seed=args.seed)
    randomizer = EnvRandomizer(bundle, cfg)

    n_success = 0
    for ep in range(args.episodes):
        episode_seed = args.seed + ep
        task = randomizer.reset(seed=episode_seed)
        success, _ = run_pick_place(bundle, task)
        n_success += int(success)
        print(
            f"[randomize] ep {ep:03d} seed={episode_seed} "
            f"pick={task.pick_object} place={task.place_target} -> success={success}"
        )

    print(f"[randomize] {n_success}/{args.episodes} succeeded")


if __name__ == "__main__":
    main()
