"""
Live rollout evaluation of a saved FlowMatchingPolicy in the RoboMME BinFill environment.

Runs the trained policy open-loop in the actual simulator and reads the real success signal
(info["status"] == "success") from the environment, plus per-subtask progress.

Setup
-----
- Requires robomme_benchmark installed in the same Python environment:
    cd /path/to/robomme_benchmark && pip install -e .
- Run from the ManiSkill root:
    python examples/baselines/bc/rollout_eval_binfill_fm.py \\
        --checkpoint runs/BinFill__bc_binfill_fm__1__<ts>/checkpoints/final.pt \\
        --robomme_root /home/ubuntu/robomme_benchmark \\
        --h5_file robomme_data/record_dataset_BinFill.h5 \\
        --splits val train

Action space
------------
Uses action_space="joint_angle" (8-D joint positions + gripper) by default.
ee_pose is also supported — the IK planner conflict that caused segfaults in the
original approach is resolved because bypass_demo_reset() skips
DemonstrationWrapper.reset() entirely, so DemonstrationWrapper never creates its
motion planner (EndeffectorDemonstrationWrapper creates its IK planner lazily).

DemonstrationWrapper bypass
---------------------------
make_env_for_episode() creates the full wrapper stack:
  FailAwareWrapper → DemonstrationWrapper → BinFill
DemonstrationWrapper.reset() would normally replay the full demonstration via a
motion planner, leaving the robot at the task-completion pose.  bypass_demo_reset()
skips this by: (1) manually initialising DemonstrationWrapper's episode-level state
variables, (2) calling the underlying env.reset() so the robot starts at home with
objects randomised by episode seed.  _step_batch() never reads demonstration_data
(confirmed from source), so setting it to None is safe.

Live subgoal tracking
---------------------
After bypass reset, env.step() goes through the full DemonstrationWrapper stack.
_augment_obs_and_info() reads BinFill.current_task_name via __getattr__ proxy
(updated every step by sequential_task_check() based on physics state) and writes
it to info["simple_subgoal_online"].  This gives correct, live subgoal updates
without any dependence on the demonstration data.

Observation construction per step
----------------------------------
Continuous (37-D):
  eef_state(6)               ← obs["eef_state_list"]
  joint_state(7)             ← obs["joint_state_list"][:7]
  gripper_state(2)           ← obs["gripper_state_list"]
  is_gripper_close(1)        ← gripper_state.mean() < 0.02
  front_camera_intrinsic(9)  ← info["front_camera_intrinsic"]  (constant per episode)
  wrist_camera_intrinsic(9)  ← info["wrist_camera_intrinsic"]  (constant per episode)
  is_completed(1)            ← 1.0 if info["status"]=="success" else 0.0
  is_subgoal_boundary(1)     ← 1.0 when simple_subgoal_online changes
  is_video_demo(1)           ← always 0.0 (policy rollout, not video demo)

Categorical:
  difficulty      ← episode metadata
  task_goal       ← info["task_goal"][0]
  simple_subgoal  ← info["simple_subgoal_online"]
"""

import argparse
import faulthandler
import os
import sys
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

faulthandler.enable()  # print Python stack trace on SIGSEGV

import h5py
import numpy as np
import torch

# ── locate bc scripts ──────────────────────────────────────────────────────
_BC_DIR = os.path.dirname(__file__)
sys.path.insert(0, _BC_DIR)
from bc_binfill import CAT_FIELDS, CONT_OBS_DIM, Vocab
from bc_binfill_fm import FlowMatchingPolicy

GRIPPER_CLOSE_THRESH = 0.02  # metres — gripper counted as closed below this

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True,
                   help="path to saved .pt checkpoint")
    p.add_argument("--robomme_root",  default="/home/ubuntu/robomme_benchmark",
                   help="path to robomme_benchmark repo root (must be pip-installed)")
    p.add_argument("--h5_file",       default="robomme_data/record_dataset_BinFill.h5",
                   help="h5 file used for training (to reproduce train/val split)")
    p.add_argument("--dataset",       default="train",
                   choices=["train", "test"],
                   help="which robomme metadata set the h5 file belongs to")
    p.add_argument("--splits",        nargs="+", default=["val"],
                   choices=["val", "train"],
                   help="which splits to evaluate (val, train, or both)")
    p.add_argument("--val_fraction",  type=float, default=0.2)
    p.add_argument("--seed",          type=int,   default=1,
                   help="must match training seed to reproduce same split")
    p.add_argument("--max_steps",     type=int,   default=1300)
    p.add_argument("--n_inference_steps", type=int, default=10)
    p.add_argument("--cuda",          action="store_true", default=True)
    p.add_argument("--no_cuda",       dest="cuda", action="store_false")
    # model architecture — must match training
    p.add_argument("--embed_dim",     type=int, default=16)
    p.add_argument("--context_dim",   type=int, default=256)
    p.add_argument("--hidden_dim",    type=int, default=256)
    p.add_argument("--time_emb_dim",  type=int, default=64)
    p.add_argument("--action_space",  default="joint_angle",
                   choices=["joint_angle", "ee_pose"],
                   help="joint_angle: send 8-D joint targets directly; "
                        "ee_pose: send 7-D [xyz,rpy,gripper] converted via IK "
                        "(bypass_demo_reset avoids the double-planner conflict that "
                        "previously caused a segfault with ee_pose)")
    p.add_argument("--action_key",    default=None,
                   choices=["eef_action", "joint_action", "waypoint_action"],
                   help="which h5 action field the checkpoint was trained on; "
                        "defaults to joint_action for joint_angle space, "
                        "eef_action for ee_pose space")
    p.add_argument("--max_episodes_per_split", type=int, default=None,
                   help="cap episodes per split for quick testing (None = all)")
    args = p.parse_args()

    # derive action_key default from action_space if not explicitly set
    if args.action_key is None:
        args.action_key = "joint_action" if args.action_space == "joint_angle" else "eef_action"

    return args


# ---------------------------------------------------------------------------
# Vocabulary builder  (must reproduce training vocab from h5)
# ---------------------------------------------------------------------------

def _decode(raw) -> str:
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def build_vocabs_from_h5(h5_file: str) -> Dict[str, Vocab]:
    """Scan the h5 file and build the same vocabs as training."""
    vocabs = {f: Vocab() for f in CAT_FIELDS}
    with h5py.File(h5_file, "r") as f:
        for ep_key in sorted(f.keys()):
            ep = f[ep_key]
            if "setup" not in ep:
                continue
            diff = _decode(ep["setup"]["difficulty"][()])
            tg_raw = ep["setup"]["task_goal"]
            tg = _decode(tg_raw[0] if tg_raw.shape[0] > 0 else tg_raw[()])
            vocabs["difficulty"].add(diff)
            vocabs["task_goal"].add(tg)

            for ts_key in [k for k in ep.keys() if k.startswith("timestep_")]:
                sg = _decode(ep[ts_key]["info"]["simple_subgoal"][()])
                vocabs["simple_subgoal"].add(sg)
    return vocabs


# ---------------------------------------------------------------------------
# Reproduce train/val episode index split
# ---------------------------------------------------------------------------

def get_split_episode_indices(
    h5_file: str, val_fraction: float, seed: int
) -> Tuple[List[int], List[int]]:
    """
    Returns (train_episode_indices, val_episode_indices) matching the split
    used during bc_binfill training.  Indices are h5 episode numbers (0-based).
    """
    with h5py.File(h5_file, "r") as f:
        n = len(f.keys())
    rng   = np.random.default_rng(seed)
    order = rng.permutation(n).tolist()  # same as bc_binfill.py
    n_val = max(1, int(n * val_fraction))
    val_indices   = sorted(order[:n_val])
    train_indices = sorted(order[n_val:])
    return train_indices, val_indices


# ---------------------------------------------------------------------------
# DemonstrationWrapper bypass — keeps the full wrapper stack, skips demo replay
# ---------------------------------------------------------------------------

def find_demo_wrapper(env):
    """Traverse the gymnasium wrapper stack to find the DemonstrationWrapper instance."""
    current = env
    while current is not None:
        if type(current).__name__ == "DemonstrationWrapper":
            return current
        current = getattr(current, 'env', None)
    return None


def bypass_demo_reset(env):
    """
    Reset env without running DemonstrationWrapper.reset()'s demo replay.

    DemonstrationWrapper.reset() calls get_demonstration_trajectory() which runs the
    full demonstration via a motion planner, leaving the robot at the task-completion
    pose.  This bypass:
      1. Manually replicates DemonstrationWrapper.reset()'s episode-level state init
         (lines 149-160 of DemonstrationWrapper.py) without get_demonstration_trajectory()
      2. Calls demo_wrapper.env.reset() directly — robot at home, objects randomised
         by episode seed
      3. Sets demonstration_data = None (safe: _step_batch() never reads it)

    Returns (obs, info, demo_wrapper).  After this call, env.step() goes through the
    full DemonstrationWrapper stack and provides live subgoal via info["simple_subgoal_online"].
    """
    demo_wrapper = find_demo_wrapper(env)
    if demo_wrapper is None:
        obs, info = env.reset()
        return obs, info, None

    # Replicate DemonstrationWrapper.reset() state initialisation (without demo)
    demo_wrapper.last_subgoal_segment = None
    demo_wrapper.latched_replacements = None
    demo_wrapper._failed_match_save_count = 0
    demo_wrapper.steps_without_demonstration = 0
    demo_wrapper._prev_ee_quat_wxyz = None
    demo_wrapper._prev_ee_rpy_xyz = None
    demo_wrapper.demonstration_data = None  # safe: _step_batch() never reads this

    # Reset the underlying env (robot at home, scene randomised by seed, no demo)
    obs, info = demo_wrapper.env.reset()
    demo_wrapper.episode_success = False

    return obs, info, demo_wrapper


def _to_numpy(t) -> np.ndarray:
    if hasattr(t, "detach"):
        t = t.detach().cpu()
    return np.asarray(t).flatten()


def get_robot_state(env) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract eef(6), joint(7), gripper(2) directly from the live SAPIEN env.
    Works with the raw gym env (no DemonstrationWrapper needed).
    """
    from scipy.spatial.transform import Rotation

    base_env = env.unwrapped
    robot = base_env.agent.robot
    tcp   = base_env.agent.tcp

    qpos = _to_numpy(robot.qpos).astype(np.float32)
    joint_state   = qpos[:7]
    gripper_state = qpos[7:9] if len(qpos) > 7 else np.zeros(2, np.float32)

    p = _to_numpy(tcp.pose.p).astype(np.float32)[:3]
    q = _to_numpy(tcp.pose.q).astype(np.float32)  # wxyz
    # wxyz → xyzw for scipy
    rpy = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz").astype(np.float32)
    eef_state = np.concatenate([p, rpy])           # (6,)

    return eef_state, joint_state, gripper_state


def read_cam_intrinsics_from_h5(h5_file: str, ep_num: int) -> Dict[str, np.ndarray]:
    """
    Read camera intrinsics from ep["setup"], matching _read_setup_cont() in bc_binfill.py.

    Intrinsics are episode-level constants stored in ep["setup"]["front_camera_intrinsic"]
    (shape 3×3). They are NOT stored per-timestep in ts["info"].
    Using the wrong location returns an identity matrix, which breaks 18/37 obs dims.
    """
    ep_key = f"episode_{ep_num}"
    fallback = {"front": np.eye(3, np.float32).ravel(), "wrist": np.eye(3, np.float32).ravel()}
    with h5py.File(h5_file, "r") as f:
        if ep_key not in f or "setup" not in f[ep_key]:
            return fallback
        setup = f[ep_key]["setup"]
        front = setup["front_camera_intrinsic"][()].astype(np.float32).ravel()
        wrist = setup["wrist_camera_intrinsic"][()].astype(np.float32).ravel()
    return {
        "front": front[:9] if len(front) >= 9 else np.pad(front, (0, 9 - len(front))),
        "wrist": wrist[:9] if len(wrist) >= 9 else np.pad(wrist, (0, 9 - len(wrist))),
    }


def build_obs_tensor_raw(
    eef: np.ndarray,
    joint: np.ndarray,
    grip: np.ndarray,
    cam_intrinsics: Dict[str, np.ndarray],
    is_completed: float,
    is_subgoal_bound: float,
    task_goal: str,
    difficulty: str,
    subgoal: str,
    vocabs: Dict[str, Vocab],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Build the same 37-D continuous + categorical obs as during training."""
    is_close = np.array([float(grip.mean() < GRIPPER_CLOSE_THRESH)], np.float32)

    cont = np.concatenate([
        eef, joint, grip, is_close,
        cam_intrinsics["front"],
        cam_intrinsics["wrist"],
        np.array([is_completed],     np.float32),
        np.array([is_subgoal_bound], np.float32),
        np.array([0.0],              np.float32),  # is_video_demo always 0 at rollout
    ]).astype(np.float32)
    cont_t = torch.from_numpy(cont).unsqueeze(0).to(device)

    cat_t = {
        "difficulty":     torch.tensor([vocabs["difficulty"](difficulty)],  device=device),
        "task_goal":      torch.tensor([vocabs["task_goal"](task_goal)],    device=device),
        "simple_subgoal": torch.tensor([vocabs["simple_subgoal"](subgoal)], device=device),
    }
    return cont_t, cat_t


# ---------------------------------------------------------------------------
# Single episode rollout (DemonstrationWrapper bypass for live subgoal tracking)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_episode(
    env,
    task_goal: str,
    difficulty: str,
    cam_intrinsics: Dict[str, np.ndarray],
    policy: FlowMatchingPolicy,
    vocabs: Dict[str, Vocab],
    device: torch.device,
    n_steps: int,
    max_steps: int,
) -> dict:
    print("    [run_episode] bypass_demo_reset ...", flush=True)
    _, _, demo_wrapper = bypass_demo_reset(env)
    print("    [run_episode] reset done", flush=True)

    # Initial subgoal: BinFill.current_task_name is set during reset's evaluate() call,
    # accessible via gymnasium.Wrapper.__getattr__ proxy on demo_wrapper.
    prev_subgoal: str = getattr(demo_wrapper, 'current_task_name', '') if demo_wrapper else ''

    step_count = 0
    is_completed = 0.0
    is_subgoal_bound = 0.0
    subtasks_seen: List[str] = ([prev_subgoal] if prev_subgoal else [])
    subtasks_done: List[str] = []

    while step_count < max_steps:
        eef, joint, grip = get_robot_state(env)
        cont_t, cat_t = build_obs_tensor_raw(
            eef, joint, grip, cam_intrinsics,
            is_completed, is_subgoal_bound,
            task_goal, difficulty, prev_subgoal, vocabs, device,
        )

        action = policy.sample(cont_t, cat_t, n_steps=n_steps)
        action_np = action.squeeze(0).cpu().numpy().astype(np.float64)
        if step_count == 0:
            print(f"    [run_episode] first action shape={action_np.shape} values={action_np}",
                  flush=True)

        _, _reward, terminated, truncated, info = env.step(action_np)
        step_count += 1

        # Live subgoal from DemonstrationWrapper._augment_obs_and_info() via __getattr__ proxy.
        # Falls back to reading current_task_name directly if the key is absent.
        current_subgoal = str(info.get("simple_subgoal_online", ""))
        if not current_subgoal and demo_wrapper is not None:
            current_subgoal = getattr(demo_wrapper, 'current_task_name', '') or ''

        # Detect subgoal transition
        if current_subgoal and current_subgoal != prev_subgoal:
            is_subgoal_bound = 1.0
            if prev_subgoal and prev_subgoal not in subtasks_done:
                subtasks_done.append(prev_subgoal)
            if current_subgoal not in subtasks_seen:
                subtasks_seen.append(current_subgoal)
        else:
            is_subgoal_bound = 0.0
        prev_subgoal = current_subgoal

        # DemonstrationWrapper sets info["status"]: "success"/"fail"/"timeout"/"ongoing"/"error"
        status = info.get("status", "ongoing")
        is_completed = 1.0 if status == "success" else 0.0

        if status == "success":
            if current_subgoal and current_subgoal not in subtasks_done:
                subtasks_done.append(current_subgoal)
            n_seen = max(len(subtasks_seen), 1)
            return {
                "status": "success",
                "steps": step_count,
                "task_goal": task_goal,
                "subtask_success_rate": len(subtasks_done) / n_seen,
                "subtasks_seen": subtasks_seen,
                "subtasks_done": subtasks_done,
            }

        term = _to_numpy(terminated).any() if hasattr(terminated, '__iter__') else bool(terminated)
        trun = _to_numpy(truncated).any()  if hasattr(truncated,  '__iter__') else bool(truncated)
        if status in ("fail", "error", "timeout") or term or trun:
            n_seen = max(len(subtasks_seen), 1)
            return {
                "status": status if status in ("fail", "error", "timeout") else "fail",
                "steps": step_count,
                "task_goal": task_goal,
                "subtask_success_rate": len(subtasks_done) / n_seen,
                "subtasks_seen": subtasks_seen,
                "subtasks_done": subtasks_done,
            }

    n_seen = max(len(subtasks_seen), 1)
    return {
        "status": "timeout",
        "steps": step_count,
        "task_goal": task_goal,
        "subtask_success_rate": len(subtasks_done) / n_seen,
        "subtasks_seen": subtasks_seen,
        "subtasks_done": subtasks_done,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── add robomme to path ──
    robomme_src = os.path.join(args.robomme_root, "src")
    sys.path.insert(0, robomme_src)
    import robomme.robomme_env  # noqa: F401 — registers BinFill (and all tasks) with gym
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")

    # ── vocabularies ──
    print(f"Building vocabularies from {args.h5_file} ...")
    vocabs = build_vocabs_from_h5(args.h5_file)
    print("  Vocab sizes:", {f: len(v) for f, v in vocabs.items()})

    # ── model ──
    from bc_binfill import ACTION_DIM
    action_dim  = ACTION_DIM[args.action_key]
    vocab_sizes = {f: len(vocabs[f]) for f in CAT_FIELDS}
    policy = FlowMatchingPolicy(
        action_dim   = action_dim,
        vocab_sizes  = vocab_sizes,
        embed_dim    = args.embed_dim,
        context_dim  = args.context_dim,
        hidden_dim   = args.hidden_dim,
        time_emb_dim = args.time_emb_dim,
    ).to(device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    policy.eval()
    print(f"Policy loaded from {args.checkpoint}")

    # ── episode split ──
    train_idx, val_idx = get_split_episode_indices(
        args.h5_file, args.val_fraction, args.seed
    )
    split_map = {"train": train_idx, "val": val_idx}

    print(f"Action space : {args.action_space}  (action_key={args.action_key})")

    # BenchmarkEnvBuilder creates the full wrapper stack (DemonstrationWrapper + FailAwareWrapper).
    # bypass_demo_reset() skips DemonstrationWrapper.reset() to avoid demo replay.
    env_builder = BenchmarkEnvBuilder(
        env_id       = "BinFill",
        dataset      = args.dataset,
        action_space = args.action_space,
        gui_render   = False,
        max_steps    = args.max_steps,
    )

    # ── evaluate each requested split ──
    for split_name in args.splits:
        episode_indices = split_map[split_name]
        if args.max_episodes_per_split is not None:
            episode_indices = episode_indices[:args.max_episodes_per_split]

        print(f"\n{'='*60}")
        print(f"SPLIT: {split_name.upper()}  ({len(episode_indices)} episodes)")
        print("="*60)
        print(f"{'Ep':>4}  {'Status':>8}  {'Steps':>6}  Task goal")
        print("-"*70)

        results = []
        for ep_num in episode_indices:
            _, difficulty = env_builder.resolve_episode(ep_num)
            difficulty = difficulty or "easy"

            # read task_goal and camera intrinsics from h5 (no DemonstrationWrapper needed)
            ep_key = f"episode_{ep_num}"
            with h5py.File(args.h5_file, "r") as f:
                ep = f[ep_key]
                tg_raw = ep["setup"]["task_goal"]
                task_goal = _decode(tg_raw[0] if tg_raw.shape[0] > 0 else tg_raw[()])
            cam_intrinsics = read_cam_intrinsics_from_h5(args.h5_file, ep_num)

            print(f"  [ep {ep_num}] creating env (DemonstrationWrapper stack) ...", flush=True)
            env = env_builder.make_env_for_episode(ep_num, max_steps=args.max_steps)
            print(f"  [ep {ep_num}] env ready, running episode ...", flush=True)
            try:
                result = run_episode(
                    env, task_goal, difficulty, cam_intrinsics,
                    policy, vocabs, device, args.n_inference_steps, args.max_steps,
                )
            except Exception as e:
                result = {"status": "error", "error_message": str(e), "steps": 0,
                          "task_goal": task_goal, "subtask_success_rate": 0.0,
                          "subtasks_seen": [], "subtasks_done": []}
            finally:
                env.close()

            result["episode"]    = ep_num
            result["difficulty"] = difficulty
            results.append(result)

            print(
                f"{ep_num:>4d}  {result['status']:>8}  {result.get('steps',0):>6d}  "
                f"{result['task_goal'][:40]}"
            )

        # ── aggregate ──
        n          = len(results)
        successes  = [r for r in results if r["status"] == "success"]
        fails      = [r for r in results if r["status"] == "fail"]
        timeouts   = [r for r in results if r["status"] == "timeout"]
        errors     = [r for r in results if r["status"] == "error"]
        sr_list    = [r["subtask_success_rate"] for r in results]
        steps_list = [r.get("steps", 0) for r in results]

        # breakdown by difficulty
        diff_results: Dict[str, list] = {}
        for r in results:
            diff_results.setdefault(r["difficulty"], []).append(r)

        print(f"\n{'─'*60}")
        print(f"RESULTS — {split_name.upper()} ({n} episodes)")
        print(f"{'─'*60}")
        print(f"  Episode success rate : {len(successes)/n:.3f}  ({len(successes)}/{n})")
        print(f"  Fail                 : {len(fails)/n:.3f}  ({len(fails)}/{n})")
        print(f"  Timeout              : {len(timeouts)/n:.3f}  ({len(timeouts)}/{n})")
        print(f"  Error                : {len(errors)/n:.3f}  ({len(errors)}/{n})")
        print(f"  Subtask success rate : {np.mean(sr_list):.3f}  (1.0=success, 0.0=fail/timeout)")
        print(f"  Mean steps/episode   : {np.mean(steps_list):.1f}")

        print(f"\n  Breakdown by difficulty:")
        for diff in sorted(diff_results):
            dr   = diff_results[diff]
            succ = sum(1 for r in dr if r["status"] == "success")
            print(f"    {diff:8s}: {succ}/{len(dr)}  ({succ/len(dr):.3f})")

        # ── save JSON ──
        out_path = f"runs/rollout_eval_{split_name}.json"
        os.makedirs("runs", exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump({
                "split": split_name,
                "checkpoint": args.checkpoint,
                "n_episodes": n,
                "success_rate": len(successes) / n,
                "subtask_success_rate": float(np.mean(sr_list)),
                "results": results,
            }, fh, indent=2)
        print(f"\n  Full results saved to {out_path}")


if __name__ == "__main__":
    main()
