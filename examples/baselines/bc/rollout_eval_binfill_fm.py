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
Uses action_space="joint_angle" (8-D joint positions + gripper).
Train the policy with --action_key joint_action to match.
Using ee_pose wraps the env with an extra IK planner that conflicts with
DemonstrationWrapper's own planner during reset, causing a segfault.

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
    p.add_argument("--action_key",    default="joint_action",
                   choices=["eef_action", "joint_action", "waypoint_action"])
    p.add_argument("--max_episodes_per_split", type=int, default=None,
                   help="cap episodes per split for quick testing (None = all)")
    return p.parse_args()


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
# Obs builder from live env step output
# ---------------------------------------------------------------------------

def build_obs_tensor(
    obs: dict,
    info: dict,
    cam_intrinsics: Dict[str, np.ndarray],  # captured once per episode
    prev_subgoal: Optional[str],
    vocabs: Dict[str, Vocab],
    difficulty: str,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], str]:
    """
    Build (cont_tensor, cat_tensor_dict, current_subgoal) from one env step.
    cam_intrinsics = {"front": (9,), "wrist": (9,)} captured at episode start.
    """
    # ── continuous obs ──
    eef    = np.asarray(obs["eef_state_list"],    dtype=np.float32).flatten()[:6]   # (6,)
    joint  = np.asarray(obs["joint_state_list"],  dtype=np.float32).flatten()[:7]   # (7,)
    grip   = np.asarray(obs["gripper_state_list"],dtype=np.float32).flatten()[:2]   # (2,)
    is_close = np.array([float(grip.mean() < GRIPPER_CLOSE_THRESH)], dtype=np.float32)

    status = info.get("status", "ongoing")
    is_completed     = np.array([1.0 if status == "success" else 0.0], dtype=np.float32)
    current_subgoal  = str(info.get("simple_subgoal_online", ""))
    is_subgoal_bound = np.array(
        [1.0 if (prev_subgoal is not None and current_subgoal != prev_subgoal) else 0.0],
        dtype=np.float32,
    )
    is_video_demo = np.array([0.0], dtype=np.float32)

    cont = np.concatenate([
        eef, joint, grip, is_close,
        cam_intrinsics["front"],           # (9,)
        cam_intrinsics["wrist"],           # (9,)
        is_completed, is_subgoal_bound, is_video_demo,
    ]).astype(np.float32)                  # (37,)

    cont_t = torch.from_numpy(cont).unsqueeze(0).to(device)  # (1, 37)

    # ── categorical obs ──
    task_goal_raw = info.get("task_goal", [""])
    task_goal_str = task_goal_raw[0] if isinstance(task_goal_raw, (list, tuple)) else str(task_goal_raw)

    cat_t = {
        "difficulty":     torch.tensor([vocabs["difficulty"](difficulty)],    device=device),
        "task_goal":      torch.tensor([vocabs["task_goal"](task_goal_str)],  device=device),
        "simple_subgoal": torch.tensor([vocabs["simple_subgoal"](current_subgoal)], device=device),
    }

    return cont_t, cat_t, current_subgoal


def extract_camera_intrinsics(info: dict) -> Dict[str, np.ndarray]:
    """Pull camera intrinsics from reset info (constant for the whole episode)."""
    def _get(key):
        raw = info.get(key)
        if raw is None:
            return np.eye(3, dtype=np.float32).ravel()
        arr = np.asarray(raw, dtype=np.float32).ravel()
        return arr[:9] if len(arr) >= 9 else np.pad(arr, (0, 9 - len(arr)))

    return {
        "front": _get("front_camera_intrinsic"),
        "wrist": _get("wrist_camera_intrinsic"),
    }


# ---------------------------------------------------------------------------
# Single episode rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_episode(
    env,
    policy: FlowMatchingPolicy,
    vocabs: Dict[str, Vocab],
    difficulty: str,
    device: torch.device,
    n_steps: int,
    max_steps: int,
) -> dict:
    print("    [run_episode] calling env.reset() ...", flush=True)
    obs, info = env.reset()
    print("    [run_episode] reset done", flush=True)
    cam_intrinsics  = extract_camera_intrinsics(info)
    task_goal       = info.get("task_goal", [""])[0] if isinstance(info.get("task_goal"), (list, tuple)) else str(info.get("task_goal", ""))
    prev_subgoal    = None
    subtasks_seen   = []
    subtasks_done   = []
    step_count      = 0

    # --- initial subgoal from reset info ---
    init_subgoal = str(info.get("simple_subgoal_online", ""))
    if init_subgoal:
        subtasks_seen.append(init_subgoal)
        prev_subgoal = init_subgoal

    while step_count < max_steps:
        cont_t, cat_t, current_subgoal = build_obs_tensor(
            obs, info, cam_intrinsics, prev_subgoal, vocabs, difficulty, device
        )

        # track subtask transitions
        if current_subgoal and current_subgoal != prev_subgoal:
            if prev_subgoal is not None:
                subtasks_done.append(prev_subgoal)   # completed previous subgoal
            if current_subgoal not in subtasks_seen:
                subtasks_seen.append(current_subgoal)
            prev_subgoal = current_subgoal

        action = policy.sample(cont_t, cat_t, n_steps=n_steps)
        action_np = action.squeeze(0).cpu().numpy().astype(np.float64)
        if step_count == 0:
            print(f"    [run_episode] first action shape={action_np.shape} values={action_np}", flush=True)

        obs, reward, terminated, truncated, info = env.step(action_np)
        step_count += 1

        status = info.get("status", "ongoing")
        if status == "error":
            return {
                "status": "error",
                "error_message": info.get("error_message", ""),
                "steps": step_count,
                "task_goal": task_goal,
                "subtasks_seen": subtasks_seen,
                "subtasks_done": subtasks_done,
            }
        if terminated or truncated:
            # mark last subgoal as done if episode succeeded
            if status == "success" and prev_subgoal and prev_subgoal not in subtasks_done:
                subtasks_done.append(prev_subgoal)
            return {
                "status": status,
                "steps": step_count,
                "task_goal": task_goal,
                "subtasks_seen":   subtasks_seen,
                "subtasks_done":   subtasks_done,
                "subtask_success_rate": len(subtasks_done) / max(len(subtasks_seen), 1),
            }

    return {
        "status": "timeout",
        "steps": step_count,
        "task_goal": task_goal,
        "subtasks_seen":   subtasks_seen,
        "subtasks_done":   subtasks_done,
        "subtask_success_rate": len(subtasks_done) / max(len(subtasks_seen), 1),
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

    # joint_angle avoids the second PandaArmMotionPlanningSolver that ee_pose creates,
    # which conflicts with the one DemonstrationWrapper already owns → segfault.
    env_builder = BenchmarkEnvBuilder(
        env_id       = "BinFill",
        dataset      = args.dataset,
        action_space = "joint_angle",
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
        print(f"{'Ep':>4}  {'Status':>8}  {'Steps':>6}  "
              f"{'SubtaskSR':>10}  {'Done/Seen':>10}  Task goal")
        print("-"*80)

        results = []
        for ep_num in episode_indices:
            # get difficulty from metadata (needed for categorical embedding)
            seed, difficulty = env_builder.resolve_episode(ep_num)
            difficulty = difficulty or "easy"

            print(f"  [ep {ep_num}] make_env_for_episode ...", flush=True)
            env = env_builder.make_env_for_episode(
                ep_num,
                max_steps                 = args.max_steps,
                include_front_camera_intrinsic = True,
                include_wrist_camera_intrinsic = True,
            )
            print(f"  [ep {ep_num}] env created, running episode ...", flush=True)
            try:
                result = run_episode(
                    env, policy, vocabs, difficulty,
                    device, args.n_inference_steps, args.max_steps,
                )
            except Exception as e:
                result = {"status": "error", "error_message": str(e), "steps": 0,
                          "task_goal": "", "subtasks_seen": [], "subtasks_done": [],
                          "subtask_success_rate": 0.0}
            finally:
                env.close()

            result["episode"] = ep_num
            result["difficulty"] = difficulty
            result.setdefault("subtask_success_rate",
                              len(result["subtasks_done"]) / max(len(result["subtasks_seen"]), 1))
            results.append(result)

            status_icon = "✓" if result["status"] == "success" else "✗"
            print(
                f"{ep_num:>4d}  {result['status']:>8}  {result.get('steps',0):>6d}  "
                f"{result['subtask_success_rate']:>10.3f}  "
                f"{len(result['subtasks_done']):>4d}/{len(result['subtasks_seen']):<4d}  "
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
        print(f"  Subtask success rate : {np.mean(sr_list):.3f}  (mean across episodes)")
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
