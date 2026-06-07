"""
Episode-level evaluation of a saved FlowMatchingPolicy on the BinFill h5 dataset.

Since no live BinFill environment is available, evaluation is done by running the
policy sequentially through each held-out validation episode (open-loop):
  - At every timestep, feed the ground-truth obs → ODE sample → predicted action
  - Compare predicted action to the ground-truth demonstration action
  - Aggregate per-episode and overall statistics

Episode "success" is defined as: the fraction of timesteps within the L2
threshold exceeds --success_step_frac (default 0.9), i.e. the policy tracks
the demonstration closely for at least 90% of the episode.

Additional breakdowns reported:
  - Accuracy at subgoal boundary timesteps (is_subgoal_boundary=True)
  - Accuracy at task completion timestep (is_completed=True)
  - Per-episode mean L2 error histogram summary

Usage
-----
python examples/baselines/bc/eval_binfill_fm.py \
    --checkpoint runs/BinFill__bc_binfill_fm__1__<ts>/checkpoints/final.pt \
    --h5_file robomme_data/record_dataset_BinFill.h5
"""

import os
import sys
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
import tyro
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from bc_binfill import (
    ACTION_DIM, CAT_FIELDS, CONT_OBS_DIM, Vocab,
    _decode, _read_continuous_obs, _read_setup_cont,
    _read_setup_cats, _read_info_cats,
)
from bc_binfill_fm import FlowMatchingPolicy


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class Args:
    checkpoint: str
    """path to saved .pt checkpoint (required)"""
    h5_file: str = "robomme_data/record_dataset_BinFill.h5"
    action_key: str = "eef_action"
    val_fraction: float = 0.2
    """must match the value used during training"""
    num_demos: Optional[int] = None
    seed: int = 1
    """must match the value used during training to get the same val split"""
    cuda: bool = True

    # Model architecture — must match training args
    embed_dim: int = 16
    context_dim: int = 256
    hidden_dim: int = 256
    time_emb_dim: int = 64

    # Evaluation
    n_inference_steps: int = 10
    acc_threshold: float = 0.05
    """L2 threshold for counting one timestep as accurate"""
    success_step_frac: float = 0.9
    """min fraction of accurate timesteps to call an episode successful"""


# ---------------------------------------------------------------------------
# Episode-structured data loading
# ---------------------------------------------------------------------------

def load_episodes_for_eval(
    h5_file: str,
    action_key: str,
    num_demos: Optional[int],
    vocabs: Dict[str, "Vocab"],
) -> List[List[dict]]:
    """
    Returns a list of episodes. Each episode is a list of timestep dicts:
        cont_obs          : (37,) float32
        cat_ids           : {field: int}
        action            : (action_dim,) float32
        is_subgoal_boundary : bool
        is_completed        : bool
        simple_subgoal      : str
    Populates vocabs in-place (same as training).
    """
    episodes = []
    with h5py.File(h5_file, "r") as f:
        ep_keys = sorted(f.keys())
        if num_demos is not None:
            ep_keys = ep_keys[:num_demos]

        print(f"Loading {len(ep_keys)} episodes for evaluation ...")
        for ep_key in tqdm(ep_keys):
            ep = f[ep_key]
            timesteps = sorted(
                [k for k in ep.keys() if k.startswith("timestep_")],
                key=lambda x: int(x.split("_")[1]),
            )
            if not timesteps:
                continue

            setup_cont = _read_setup_cont(ep)
            setup_cats = _read_setup_cats(ep)
            for field, val in setup_cats.items():
                vocabs[field].add(val)

            ep_records = []
            for ts_key in timesteps:
                ts = ep[ts_key]

                act_node = ts["action"][action_key]
                if isinstance(act_node, h5py.Group):
                    continue
                action = act_node[()].astype(np.float32)

                cont_obs  = _read_continuous_obs(ts, setup_cont)
                info_cats = _read_info_cats(ts)
                for field, val in info_cats.items():
                    vocabs[field].add(val)

                cat_ids = {
                    field: vocabs[field](val)
                    for field, val in {**setup_cats, **info_cats}.items()
                }

                ep_records.append({
                    "cont_obs":            cont_obs,
                    "cat_ids":             cat_ids,
                    "action":              action,
                    "is_subgoal_boundary": bool(ts["info"]["is_subgoal_boundary"][()]),
                    "is_completed":        bool(ts["info"]["is_completed"][()]),
                    "simple_subgoal":      _decode(ts["info"]["simple_subgoal"][()]),
                })

            if ep_records:
                episodes.append(ep_records)

    return episodes


# ---------------------------------------------------------------------------
# Episode evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_episode(
    policy: FlowMatchingPolicy,
    episode: List[dict],
    device: torch.device,
    n_steps: int,
    threshold: float,
) -> dict:
    """Run the policy open-loop through one episode, return per-episode stats."""
    l2_errors, subgoal_errors, completed_errors = [], [], []

    for record in episode:
        cont = torch.from_numpy(record["cont_obs"]).unsqueeze(0).to(device)   # (1, 37)
        cat  = {f: torch.tensor([record["cat_ids"][f]], device=device)
                for f in CAT_FIELDS}
        gt   = torch.from_numpy(record["action"]).unsqueeze(0).to(device)     # (1, action_dim)

        pred = policy.sample(cont, cat, n_steps=n_steps)                      # (1, action_dim)
        l2   = (pred - gt).norm(dim=-1).item()
        l2_errors.append(l2)

        if record["is_subgoal_boundary"]:
            subgoal_errors.append(l2)
        if record["is_completed"]:
            completed_errors.append(l2)

    l2_arr   = np.array(l2_errors)
    acc_mask = l2_arr < threshold

    return {
        "n_timesteps":        len(episode),
        "mean_l2":            float(l2_arr.mean()),
        "max_l2":             float(l2_arr.max()),
        "step_acc":           float(acc_mask.mean()),           # fraction within threshold
        "success":            float(acc_mask.mean() >= 0.0),    # filled in main
        "subgoal_mean_l2":    float(np.mean(subgoal_errors)) if subgoal_errors else float("nan"),
        "subgoal_acc":        float(np.mean(np.array(subgoal_errors) < threshold))
                              if subgoal_errors else float("nan"),
        "completed_mean_l2":  float(np.mean(completed_errors)) if completed_errors else float("nan"),
        "completed_acc":      float(np.mean(np.array(completed_errors) < threshold))
                              if completed_errors else float("nan"),
        "has_completion":     len(completed_errors) > 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    action_dim = ACTION_DIM[args.action_key]

    # ------------------------------------------------------------------
    # Load episodes (episode-structured, not flat)
    # ------------------------------------------------------------------
    vocabs: Dict[str, Vocab] = {field: Vocab() for field in CAT_FIELDS}
    all_episodes = load_episodes_for_eval(
        args.h5_file, args.action_key, args.num_demos, vocabs
    )
    print(f"Total episodes loaded: {len(all_episodes)}")

    # Reproduce the same train/val split as training
    rng   = np.random.default_rng(args.seed)
    order = rng.permutation(len(all_episodes)).tolist()
    n_val = max(1, int(len(all_episodes) * args.val_fraction))

    val_episodes   = [all_episodes[i] for i in order[:n_val]]
    train_episodes = [all_episodes[i] for i in order[n_val:]]
    print(f"Val episodes: {len(val_episodes)}   Train episodes: {len(train_episodes)}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    vocab_sizes = {field: len(vocabs[field]) for field in CAT_FIELDS}
    policy = FlowMatchingPolicy(
        action_dim   = action_dim,
        vocab_sizes  = vocab_sizes,
        embed_dim    = args.embed_dim,
        context_dim  = args.context_dim,
        hidden_dim   = args.hidden_dim,
        time_emb_dim = args.time_emb_dim,
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"Model loaded from {args.checkpoint}")
    print(f"ODE inference steps: {args.n_inference_steps}")
    print(f"L2 accuracy threshold: {args.acc_threshold}")
    print(f"Episode success threshold: step_acc >= {args.success_step_frac}\n")

    # ------------------------------------------------------------------
    # Run episode-level evaluation on val set
    # ------------------------------------------------------------------
    for split_name, episodes in [("VALIDATION", val_episodes), ("TRAIN", train_episodes)]:
        print("=" * 60)
        print(f"{split_name} SET  ({len(episodes)} episodes)")
        print("=" * 60)

        ep_results = []
        for ep in tqdm(episodes, desc=split_name):
            res = evaluate_episode(policy, ep, device, args.n_inference_steps, args.acc_threshold)
            res["success"] = float(res["step_acc"] >= args.success_step_frac)
            ep_results.append(res)

        # ---------- aggregate ----------
        n_ep       = len(ep_results)
        success_rate    = np.mean([r["success"]      for r in ep_results])
        mean_step_acc   = np.mean([r["step_acc"]     for r in ep_results])
        mean_l2         = np.mean([r["mean_l2"]      for r in ep_results])
        max_l2          = np.max( [r["max_l2"]       for r in ep_results])
        mean_ep_len     = np.mean([r["n_timesteps"]  for r in ep_results])

        sg_accs  = [r["subgoal_acc"]   for r in ep_results if not np.isnan(r["subgoal_acc"])]
        cmp_accs = [r["completed_acc"] for r in ep_results if r["has_completion"]]

        print(f"\n{'Metric':<35}  {'Value':>10}")
        print("-" * 48)
        print(f"{'Episode success rate':<35}  {success_rate:>10.4f}  "
              f"({int(success_rate*n_ep)}/{n_ep} episodes)")
        print(f"{'Mean step accuracy':<35}  {mean_step_acc:>10.4f}  "
              f"(% timesteps within L2 {args.acc_threshold})")
        print(f"{'Mean episode L2 error':<35}  {mean_l2:>10.6f}")
        print(f"{'Max episode L2 error':<35}  {max_l2:>10.6f}")
        print(f"{'Mean episode length':<35}  {mean_ep_len:>10.1f}  timesteps")
        if sg_accs:
            print(f"{'Subgoal boundary accuracy':<35}  {np.mean(sg_accs):>10.4f}  "
                  f"(accuracy at is_subgoal_boundary=True steps)")
        if cmp_accs:
            print(f"{'Task completion accuracy':<35}  {np.mean(cmp_accs):>10.4f}  "
                  f"(accuracy at is_completed=True steps)")

        # ---------- per-episode table (val only) ----------
        if split_name == "VALIDATION":
            print(f"\n{'Ep':>4}  {'Steps':>6}  {'StepAcc':>8}  {'MeanL2':>8}  "
                  f"{'SgAcc':>7}  {'Success':>8}  SubgoalText")
            print("-" * 80)
            for i, (ep, res) in enumerate(zip(episodes, ep_results)):
                subgoal_text = ep[0]["simple_subgoal"][:35]
                sg_acc_str   = f"{res['subgoal_acc']:.4f}" if not np.isnan(res["subgoal_acc"]) else "  n/a "
                print(
                    f"{i:>4d}  {res['n_timesteps']:>6d}  "
                    f"{res['step_acc']:>8.4f}  {res['mean_l2']:>8.6f}  "
                    f"{sg_acc_str:>7}  "
                    f"{'YES' if res['success'] else 'no':>8}  "
                    f"{subgoal_text}"
                )

        print()
