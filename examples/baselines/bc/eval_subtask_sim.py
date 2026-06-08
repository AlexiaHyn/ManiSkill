"""
Simulation evaluation of a trained SubtaskFlowMatchingPolicy on the RoboMME
BinFill environment.

This script is the live-sim counterpart of the offline validation in
bc_subtask_train.py. It runs the real BinFill environment and drives the robot
through full multi-subtask episodes using a TODO-queue approach.

TODO-queue approach
-------------------
At every episode:
  1. Reset environment → receive obs + info.
  2. Read initial grounded_subgoal_online from info → push as current subtask.
  3. Execute policy conditioned on (robot_state, current_subgoal) → joint_action.
  4. After env.step():
       - If info["is_subgoal_boundary"] is True:
           the current subtask was just completed; advance the queue
           (the new grounded_subgoal_online in info IS the next subtask).
         Increment completed_subtask counter.
       - If status in {"success","fail","timeout","error"} or terminated:
           episode is done.
  5. For episodes that end with status=="success" add +1 to completed count
     for the final subtask (no boundary transition fires after pressing the
     last button — success terminates the episode directly).

Metrics reported per episode and aggregated:
  episode_success_rate      fraction of episodes with status == "success"
  mean_subtasks_completed   avg subtask completions before episode end
  subtask_completion_rate   mean_subtasks_completed / expected_per_episode
                            (estimated from data if not passed explicitly)
  mean_episode_steps        avg timesteps per episode

Usage
-----
python examples/baselines/bc/eval_subtask_sim.py \\
    --checkpoint runs/BinFill_subtask__bc_subtask_train__1__<ts>/checkpoints/best_eval_fm_loss.pt \\
    --env_difficulty easy \\
    --num_eval_episodes 50

The checkpoint contains all normalisation stats as torch buffers; no separate
.npy files are required (though they are also saved alongside the checkpoint
for convenience).
"""

import os
import sys
import random
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import re
import numpy as np
import torch
import tyro

# ── imports from training module ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from bc_subtask_train import (
    SubtaskFlowMatchingPolicy,
    parse_grounded_subgoal,
    STATE_DIM, ACTION_DIM, ACTION_TYPE_VOCAB, COLOR_VOCAB,
    load_subtasks_from_h5,
)


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalArgs:
    checkpoint: str
    """Path to .pt file saved by bc_subtask_train.py  (required)"""

    # Environment
    dataset:           str           = "val"
    """Built-in split when h5_file is not set: train (100 ep) / val (50 ep) / test"""
    h5_file:           Optional[str] = None
    """If set, override episode metadata from this H5 file so env seeds/difficulties
    exactly match training data. Overrides --dataset."""
    max_episode_steps: int = 800
    """Hard cap on timesteps per episode (before timeout)"""
    num_eval_episodes: int = 20
    """Number of episodes to evaluate (capped at available count)"""
    expected_subtasks_per_episode: Optional[int] = None
    """For subtask_completion_rate. If None, estimated from mean of completed episodes."""

    # Policy inference
    n_inference_steps: int = 20
    """Euler ODE steps (more → smoother action, slower per step)"""

    # Model arch — must match the training run
    embed_dim:    int = 16
    context_dim:  int = 512
    hidden_dim:   int = 512
    time_emb_dim: int = 64

    seed: int  = 42
    cuda: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Rolling state buffer — obs_horizon stacking
# ─────────────────────────────────────────────────────────────────────────────

def _robot_state_from_obs(obs: dict) -> np.ndarray:
    """
    Build the 16-D proprioceptive state from a DemonstrationWrapper obs dict.

    DemonstrationWrapper computes eef_state via build_endeffector_pose_dict
    (sign-aligned quaternion + unwrapped continuous RPY), exactly matching the
    pipeline used during H5 data recording.  Ordering matches training:
      joint(7) + eef(6) + gripper(2) + is_gripper_close(1) = 16
    """
    joint   = np.asarray(obs["joint_state_list"][-1],   np.float32).ravel()[:7]   # (7,)
    eef     = np.asarray(obs["eef_state_list"][-1],     np.float32).ravel()[:6]   # (6,)
    gripper = np.asarray(obs["gripper_state_list"][-1], np.float32).ravel()[:2]   # (2,)
    # Use same threshold as RecordWrapper: closed if ANY joint < 0.03
    is_gc   = np.array([float(np.any(gripper < 0.03))], np.float32)               # (1,)
    return np.concatenate([joint, eef, gripper, is_gc])                            # (16,)


class StateBuffer:
    """
    Maintains a rolling window of the last `obs_horizon` normalised state
    vectors so the policy receives velocity/history information.

    reset(obs) fills the window with the initial robot state (padding matches
    training).  push(obs) is called after every env.step() when obs is not None.
    get_stacked() returns a (1, obs_horizon * STATE_DIM) tensor.

    obs must be the dict[str, list] returned by DemonstrationWrapper / FailAwareWrapper.
    """

    def __init__(
        self,
        obs_horizon: int,
        policy:      SubtaskFlowMatchingPolicy,
        device:      torch.device,
    ) -> None:
        self.obs_horizon = obs_horizon
        self._s_mean = policy.state_mean.cpu().numpy()
        self._s_std  = policy.state_std.cpu().numpy()
        self.device  = device
        self._buf: deque = deque(maxlen=obs_horizon)

    def _extract_normalised(self, obs: dict) -> np.ndarray:
        raw = _robot_state_from_obs(obs)
        return (raw - self._s_mean) / self._s_std

    def reset(self, obs: dict) -> None:
        """Fill window with initial robot state from a DemonstrationWrapper obs."""
        first = self._extract_normalised(obs)
        self._buf.clear()
        for _ in range(self.obs_horizon):
            self._buf.append(first.copy())

    def reset_from_h5(self, raw_state: np.ndarray) -> None:
        """Fill window with a pre-loaded H5 state (16-D raw, not normalised).
        Guarantees the initial observation exactly matches training data."""
        normalised = (raw_state.astype(np.float32) - self._s_mean) / self._s_std
        self._buf.clear()
        for _ in range(self.obs_horizon):
            self._buf.append(normalised.copy())

    def push(self, obs: Optional[dict]) -> None:
        """Append current robot state; no-op if obs is None (FailAwareWrapper error)."""
        if obs is None:
            return
        self._buf.append(self._extract_normalised(obs))

    def get_stacked(self) -> torch.Tensor:
        """Return (1, obs_horizon * STATE_DIM) float32 tensor."""
        stacked = np.concatenate(list(self._buf), axis=0)
        return torch.from_numpy(stacked).float().unsqueeze(0).to(self.device)


def _decode_info_str(val) -> str:
    """Safely decode bytes / numpy scalar info fields to str."""
    if isinstance(val, bytes):
        return val.decode("utf-8")
    if isinstance(val, np.ndarray):
        item = val.item()
        return item.decode("utf-8") if isinstance(item, bytes) else str(item)
    return str(val) if val is not None else ""



# ─────────────────────────────────────────────────────────────────────────────
# SubtaskQueue — TODO-queue for one episode
# ─────────────────────────────────────────────────────────────────────────────

class SubtaskQueue:
    """
    Maintains the active subtask and tracks completions for one episode.

    Usage
    -----
    queue = SubtaskQueue(initial_subgoal_text)
    ...
    # when info["is_subgoal_boundary"] is True after env.step():
    queue.advance(info["grounded_subgoal_online"])
    ...
    # at episode end:
    total = queue.completed + (1 if episode_success else 0)
    """

    def __init__(self, initial_subgoal_text: str):
        self.current_text   = initial_subgoal_text
        self.current_parsed = parse_grounded_subgoal(initial_subgoal_text)
        self.completed      = 0          # subtask completions signalled by is_subgoal_boundary
        self._history:      List[str] = [initial_subgoal_text]

    def advance(self, new_subgoal_text: str) -> None:
        """
        Called when is_subgoal_boundary=True in info.
        Increments completed counter and switches to the new subgoal.
        """
        self.completed     += 1
        self.current_text   = new_subgoal_text
        self.current_parsed = parse_grounded_subgoal(new_subgoal_text)
        self._history.append(new_subgoal_text)

    def subgoal_tensors(self, device: torch.device):
        """Return (action_type, color, pixel) tensors with batch dim=1."""
        sg          = self.current_parsed
        action_type = torch.tensor([sg["action_type"]], dtype=torch.long,    device=device)
        color       = torch.tensor([sg["color"]],       dtype=torch.long,    device=device)
        pixel       = torch.tensor(
            [[sg["pixel_y"], sg["pixel_x"]]], dtype=torch.float32, device=device
        )
        return action_type, color, pixel

    def history_str(self, max_chars: int = 90) -> str:
        h = " → ".join(self._history)
        return h[:max_chars] + ("…" if len(h) > max_chars else "")


# ─────────────────────────────────────────────────────────────────────────────
# H5 → BenchmarkEnvBuilder metadata bridge
# ─────────────────────────────────────────────────────────────────────────────

def _build_env_builder_from_h5(
    h5_file: str,
    num_episodes: Optional[int] = None,
) -> "BenchmarkEnvBuilder":
    """
    Load seed/difficulty from an H5 recording file and create a
    BenchmarkEnvBuilder whose episode metadata exactly matches the H5 episodes.

    BenchmarkEnvBuilder normally reads from its bundled env_metadata JSON files.
    By writing a temporary metadata JSON derived from the H5 file and passing it
    via override_metadata_path, we guarantee that make_env_for_episode(ep_idx)
    recreates the identical scene (same seed + difficulty) that was used during
    data collection.

    Returns the BenchmarkEnvBuilder instance.  Caller is responsible for
    deleting the temp directory if desired (not critical — it is very small).
    """
    import json
    import tempfile
    from pathlib import Path
    from robomme.env_record_wrapper.episode_config_resolver import BenchmarkEnvBuilder

    _, episode_setups = load_subtasks_from_h5(h5_file, num_episodes)

    records = []
    for setup in episode_setups:
        ep_key = setup["episode_key"]          # e.g. "episode_42"
        ep_idx = int(ep_key.split("_")[1])
        records.append({
            "task":       "BinFill",
            "episode":    ep_idx,
            "seed":       setup["seed"],
            "difficulty": setup["difficulty"],
        })

    metadata = {
        "env_id":       "BinFill",
        "record_count": len(records),
        "records":      records,
    }

    # Write to a temp dir; BenchmarkEnvBuilder expects the directory, then
    # appends "record_dataset_BinFill_metadata.json" itself.
    tmp_dir = Path(tempfile.mkdtemp(prefix="binfill_meta_"))
    json_path = tmp_dir / "record_dataset_BinFill_metadata.json"
    json_path.write_text(json.dumps(metadata, indent=2))

    return BenchmarkEnvBuilder(
        "BinFill",
        dataset                = "train",   # satisfies the validation check; real data comes from override
        override_metadata_path = tmp_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-episode runner
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_episode(
    env,
    policy:      SubtaskFlowMatchingPolicy,
    device:      torch.device,
    n_steps:     int,
    max_steps:   int,
    obs_horizon: int,
) -> Dict:
    """
    Execute one full episode driven by the subtask-queue policy.

    env must be created via BenchmarkEnvBuilder.make_env_for_episode(), which
    wraps BinFill with DemonstrationWrapper (providing grounded_subgoal_online)
    and FailAwareWrapper.

    Returns a dict of per-episode metrics.
    """
    obs, info = env.reset()

    # ── One-time diagnostic: print obs and info structure after reset ─────
    if not hasattr(run_episode, "_reset_printed"):
        run_episode._reset_printed = True
        print("\n" + "=" * 60)
        print("OBS keys after env.reset():")
        if isinstance(obs, dict):
            for k, v in obs.items():
                import numpy as _np
                if hasattr(v, "__len__"):
                    try:
                        arr = _np.asarray(v)
                        print(f"  {k:<30} shape={arr.shape}  dtype={arr.dtype}"
                              f"  sample={arr.flat[0]:.4f}")
                    except Exception:
                        print(f"  {k:<30} len={len(v)}  type={type(v[0]).__name__}")
                else:
                    print(f"  {k:<30} value={v!r}")
        else:
            print(f"  (not a dict)  type={type(obs).__name__}")
        print("\nINFO keys after env.reset():")
        if isinstance(info, dict):
            for k, v in info.items():
                print(f"  {k:<30} type={type(v).__name__}  value={str(v)[:80]!r}")
        else:
            print(f"  (not a dict)  type={type(info).__name__}")
        print("=" * 60 + "\n")
    # ─────────────────────────────────────────────────────────────────────

    # Track env.unwrapped.timestep for task boundary detection.
    # DemonstrationWrapper sets allow_subgoal_change_this_timestep=True, so
    # grounded_subgoal_online updates in the same step that a task completes.
    _prev_task_idx = int(getattr(env.unwrapped, "timestep", 0))

    # Initialise rolling state buffer from DemonstrationWrapper obs.
    state_buf = StateBuffer(obs_horizon, policy, device)
    state_buf.reset(obs)

    # Read initial subgoal from DemonstrationWrapper.
    initial_sg = _decode_info_str(info.get("grounded_subgoal_online", ""))
    if not initial_sg:
        initial_sg = "pick up the object"
    queue = SubtaskQueue(initial_sg)

    ep_steps        = 0
    episode_success = False
    status          = "ongoing"

    for _ in range(max_steps):
        # ── Build policy inputs (all with batch dim=1) ────────────────────
        state_tensor              = state_buf.get_stacked()   # (1, obs_horizon * STATE_DIM)
        action_type, color, pixel = queue.subgoal_tensors(device)

        # ── Sample action via Euler ODE ───────────────────────────────────
        action_tensor = policy.sample(
            state_tensor, action_type, color, pixel,
            n_steps=n_steps,
        )
        action_np = action_tensor.squeeze(0).cpu().numpy()   # (8,)

        # ── Step environment ──────────────────────────────────────────────
        obs, _reward, terminated, truncated, info = env.step(action_np)
        ep_steps += 1
        # DemonstrationWrapper returns scalar bool tensors; normalise to Python bool.
        if isinstance(terminated, torch.Tensor):
            terminated = bool(terminated.any().item())
        if isinstance(truncated, torch.Tensor):
            truncated = bool(truncated.any().item())

        # Update rolling buffer (obs is None on FailAwareWrapper error; push is a no-op).
        state_buf.push(obs)

        # ── Check for subgoal boundary ────────────────────────────────────
        cur_task_idx = int(getattr(env.unwrapped, "timestep", 0))
        is_boundary  = (cur_task_idx != _prev_task_idx)
        _prev_task_idx = cur_task_idx

        if is_boundary:
            new_sg = _decode_info_str(info.get("grounded_subgoal_online", ""))

            # One-time diagnostic log
            if not hasattr(run_episode, "_subgoal_logged"):
                run_episode._subgoal_logged = True
                parsed = parse_grounded_subgoal(new_sg)
                print(f"[SUBGOAL CHECK] boundary → '{new_sg}'")
                print(f"[SUBGOAL CHECK] parsed : action_type={parsed['action_type']}"
                      f"  color={parsed['color']}"
                      f"  pixel=({parsed['pixel_y']:.4f}, {parsed['pixel_x']:.4f})")

            # Advance only when a NEW distinct subgoal appears.
            # When the last task completes, grounded_subgoal_online stays the same
            # text, so new_sg == queue.current_text → we skip here.
            # The final subtask is counted via episode_success below.
            if new_sg and new_sg != queue.current_text:
                queue.advance(new_sg)

        # ── Check episode termination ─────────────────────────────────────
        # DemonstrationWrapper always sets info["status"] to one of:
        #   "success" | "fail" | "timeout" | "ongoing" | "error"
        status = _decode_info_str(info.get("status", "ongoing"))

        if terminated or truncated or status in ("success", "fail", "timeout", "error"):
            episode_success = (status == "success")
            break

    # The final subtask has no subsequent boundary transition (the episode ends
    # instead). If the episode succeeded, the last subtask was completed.
    subtasks_completed = queue.completed + (1 if episode_success else 0)

    return {
        "episode_success":      episode_success,
        "steps":                ep_steps,
        "subtasks_completed":   subtasks_completed,
        "boundary_transitions": queue.completed,   # raw boundary count (excluding last)
        "final_status":         status,
        "subtask_history":      queue.history_str(),
        "n_unique_subtasks":    len(queue._history),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = tyro.cli(EvalArgs)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device          : {device}")
    print(f"Checkpoint      : {args.checkpoint}")
    print(f"Dataset         : {args.h5_file if args.h5_file else args.dataset}")
    print(f"Eval episodes   : {args.num_eval_episodes}")
    print(f"Max ep steps    : {args.max_episode_steps}")
    print(f"ODE steps       : {args.n_inference_steps}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt       = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["policy_state_dict"] if isinstance(ckpt, dict) and "policy_state_dict" in ckpt else ckpt

    # Read obs_horizon from checkpoint buffers (backwards-compatible with old
    # checkpoints that lacked obs_horizon_buf → fall back to 1)
    obs_horizon = int(state_dict.get("obs_horizon_buf", torch.tensor(1)).item())
    print(f"obs_horizon     : {obs_horizon}")

    # ── Instantiate and load policy ───────────────────────────────────────
    policy = SubtaskFlowMatchingPolicy(
        obs_horizon  = obs_horizon,
        embed_dim    = args.embed_dim,
        context_dim  = args.context_dim,
        hidden_dim   = args.hidden_dim,
        time_emb_dim = args.time_emb_dim,
    ).to(device)

    policy.load_state_dict(state_dict)
    policy.eval()

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy loaded   : {n_params:,} parameters")
    print(f"Action mean     : {policy.action_mean.cpu().numpy()}")
    print(f"Action std      : {policy.action_std.cpu().numpy()}")

    # ── Benchmark simulation evaluation ──────────────────────────────────
    # If --h5_file is given, episode seeds/difficulties are taken directly from
    # the H5 recording so scenes exactly match training data.
    # Otherwise, BenchmarkEnvBuilder's built-in split metadata is used.
    if args.h5_file is not None:
        print(f"\nBuilding env metadata from H5: {args.h5_file}")
        env_builder = _build_env_builder_from_h5(args.h5_file, args.num_eval_episodes)
        split_desc  = f"H5({args.h5_file})"
    else:
        try:
            from robomme.env_record_wrapper.episode_config_resolver import BenchmarkEnvBuilder
        except ImportError as exc:
            print(f"\n[ERROR] Cannot import BenchmarkEnvBuilder: {exc}")
            print("  Ensure the robomme package is installed:")
            print("    cd <robomme_benchmark>  &&  pip install -e .")
            raise
        env_builder = BenchmarkEnvBuilder("BinFill", dataset=args.dataset)
        split_desc  = args.dataset

    n_available  = env_builder.get_episode_num()
    n_to_eval    = min(args.num_eval_episodes, n_available)

    print(f"\nDataset         : {split_desc}  ({n_available} episodes available)")
    print(f"Evaluating      : {n_to_eval} episodes\n")

    col = (f"{'Ep':>4}  {'Steps':>5}  {'SubtasksDone':>12}  "
           f"{'Boundaries':>10}  {'Status':>9}  Subgoal history")
    print(col)
    print("-" * 110)

    results: List[Dict] = []
    for ep_idx in range(n_to_eval):
        _env = env_builder.make_env_for_episode(ep_idx)
        r = run_episode(
            env         = _env,
            policy      = policy,
            device      = device,
            n_steps     = args.n_inference_steps,
            max_steps   = args.max_episode_steps,
            obs_horizon = obs_horizon,
        )
        _env.close()
        results.append(r)
        print(
            f"{ep_idx:>4d}"
            f"  {r['steps']:>5d}"
            f"  {r['subtasks_completed']:>12d}"
            f"  {r['boundary_transitions']:>10d}"
            f"  {'SUCCESS' if r['episode_success'] else r['final_status']:>9}"
            f"  {r['subtask_history']}"
        )

    n_ep               = len(results)
    successes          = [r["episode_success"]    for r in results]
    subtasks_done_list = [r["subtasks_completed"] for r in results]
    steps_list         = [r["steps"]              for r in results]

    success_rate           = float(np.mean(successes))
    mean_subtasks_done     = float(np.mean(subtasks_done_list))
    total_subtasks_done    = int(np.sum(subtasks_done_list))
    mean_steps             = float(np.mean(steps_list))
    status_counts          = Counter(r["final_status"] for r in results)

    if args.expected_subtasks_per_episode is not None:
        expected = args.expected_subtasks_per_episode
    else:
        success_done = [subtasks_done_list[i] for i, s in enumerate(successes) if s]
        expected = int(np.max(subtasks_done_list)) if not success_done else int(np.mean(success_done))
    subtask_completion_rate = mean_subtasks_done / max(expected, 1)

    print("\n" + "=" * 64)
    print(f"EVAL SUMMARY  [{args.dataset}]")
    print("=" * 64)
    print(f"  Episodes evaluated         : {n_ep}")
    print(f"  Episode success rate       : {success_rate:.3f}"
          f"  ({int(success_rate*n_ep)}/{n_ep})")
    print(f"  Mean subtasks completed    : {mean_subtasks_done:.2f}  per episode")
    print(f"  Expected subtasks/episode  : ~{expected}")
    print(f"  Subtask completion rate    : {subtask_completion_rate:.3f}")
    print(f"  Total subtasks completed   : {total_subtasks_done}")
    print(f"  Mean episode length        : {mean_steps:.1f}  steps")
    print(f"\n  Episode status breakdown:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"    {status:<12}: {count:>4}  ({count/n_ep:.1%})")
    print("\n  Per-episode subtask completions:")
    for r in results:
        bar = "█" * r["subtasks_completed"]
        print(f"    {bar:<20}  {r['subtasks_completed']:>2}  "
              f"{'✓' if r['episode_success'] else '✗'}")
    print("=" * 64)
