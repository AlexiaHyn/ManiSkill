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

import numpy as np
import torch
import tyro
from scipy.spatial.transform import Rotation as _Rotation
from scipy.spatial.transform import Rotation as _Rotation

# ── imports from training module ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from bc_subtask_train import (
    SubtaskFlowMatchingPolicy,
    parse_grounded_subgoal,
    STATE_DIM, ACTION_DIM, ACTION_TYPE_VOCAB, COLOR_VOCAB,
)


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalArgs:
    checkpoint: str
    """Path to .pt file saved by bc_subtask_train.py  (required)"""

    # Environment
    env_difficulty:    str = "easy"
    """BinFill difficulty level: easy / medium / hard"""
    max_episode_steps: int = 800
    """Hard cap on timesteps per episode (before timeout)"""
    num_eval_episodes: int = 20
    """Number of episodes to evaluate"""
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

    # Action format passed to env.step()
    # "array"  → numpy array of shape (8,) — most common
    # "dict"   → {"joint_action": array}   — if env expects a dict
    action_format: str = "array"


# ─────────────────────────────────────────────────────────────────────────────
# Rolling state buffer — obs_horizon stacking
# ─────────────────────────────────────────────────────────────────────────────

def _robot_state_from_env(env) -> np.ndarray:
    """
    Build the 16-D proprioceptive state from SAPIEN sim internals.

    The live env flat obs is qpos(9)+qvel(9)=18 and has no eef_state, so we
    bypass obs and read tcp.pose for forward-kinematics (same approach as
    rollout_eval_binfill_fm.py).  Ordering matches training:
      joint(7) + eef(6) + gripper(2) + is_gripper_close(1) = 16
    """
    base  = env.unwrapped
    robot = base.agent.robot
    tcp   = base.agent.tcp

    qpos          = np.asarray(robot.qpos).astype(np.float32).ravel()
    joint_state   = qpos[:7]                                                       # (7,)
    gripper_state = qpos[7:9] if len(qpos) > 7 else np.zeros(2, np.float32)       # (2,)

    p   = np.asarray(tcp.pose.p).astype(np.float32).ravel()[:3]                   # xyz
    q   = np.asarray(tcp.pose.q).astype(np.float32).ravel()                       # wxyz
    rpy = _Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz").astype(np.float32)
    eef_state = np.concatenate([p, rpy])                                           # (6,)

    is_gripper_close = np.array([float(gripper_state.mean() < 0.02)], np.float32) # (1,)

    return np.concatenate([joint_state, eef_state, gripper_state, is_gripper_close])  # (16,)


class StateBuffer:
    """
    Maintains a rolling window of the last `obs_horizon` normalised state
    vectors so the policy receives velocity/history information.

    reset(env) fills the window with the initial robot state (padding matches
    training).  push(env) is called after every env.step().
    get_stacked() returns a (1, obs_horizon * STATE_DIM) tensor.
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

    def _extract_normalised(self, env) -> np.ndarray:
        raw = _robot_state_from_env(env)
        return (raw - self._s_mean) / self._s_std

    def reset(self, env) -> None:
        """Fill window with initial robot state (padding matches training)."""
        first = self._extract_normalised(env)
        self._buf.clear()
        for _ in range(self.obs_horizon):
            self._buf.append(first.copy())

    def push(self, env) -> None:
        """Append current robot state; oldest is automatically evicted."""
        self._buf.append(self._extract_normalised(env))

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
# Single-episode runner
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_episode(
    env,
    policy:        SubtaskFlowMatchingPolicy,
    device:        torch.device,
    n_steps:       int,
    max_steps:     int,
    action_format: str,
    obs_horizon:   int,
) -> Dict:
    """
    Execute one full episode driven by the subtask-queue policy.

    Boundary detection
    ------------------
    is_subgoal_boundary=True in the info returned by env.step() means:
      - The step that was just taken was part of a subtask that has now ended.
      - The NEW subgoal is already reflected in grounded_subgoal_online.
    We advance the queue immediately so the NEXT action uses the new subgoal.

    Returns a dict of per-episode metrics.
    """
    obs, info = env.reset()

    # Initialise rolling state buffer (reads SAPIEN robot state, not flat obs tensor)
    state_buf = StateBuffer(obs_horizon, policy, device)
    state_buf.reset(env)

    # Initialise TODO queue from the very first subgoal
    initial_sg = _decode_info_str(info.get("grounded_subgoal_online", "pick up the object"))
    queue      = SubtaskQueue(initial_sg)

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
        if action_format == "dict":
            step_action = {"joint_action": action_np}
        else:
            step_action = action_np

        obs, _reward, terminated, truncated, info = env.step(step_action)
        ep_steps += 1

        # Update rolling buffer from SAPIEN state after the step
        state_buf.push(env)

        # ── Check for subgoal boundary ────────────────────────────────────
        is_boundary = info.get("is_subgoal_boundary", False)
        if isinstance(is_boundary, np.ndarray):
            is_boundary = bool(is_boundary.item())
        is_boundary = bool(is_boundary)

        if is_boundary:
            # The subtask that was just running is complete.
            # grounded_subgoal_online already holds the NEW subgoal.
            new_sg = _decode_info_str(info.get("grounded_subgoal_online", queue.current_text))
            if new_sg != queue.current_text:
                queue.advance(new_sg)
            else:
                # Boundary fired but text didn't change (end of episode boundary).
                # Count it anyway — the task is done.
                queue.completed += 1

        # ── Check episode termination ─────────────────────────────────────
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
    print(f"Difficulty      : {args.env_difficulty}")
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

    # ── Create BinFill environment ────────────────────────────────────────
    # Adjust the import / constructor to match your local robomme installation.
    # The env must expose the standard gym interface:
    #   obs, info = env.reset()
    #   obs, reward, terminated, truncated, info = env.step(action)
    # with info containing:
    #   grounded_subgoal_online  (str / bytes)
    #   is_subgoal_boundary      (bool)
    #   status                   (str: "success"/"fail"/"timeout"/"ongoing"/"error")
    try:
        from robomme.robomme_env.BinFill import BinFill
        env = BinFill(difficulty=args.env_difficulty, seed=args.seed)
        print(f"\nEnvironment     : BinFill  (difficulty={args.env_difficulty})\n")
    except ImportError as exc:
        print(f"\n[ERROR] Cannot import RoboMME BinFill: {exc}")
        print("  Ensure the robomme package is installed:")
        print("    cd <robomme_benchmark>  &&  pip install -e .")
        raise

    # ── Evaluation loop ───────────────────────────────────────────────────
    col = (f"{'Ep':>4}  {'Steps':>5}  {'SubtasksDone':>12}  "
           f"{'Boundaries':>10}  {'Status':>9}  Subgoal history")
    print(col)
    print("-" * 110)

    results: List[Dict] = []
    for ep_idx in range(args.num_eval_episodes):
        r = run_episode(
            env           = env,
            policy        = policy,
            device        = device,
            n_steps       = args.n_inference_steps,
            max_steps     = args.max_episode_steps,
            action_format = args.action_format,
            obs_horizon   = obs_horizon,
        )
        results.append(r)
        print(
            f"{ep_idx:>4d}"
            f"  {r['steps']:>5d}"
            f"  {r['subtasks_completed']:>12d}"
            f"  {r['boundary_transitions']:>10d}"
            f"  {'SUCCESS' if r['episode_success'] else r['final_status']:>9}"
            f"  {r['subtask_history']}"
        )

    env.close()

    # ── Aggregate metrics ─────────────────────────────────────────────────
    n_ep   = len(results)
    successes            = [r["episode_success"]    for r in results]
    subtasks_done_list   = [r["subtasks_completed"] for r in results]
    steps_list           = [r["steps"]              for r in results]

    success_rate         = float(np.mean(successes))
    mean_subtasks_done   = float(np.mean(subtasks_done_list))
    total_subtasks_done  = int(np.sum(subtasks_done_list))
    mean_steps           = float(np.mean(steps_list))
    status_counts        = Counter(r["final_status"] for r in results)

    # Subtask completion rate
    if args.expected_subtasks_per_episode is not None:
        expected = args.expected_subtasks_per_episode
    else:
        # Estimate from successful episodes; fall back to max observed
        success_done = [subtasks_done_list[i] for i, s in enumerate(successes) if s]
        expected = int(np.max(subtasks_done_list)) if not success_done else int(np.mean(success_done))
    subtask_completion_rate = mean_subtasks_done / max(expected, 1)

    print("\n" + "=" * 64)
    print("EVALUATION SUMMARY")
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

    # Per-difficulty breakdown (all same here, but useful if mixed)
    print("\n  Per-episode subtask completions:")
    for r in results:
        bar_len = r["subtasks_completed"]
        bar     = "█" * bar_len
        print(f"    {bar:<20}  {r['subtasks_completed']:>2}  "
              f"{'✓' if r['episode_success'] else '✗'}")

    print("=" * 64)
