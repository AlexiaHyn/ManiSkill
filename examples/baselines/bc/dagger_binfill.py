"""
Online DAgger for RoboMME BinFill with rich state observations.

DAgger (Dataset Aggregation) alternates between:
  1. Collecting expert trajectories via the motion planner from initial env states
  2. Rolling out the current policy in the live environment
  3. Labeling policy-visited states with expert actions (time-aligned)
  4. Aggregating all labeled data and retraining the policy

Rich state observation (116-D continuous + 4 categorical):
  Continuous (116-D):
    robot state (16)       = eef(6) + joint(7) + grip(2) + is_gripper_close(1)
    cube positions (36)    = up to MAX_CUBES cubes × xyz, padded with zeros
    cube colors (36)       = up to MAX_CUBES cubes × [is_red, is_blue, is_green]
    bin position (3)       = board_with_hole.pose.p (xyz)
    button position (3)    = button.pose.p (xyz)
    camera intrinsics (18) = front_intrinsic(9) + wrist_intrinsic(9)
    episode flags (4)      = is_completed + is_subgoal_boundary + is_video_demo + subgoal_count/12

  Categorical (4 fields):
    difficulty     from episode setup: easy / medium / hard
    task_goal      from episode setup: ultimate goal string
    subgoal_action parsed from live subgoal: pick / place / press / none
    subgoal_color  parsed from live subgoal: red / blue / green / none

Action space: joint_action (8-D joint positions + gripper open/close)
  Using joint space avoids IK conversion complexity when labeling expert actions.

Expert oracle:
  PandaArmMotionPlanningSolver + BinFill.task_list[i]["solve"](env, planner)
  called from the initial env state (after bypass_demo_reset).
  env.step is monkey-patched during expert execution to record (state, action) pairs.

DAgger rounds:
  Round 0: Pure expert demonstrations (no policy rollout)
  Round i>0:
    - Roll out policy for --episodes_per_round episodes
    - For each policy episode, run the same expert from scratch → expert trajectory
    - Time-align: policy state at step t → expert action at step t
    - Aggregate into growing dataset
    - Retrain for --iters_per_round gradient steps

Usage:
  python examples/baselines/bc/dagger_binfill.py \\
      --robomme_root /home/ubuntu/robomme_benchmark \\
      --h5_file /home/ubuntu/ManiSkill/record_dataset_BinFill.h5 \\
      --dagger_rounds 5 \\
      --episodes_per_round 20 \\
      --iters_per_round 5000

  Optionally warm-start from a BC checkpoint:
      --bc_checkpoint runs/BinFill__bc_binfill_fm__1__<ts>/checkpoints/final.pt
"""

import math
import os
import re
import sys
import time
import random
import argparse
import contextlib
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CUBES       = 12           # hard difficulty max
ACTION_DIM      = 8            # joint_action: 7 joint positions + 1 gripper
RICH_CONT_DIM   = 116          # see module docstring breakdown
GRIPPER_CLOSE_THRESH = 0.02   # metres

# Ordered: all red cubes, then blue, then green — matches BinFill._load_scene order
COLOR_IDX = {"red": 0, "blue": 1, "green": 2}
COLORS    = ["red", "blue", "green"]

# Categorical subgoal components
SUBGOAL_ACTIONS = ["pick", "place", "press", "none"]
SUBGOAL_COLORS  = ["red", "blue", "green", "none"]
RICH_CAT_FIELDS = ["difficulty", "task_goal", "subgoal_action", "subgoal_color"]


# ---------------------------------------------------------------------------
# Vocab (identical to bc_binfill.py)
# ---------------------------------------------------------------------------

class Vocab:
    """Maps string tokens to integer IDs (0 = UNK)."""
    def __init__(self):
        self._tok2id: Dict[str, int] = {}

    def add(self, token: str) -> None:
        if token not in self._tok2id:
            self._tok2id[token] = len(self._tok2id) + 1

    def __call__(self, token: str) -> int:
        return self._tok2id.get(token, 0)

    def __len__(self) -> int:
        return len(self._tok2id) + 1

    def state_dict(self) -> dict:
        return dict(self._tok2id)

    def load_state_dict(self, d: dict) -> None:
        self._tok2id = dict(d)


def _decode(raw) -> str:
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


# ---------------------------------------------------------------------------
# Subgoal parser
#
# Simple_subgoal strings observed in BinFill:
#   "pick up the 1st red cube"
#   "pick up the 2nd blue cube"
#   "put it into the bin"
#   "press the button"
# (Coordinates like "at <105, 118>" may be appended but we strip them.)
#
# Structured encoding → (action, count, color):
#   pick up the Nth COLOR cube → ("pick", N, COLOR)
#   put it into the bin        → ("place", 0, "none")
#   press the button           → ("press", 0, "none")
#   anything else              → ("none", 0, "none")
# ---------------------------------------------------------------------------

_ORDINALS = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
             "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10,
             "11th": 11, "12th": 12,
             "first": 1, "second": 2, "third": 3, "fourth": 4}


def parse_subgoal(subgoal: str) -> Tuple[str, int, str]:
    """
    Parse a simple_subgoal string into (action, count, color).

    Returns:
        action : "pick" | "place" | "press" | "none"
        count  : ordinal index (1-based) for pick, 0 otherwise
        color  : "red" | "blue" | "green" | "none"
    """
    s = re.sub(r"at\s*<[^>]*>", "", subgoal).lower().strip()

    if "press" in s or "button" in s:
        return "press", 0, "none"

    if "put" in s or "bin" in s or "place" in s:
        for c in COLORS:
            if c in s:
                m = re.search(r"(\d+)", s)
                cnt = int(m.group(1)) if m else 0
                return "place", cnt, c
        return "place", 0, "none"

    if "pick" in s or "grasp" in s or "grab" in s or "take" in s:
        color = "none"
        for c in COLORS:
            if c in s:
                color = c
                break
        count = 0
        for ord_str, val in _ORDINALS.items():
            if ord_str in s:
                count = val
                break
        if count == 0:
            m = re.search(r"(\d+)", s)
            if m:
                count = int(m.group(1))
        return "pick", count, color

    return "none", 0, "none"


def structured_subgoal_str(subgoal: str) -> str:
    """
    Convert subgoal string to compact structured form for logging/vocab.
    E.g.: "pick up the 1st red cube" → "pick,1,red"
          "put it into the bin"       → "place,0,none"
          "press the button"          → "press,0,none"
    """
    action, count, color = parse_subgoal(subgoal)
    return f"{action},{count},{color}"


# ---------------------------------------------------------------------------
# Rich state extraction from live environment
# ---------------------------------------------------------------------------

def _to_np(t, dtype=np.float32) -> np.ndarray:
    if hasattr(t, "detach"):
        t = t.detach().cpu()
    arr = np.asarray(t).flatten()
    return arr.astype(dtype)


def get_robot_state(env) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return eef(6), joint(7), gripper(2) from the live SAPIEN env."""
    from scipy.spatial.transform import Rotation
    base = env.unwrapped
    qpos = _to_np(base.agent.robot.qpos)
    joint  = qpos[:7]
    gripper = qpos[7:9] if len(qpos) > 7 else np.zeros(2, np.float32)
    p = _to_np(base.agent.tcp.pose.p)[:3]
    q = _to_np(base.agent.tcp.pose.q)  # wxyz
    rpy = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz").astype(np.float32)
    return np.concatenate([p, rpy]), joint, gripper


def get_scene_state(env) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return cube_positions(36), cube_colors(36), bin_pos(3), button_pos(3).

    Cube order: all red cubes (index 0…), then blue, then green.
    Each cube contributes [x, y, z] to positions and [is_red, is_blue, is_green] to colors.
    Slots beyond the actual cube count are zero-padded.
    """
    base = env.unwrapped
    cube_xyz   = np.zeros(MAX_CUBES * 3, np.float32)
    cube_color = np.zeros(MAX_CUBES * 3, np.float32)

    idx = 0
    for color_i, cube_list in enumerate([base.red_cubes, base.blue_cubes, base.green_cubes]):
        for cube in cube_list:
            if idx >= MAX_CUBES:
                break
            pos = _to_np(cube.pose.p)[:3]
            cube_xyz[idx * 3: idx * 3 + 3] = pos
            cube_color[idx * 3 + color_i] = 1.0
            idx += 1

    bin_pos    = _to_np(base.board_with_hole.pose.p)[:3]
    button_pos = _to_np(base.button.pose.p)[:3]
    return cube_xyz, cube_color, bin_pos, button_pos


def read_cam_intrinsics_from_h5(h5_file: str, ep_num: int) -> Dict[str, np.ndarray]:
    """Read camera intrinsics from ep['setup'] (same location as training code)."""
    ep_key  = f"episode_{ep_num}"
    fallback = {
        "front": np.eye(3, dtype=np.float32).ravel(),
        "wrist": np.eye(3, dtype=np.float32).ravel(),
    }
    with h5py.File(h5_file, "r") as f:
        if ep_key not in f or "setup" not in f[ep_key]:
            return fallback
        s = f[ep_key]["setup"]
        front = s["front_camera_intrinsic"][()].astype(np.float32).ravel()
        wrist = s["wrist_camera_intrinsic"][()].astype(np.float32).ravel()
    def _pad9(v):
        return v[:9] if len(v) >= 9 else np.pad(v, (0, 9 - len(v)))
    return {"front": _pad9(front), "wrist": _pad9(wrist)}


def build_rich_cont(
    env,
    cam_intrinsics: Dict[str, np.ndarray],
    is_completed: float,
    is_subgoal_boundary: float,
    subgoal_count: int,
) -> np.ndarray:
    """
    Build the 116-D continuous rich observation vector.

    Layout:
      [0:16]   robot state: eef(6) + joint(7) + grip(2) + is_close(1)
      [16:52]  cube positions: MAX_CUBES × 3
      [52:88]  cube colors:    MAX_CUBES × 3
      [88:91]  bin position
      [91:94]  button position
      [94:103] front camera intrinsic (9)
      [103:112] wrist camera intrinsic (9)
      [112:116] episode flags: is_completed + is_subgoal_boundary + is_video_demo + subgoal_count/12
    """
    eef, joint, grip = get_robot_state(env)
    cube_xyz, cube_color, bin_pos, btn_pos = get_scene_state(env)

    is_close = np.array([float(grip.mean() < GRIPPER_CLOSE_THRESH)], np.float32)

    cont = np.concatenate([
        eef, joint, grip, is_close,          # 16
        cube_xyz,                             # 36
        cube_color,                           # 36
        bin_pos,                              # 3
        btn_pos,                              # 3
        cam_intrinsics["front"],              # 9
        cam_intrinsics["wrist"],              # 9
        np.array([
            float(is_completed),
            float(is_subgoal_boundary),
            0.0,                              # is_video_demo = 0 during rollout
            float(subgoal_count) / 12.0,
        ], np.float32),                       # 4
    ]).astype(np.float32)

    assert len(cont) == RICH_CONT_DIM, f"Expected {RICH_CONT_DIM}-D, got {len(cont)}"
    return cont


def build_rich_cat_ids(
    difficulty: str,
    task_goal: str,
    subgoal_action: str,
    subgoal_color: str,
    vocabs: Dict[str, Vocab],
) -> Dict[str, int]:
    return {
        "difficulty":     vocabs["difficulty"](difficulty),
        "task_goal":      vocabs["task_goal"](task_goal),
        "subgoal_action": vocabs["subgoal_action"](subgoal_action),
        "subgoal_color":  vocabs["subgoal_color"](subgoal_color),
    }


# ---------------------------------------------------------------------------
# Episode metadata from h5
# ---------------------------------------------------------------------------

def read_episode_meta(h5_file: str, ep_num: int) -> Dict:
    """Read difficulty and task_goal from episode setup."""
    ep_key = f"episode_{ep_num}"
    with h5py.File(h5_file, "r") as f:
        if ep_key not in f:
            return {"difficulty": "easy", "task_goal": ""}
        s = f[ep_key]["setup"]
        diff = _decode(s["difficulty"][()])
        tg_raw = s["task_goal"]
        tg = _decode(tg_raw[0] if tg_raw.shape[0] > 0 else tg_raw[()])
    return {"difficulty": diff, "task_goal": tg}


# ---------------------------------------------------------------------------
# DemonstrationWrapper bypass (same as rollout_eval_binfill_fm.py)
# ---------------------------------------------------------------------------

def find_demo_wrapper(env):
    cur = env
    while cur is not None:
        if type(cur).__name__ == "DemonstrationWrapper":
            return cur
        cur = getattr(cur, "env", None)
    return None


def bypass_demo_reset(env):
    """Reset env without triggering DemonstrationWrapper demo replay."""
    demo_wrapper = find_demo_wrapper(env)
    if demo_wrapper is None:
        obs, info = env.reset()
        return obs, info, None

    demo_wrapper.last_subgoal_segment      = None
    demo_wrapper.latched_replacements      = None
    demo_wrapper._failed_match_save_count  = 0
    demo_wrapper.steps_without_demonstration = 0
    demo_wrapper._prev_ee_quat_wxyz        = None
    demo_wrapper._prev_ee_rpy_xyz          = None
    demo_wrapper.demonstration_data        = None

    obs, info = demo_wrapper.env.reset()
    demo_wrapper.episode_success = False
    return obs, info, demo_wrapper


# ---------------------------------------------------------------------------
# Expert trajectory collection
#
# Monkey-patches env.step to record (rich_cont, joint_action) pairs while the
# motion planner's solve functions drive the robot to complete each task.
# ---------------------------------------------------------------------------

def collect_expert_trajectory(
    env,
    ep_meta: Dict,
    cam_intrinsics: Dict[str, np.ndarray],
    vocabs: Dict[str, Vocab],
    max_steps: int = 2000,
    verbose: bool = False,
) -> List[Dict]:
    """
    Run the motion planner expert from the initial env state.
    Returns a list of {rich_cont, cat_ids, action} dicts (joint_action space).

    The expert uses BinFill.task_list[i]["solve"](env, planner) for each task.
    env.step is monkey-patched to record transitions before each physics step.
    """
    try:
        from mani_skill.examples.motionplanning.panda.motionplanner import (
            PandaArmMotionPlanningSolver,
        )
    except ImportError as e:
        raise RuntimeError(
            "Could not import PandaArmMotionPlanningSolver. "
            "Ensure mani_skill is installed."
        ) from e

    obs, info, demo_wrapper = bypass_demo_reset(env)

    planner = PandaArmMotionPlanningSolver(
        demo_wrapper if demo_wrapper is not None else env,
        debug=False,
        vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )

    transitions: List[Dict] = []
    step_count = [0]  # mutable cell for closure

    prev_subgoal = ""
    is_completed  = 0.0
    is_subgoal_boundary = 0.0
    subgoal_count = 0

    difficulty = ep_meta.get("difficulty", "easy")
    task_goal  = ep_meta.get("task_goal", "")

    original_step = env.step

    def recording_step(action):
        nonlocal prev_subgoal, is_completed, is_subgoal_boundary, subgoal_count

        if step_count[0] >= max_steps:
            return original_step(action)

        # Current subgoal from live env
        raw_subgoal = str(getattr(env.unwrapped, "current_task_name", "") or "")
        s_action, s_count, s_color = parse_subgoal(raw_subgoal)

        if raw_subgoal and raw_subgoal != prev_subgoal:
            is_subgoal_boundary = 1.0
            subgoal_count       = s_count
        else:
            is_subgoal_boundary = 0.0

        # Build rich obs BEFORE the step
        try:
            rich_cont = build_rich_cont(
                env, cam_intrinsics,
                is_completed, is_subgoal_boundary, subgoal_count,
            )
        except Exception:
            rich_cont = None

        cat_ids = build_rich_cat_ids(difficulty, task_goal, s_action, s_color, vocabs)

        # Normalise joint action to 8-D float64
        act_np = _to_np(action, dtype=np.float64)
        if len(act_np) < 8:
            act_np = np.pad(act_np, (0, 8 - len(act_np)))
        else:
            act_np = act_np[:8]

        result = original_step(action)

        obs_r, reward, terminated, truncated, info_r = result
        status = info_r.get("status", "ongoing") if isinstance(info_r, dict) else "ongoing"
        is_completed = 1.0 if status == "success" else 0.0

        prev_subgoal = str(info_r.get("simple_subgoal_online", raw_subgoal)) if isinstance(info_r, dict) else raw_subgoal

        if rich_cont is not None:
            transitions.append({
                "rich_cont": rich_cont,
                "cat_ids":   cat_ids,
                "action":    act_np.astype(np.float32),
            })

        step_count[0] += 1
        return result

    env.step = recording_step

    try:
        task_list = getattr(env.unwrapped, "task_list", [])
        for task in task_list:
            if step_count[0] >= max_steps:
                break
            solve_fn = task.get("solve")
            if solve_fn is None:
                continue
            try:
                solve_fn(demo_wrapper if demo_wrapper is not None else env, planner)
            except Exception as exc:
                if verbose:
                    print(f"[Expert] solve failed for task '{task.get('name')}': {exc}")
    finally:
        env.step = original_step
        try:
            planner.close()
        except Exception:
            pass

    if verbose:
        print(f"[Expert] collected {len(transitions)} steps")
    return transitions


# ---------------------------------------------------------------------------
# Policy rollout — collect (rich_cont, cat_ids) visited by the policy
# Returns states (for DAgger labeling) and episode outcome
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_policy_rollout(
    env,
    ep_meta: Dict,
    cam_intrinsics: Dict[str, np.ndarray],
    policy: "FlowMatchingPolicyRich",
    vocabs: Dict[str, Vocab],
    device: torch.device,
    max_steps: int,
    n_ode_steps: int,
) -> Tuple[List[Dict], str]:
    """
    Roll out the policy in the live env.
    Returns (state_list, outcome_status).
    state_list entries: {rich_cont, cat_ids}  — no action (will be labeled by expert)
    """
    obs, info, demo_wrapper = bypass_demo_reset(env)

    difficulty = ep_meta.get("difficulty", "easy")
    task_goal  = ep_meta.get("task_goal", "")

    prev_subgoal        = ""
    is_completed        = 0.0
    is_subgoal_boundary = 0.0
    subgoal_count       = 0
    states: List[Dict] = []

    for _ in range(max_steps):
        raw_subgoal = str(
            info.get("simple_subgoal_online", "")
            if isinstance(info, dict) else ""
        ) or str(getattr(env.unwrapped, "current_task_name", "") or "")

        s_action, s_count, s_color = parse_subgoal(raw_subgoal)

        if raw_subgoal and raw_subgoal != prev_subgoal:
            is_subgoal_boundary = 1.0
            subgoal_count       = s_count
        else:
            is_subgoal_boundary = 0.0

        try:
            rich_cont = build_rich_cont(
                env, cam_intrinsics,
                is_completed, is_subgoal_boundary, subgoal_count,
            )
        except Exception:
            break

        cat_ids = build_rich_cat_ids(difficulty, task_goal, s_action, s_color, vocabs)
        states.append({"rich_cont": rich_cont, "cat_ids": cat_ids})

        # Policy action
        cont_t = torch.from_numpy(rich_cont).unsqueeze(0).to(device)
        cat_t  = {f: torch.tensor([cat_ids[f]], device=device) for f in RICH_CAT_FIELDS}
        action = policy.sample(cont_t, cat_t, n_steps=n_ode_steps)
        act_np = action.squeeze(0).cpu().numpy().astype(np.float64)

        obs, reward, terminated, truncated, info = env.step(act_np)

        status = info.get("status", "ongoing") if isinstance(info, dict) else "ongoing"
        is_completed = 1.0 if status == "success" else 0.0

        prev_subgoal = str(info.get("simple_subgoal_online", raw_subgoal)) if isinstance(info, dict) else raw_subgoal

        if status in ("success", "fail", "error", "timeout"):
            return states, status

        term = bool(np.asarray(terminated).any()) if hasattr(terminated, "__iter__") else bool(terminated)
        trun = bool(np.asarray(truncated).any())  if hasattr(truncated,  "__iter__") else bool(truncated)
        if term or trun:
            return states, "fail"

    return states, "timeout"


# ---------------------------------------------------------------------------
# DAgger dataset
# ---------------------------------------------------------------------------

class DaggerDataset(Dataset):
    """Stores all aggregated (rich_cont, cat_ids, action) samples."""

    def __init__(self):
        self.rich_conts: List[np.ndarray]    = []
        self.cat_ids_list: List[Dict]        = []
        self.actions: List[np.ndarray]       = []

    def add_transitions(self, transitions: List[Dict]) -> None:
        for t in transitions:
            self.rich_conts.append(t["rich_cont"])
            self.cat_ids_list.append(t["cat_ids"])
            self.actions.append(t["action"])

    def add_dagger_pairs(
        self, policy_states: List[Dict], expert_actions: List[np.ndarray]
    ) -> None:
        """Add (policy_state, expert_action) pairs for DAgger."""
        n = min(len(policy_states), len(expert_actions))
        for i in range(n):
            s = policy_states[i]
            self.rich_conts.append(s["rich_cont"])
            self.cat_ids_list.append(s["cat_ids"])
            self.actions.append(expert_actions[i])

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> Tuple:
        cont   = torch.from_numpy(self.rich_conts[idx])
        cat    = {f: torch.tensor(self.cat_ids_list[idx][f]) for f in RICH_CAT_FIELDS}
        action = torch.from_numpy(self.actions[idx])
        return cont, cat, action


def dagger_collate(batch):
    conts, cats, actions = zip(*batch)
    cont_t   = torch.stack(conts)
    action_t = torch.stack(actions)
    cat_t = {f: torch.stack([c[f] for c in cats]) for f in RICH_CAT_FIELDS}
    return cont_t, cat_t, action_t


# ---------------------------------------------------------------------------
# Policy architecture (Flow Matching, rich obs)
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class RichObsEncoder(nn.Module):
    """
    Encodes (rich_cont 116-D, 4 categorical fields) → context vector.
    """
    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        embed_dim: int,
        context_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            f: nn.Embedding(vocab_sizes[f], embed_dim) for f in RICH_CAT_FIELDS
        })
        input_dim = RICH_CONT_DIM + len(RICH_CAT_FIELDS) * embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(
        self, cont: torch.Tensor, cat: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        parts = [cont] + [self.embeddings[f](cat[f]) for f in RICH_CAT_FIELDS]
        return self.net(torch.cat(parts, dim=-1))


class VectorFieldNet(nn.Module):
    def __init__(self, action_dim: int, context_dim: int, time_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_proj = nn.Sequential(nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU())
        in_dim = action_dim + time_emb_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_proj(sinusoidal_embedding(t, self.time_emb_dim))
        return self.net(torch.cat([x_t, t_emb, ctx], dim=-1))


class FlowMatchingPolicyRich(nn.Module):
    """
    Conditional Flow Matching policy with rich state input.
    Identical training/inference to bc_binfill_fm.py but with extended encoder.
    """
    def __init__(
        self,
        action_dim: int,
        vocab_sizes: Dict[str, int],
        embed_dim: int       = 16,
        context_dim: int     = 256,
        hidden_dim: int      = 256,
        time_emb_dim: int    = 64,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.encoder   = RichObsEncoder(vocab_sizes, embed_dim, context_dim, hidden_dim)
        self.vf_net    = VectorFieldNet(action_dim, context_dim, time_emb_dim, hidden_dim)

    def compute_loss(
        self,
        cont: torch.Tensor,
        cat: Dict[str, torch.Tensor],
        x_1: torch.Tensor,
    ) -> torch.Tensor:
        ctx = self.encoder(cont, cat)
        x_0 = torch.randn_like(x_1)
        t   = torch.rand(x_1.size(0), device=x_1.device)
        x_t = (1.0 - t[:, None]) * x_0 + t[:, None] * x_1
        u_t = x_1 - x_0
        return F.mse_loss(self.vf_net(x_t, t, ctx), u_t)

    @torch.no_grad()
    def sample(
        self,
        cont: torch.Tensor,
        cat: Dict[str, torch.Tensor],
        n_steps: int = 10,
    ) -> torch.Tensor:
        ctx = self.encoder(cont, cat)
        x   = torch.randn(cont.size(0), self.action_dim, device=cont.device)
        dt  = 1.0 / n_steps
        for k in range(n_steps):
            t = torch.full((cont.size(0),), k * dt, device=cont.device)
            x = x + self.vf_net(x, t, ctx) * dt
        return x


# ---------------------------------------------------------------------------
# Vocabulary bootstrap from h5 + pre-populate action/color tokens
# ---------------------------------------------------------------------------

def build_vocabs_from_h5(h5_file: str) -> Dict[str, Vocab]:
    """Build rich vocabs by scanning the h5 file."""
    vocabs = {f: Vocab() for f in RICH_CAT_FIELDS}

    # Pre-populate fixed-value vocabularies
    for a in SUBGOAL_ACTIONS:
        vocabs["subgoal_action"].add(a)
    for c in SUBGOAL_COLORS:
        vocabs["subgoal_color"].add(c)

    with h5py.File(h5_file, "r") as f:
        for ep_key in sorted(f.keys()):
            ep = f[ep_key]
            if "setup" not in ep:
                continue
            s   = ep["setup"]
            diff = _decode(s["difficulty"][()])
            tg_raw = s["task_goal"]
            tg  = _decode(tg_raw[0] if tg_raw.shape[0] > 0 else tg_raw[()])
            vocabs["difficulty"].add(diff)
            vocabs["task_goal"].add(tg)

    return vocabs


# ---------------------------------------------------------------------------
# Load H5 demonstrations as expert trajectories (re-simulated for rich state)
#
# We cannot read cube/bin/button positions from h5 (they are not stored).
# Instead we run each episode in the simulator, replaying joint_action from h5
# while recording rich state at each step.
# ---------------------------------------------------------------------------

def load_h5_demos_with_rich_state(
    h5_file: str,
    ep_indices: List[int],
    env_builder,
    vocabs: Dict[str, Vocab],
    max_steps_per_ep: int = 2000,
    verbose: bool = False,
) -> List[Dict]:
    """
    For each h5 episode in ep_indices:
      1. Create env with correct seed
      2. Replay the stored joint_action from h5 step-by-step
      3. At each step, record rich state + expert action

    Returns flat list of {rich_cont, cat_ids, action} dicts.
    """
    all_transitions: List[Dict] = []

    with h5py.File(h5_file, "r") as f_h5:
        ep_keys = sorted(f_h5.keys())

        for ep_idx in tqdm(ep_indices, desc="Loading H5 demos (re-simulated)"):
            if ep_idx >= len(ep_keys):
                continue
            ep_key = ep_keys[ep_idx]
            ep     = f_h5[ep_key]

            # Read setup
            s       = ep["setup"]
            diff    = _decode(s["difficulty"][()])
            tg_raw  = s["task_goal"]
            tg      = _decode(tg_raw[0] if tg_raw.shape[0] > 0 else tg_raw[()])
            front   = s["front_camera_intrinsic"][()].astype(np.float32).ravel()
            wrist   = s["wrist_camera_intrinsic"][()].astype(np.float32).ravel()
            cam_intr = {
                "front": front[:9] if len(front) >= 9 else np.pad(front, (0, 9 - len(front))),
                "wrist": wrist[:9] if len(wrist) >= 9 else np.pad(wrist, (0, 9 - len(wrist))),
            }

            # Collect stored actions (joint_action)
            ts_keys = sorted(
                [k for k in ep.keys() if k.startswith("timestep_")],
                key=lambda x: int(x.split("_")[1]),
            )
            if not ts_keys:
                continue

            stored_actions = []
            stored_subgoals = []
            for ts_k in ts_keys:
                ts  = ep[ts_k]
                act = ts["action"]["joint_action"][()].astype(np.float32)
                sg  = _decode(ts["info"]["simple_subgoal"][()])
                stored_actions.append(act)
                stored_subgoals.append(sg)

            # Create env and replay
            try:
                env = env_builder.make_env_for_episode(ep_idx, max_steps=max_steps_per_ep)
            except Exception as exc:
                if verbose:
                    print(f"[H5-reload] ep {ep_idx}: env build failed: {exc}")
                continue

            try:
                obs, info, demo_wrapper = bypass_demo_reset(env)
                ep_meta = {"difficulty": diff, "task_goal": tg}

                prev_subgoal        = ""
                is_completed        = 0.0
                is_subgoal_boundary = 0.0
                subgoal_count       = 0

                for step_i, (act_np, sg_h5) in enumerate(zip(stored_actions, stored_subgoals)):
                    if step_i >= max_steps_per_ep:
                        break

                    raw_subgoal = str(
                        info.get("simple_subgoal_online", "")
                        if isinstance(info, dict) else ""
                    ) or sg_h5

                    s_action, s_count, s_color = parse_subgoal(raw_subgoal)

                    if raw_subgoal and raw_subgoal != prev_subgoal:
                        is_subgoal_boundary = 1.0
                        subgoal_count       = s_count
                    else:
                        is_subgoal_boundary = 0.0

                    try:
                        rich_cont = build_rich_cont(
                            env, cam_intr,
                            is_completed, is_subgoal_boundary, subgoal_count,
                        )
                    except Exception:
                        break

                    cat_ids = build_rich_cat_ids(diff, tg, s_action, s_color, vocabs)

                    all_transitions.append({
                        "rich_cont": rich_cont,
                        "cat_ids":   cat_ids,
                        "action":    act_np,
                    })

                    obs, reward, terminated, truncated, info = env.step(act_np.astype(np.float64))

                    st = info.get("status", "ongoing") if isinstance(info, dict) else "ongoing"
                    is_completed  = 1.0 if st == "success" else 0.0
                    prev_subgoal  = str(info.get("simple_subgoal_online", raw_subgoal)) if isinstance(info, dict) else raw_subgoal

                    if st in ("success", "fail", "error", "timeout"):
                        break
                    term = bool(np.asarray(terminated).any()) if hasattr(terminated, "__iter__") else bool(terminated)
                    if term or (bool(np.asarray(truncated).any()) if hasattr(truncated, "__iter__") else bool(truncated)):
                        break

            except Exception as exc:
                if verbose:
                    print(f"[H5-reload] ep {ep_idx}: replay failed: {exc}")
            finally:
                try:
                    env.close()
                except Exception:
                    pass

    if verbose:
        print(f"[H5-reload] total transitions: {len(all_transitions)}")
    return all_transitions


# ---------------------------------------------------------------------------
# DAgger training utilities
# ---------------------------------------------------------------------------

def train_one_round(
    policy: FlowMatchingPolicyRich,
    dataset: DaggerDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_iters: int,
    batch_size: int,
    num_workers: int = 0,
) -> float:
    """Train policy on the current aggregated dataset for n_iters steps."""
    policy.train()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dagger_collate,
        num_workers=num_workers,
        drop_last=len(dataset) >= batch_size,
    )
    it_cycle  = iter(loader)
    total_loss = 0.0

    for _ in range(n_iters):
        try:
            cont, cat, action = next(it_cycle)
        except StopIteration:
            it_cycle = iter(loader)
            cont, cat, action = next(it_cycle)

        cont   = cont.to(device)
        cat    = {f: v.to(device) for f, v in cat.items()}
        action = action.to(device)

        optimizer.zero_grad()
        loss = policy.compute_loss(cont, cat, action)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(n_iters, 1)


# ---------------------------------------------------------------------------
# Episode-level success-rate evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_policy(
    env_builder,
    ep_indices: List[int],
    policy: FlowMatchingPolicyRich,
    vocabs: Dict[str, Vocab],
    device: torch.device,
    max_steps: int,
    n_ode_steps: int,
    max_episodes: Optional[int],
    h5_file: str,
    split_name: str = "eval",
    verbose: bool = False,
) -> Dict:
    """
    Run the policy on up to `max_episodes` episodes and report success rate.

    Returns dict with keys:
      success_rate  : float  fraction of episodes that ended with status=="success"
      n_success     : int
      n_total       : int
      n_fail        : int
      n_timeout     : int
      n_error       : int
      per_episode   : list of {ep_num, status}
    """
    policy.eval()
    eps = ep_indices[:max_episodes] if max_episodes is not None else ep_indices

    counts = {"success": 0, "fail": 0, "timeout": 0, "error": 0}
    per_ep: List[Dict] = []

    for ep_num in tqdm(eps, desc=f"Eval [{split_name}]"):
        cam_intr = read_cam_intrinsics_from_h5(h5_file, ep_num)
        ep_meta  = read_episode_meta(h5_file, ep_num)
        env      = None
        try:
            env = env_builder.make_env_for_episode(ep_num, max_steps=max_steps)
            _, outcome = collect_policy_rollout(
                env, ep_meta, cam_intr, policy, vocabs,
                device=device, max_steps=max_steps, n_ode_steps=n_ode_steps,
            )
        except Exception as exc:
            if verbose:
                print(f"[Eval ep {ep_num}] exception: {exc}")
            outcome = "error"
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

        bucket = outcome if outcome in counts else "error"
        counts[bucket] += 1
        per_ep.append({"ep_num": ep_num, "status": outcome})

    n_total = len(per_ep)
    sr      = counts["success"] / max(n_total, 1)
    print(
        f"  [{split_name}] success={counts['success']}/{n_total}  "
        f"({sr:.1%})  fail={counts['fail']}  "
        f"timeout={counts['timeout']}  error={counts['error']}"
    )
    return {
        "success_rate": sr,
        "n_success":    counts["success"],
        "n_total":      n_total,
        "n_fail":       counts["fail"],
        "n_timeout":    counts["timeout"],
        "n_error":      counts["error"],
        "per_episode":  per_ep,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Online DAgger for BinFill with rich state")
    p.add_argument("--robomme_root",       default="/home/ubuntu/robomme_benchmark")
    p.add_argument("--h5_file",            required=True)
    p.add_argument("--output_dir",         default="runs/dagger_binfill")
    p.add_argument("--bc_checkpoint",      default=None,
                   help="BC Flow Matching checkpoint to warm-start encoder "
                        "(obs dim mismatch is ignored; only shared weights are loaded)")

    # DAgger settings
    p.add_argument("--dagger_rounds",      type=int, default=5)
    p.add_argument("--episodes_per_round", type=int, default=20,
                   help="Policy rollout episodes per DAgger round")
    p.add_argument("--iters_per_round",    type=int, default=5000)
    p.add_argument("--max_steps_ep",       type=int, default=1300,
                   help="Max env steps per episode (rollout + expert)")
    p.add_argument("--n_ode_steps",        type=int, default=10)

    # Pre-training on H5 demos
    p.add_argument("--h5_pretrain_iters",  type=int, default=20000,
                   help="Gradient steps on H5 demos before any DAgger round")
    p.add_argument("--h5_episodes",        type=int, default=None,
                   help="Cap on H5 episodes to re-simulate (None = all train split)")
    p.add_argument("--val_fraction",       type=float, default=0.2)
    p.add_argument("--seed",               type=int, default=1)

    # Model
    p.add_argument("--embed_dim",          type=int, default=16)
    p.add_argument("--context_dim",        type=int, default=256)
    p.add_argument("--hidden_dim",         type=int, default=256)
    p.add_argument("--time_emb_dim",       type=int, default=64)

    # Optimiser
    p.add_argument("--lr",                 type=float, default=3e-4)
    p.add_argument("--batch_size",         type=int, default=256)
    p.add_argument("--num_workers",        type=int, default=0)

    # Evaluation
    p.add_argument("--eval_episodes",      type=int, default=20,
                   help="Number of episodes for train/val success-rate evaluation "
                        "(randomly sampled from each split; None = all)")
    p.add_argument("--eval_freq",          type=int, default=5000,
                   help="Evaluate every this many pre-training gradient steps")

    p.add_argument("--cuda",               action="store_true", default=True)
    p.add_argument("--no_cuda",            dest="cuda", action="store_false")
    p.add_argument("--verbose",            action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer  = SummaryWriter(str(out_dir / "tb"))

    # ------------------------------------------------------------------
    # Add robomme_benchmark to Python path
    # ------------------------------------------------------------------
    sys.path.insert(0, args.robomme_root)
    sys.path.insert(0, os.path.join(args.robomme_root, "src"))

    # ------------------------------------------------------------------
    # Import env builder (requires robomme on path)
    # ------------------------------------------------------------------
    from robomme.env_record_wrapper.episode_config_resolver import BenchmarkEnvBuilder
    env_builder = BenchmarkEnvBuilder(h5_file=args.h5_file)

    # ------------------------------------------------------------------
    # Build vocabs from h5
    # ------------------------------------------------------------------
    print("Building vocabularies ...")
    vocabs = build_vocabs_from_h5(args.h5_file)
    vocab_sizes = {f: len(vocabs[f]) for f in RICH_CAT_FIELDS}
    print(f"Vocab sizes: {vocab_sizes}")

    # ------------------------------------------------------------------
    # Train/val episode split (same logic as bc_binfill.py)
    # ------------------------------------------------------------------
    with h5py.File(args.h5_file, "r") as fh:
        n_episodes = len(fh.keys())
    rng      = np.random.default_rng(args.seed)
    order    = rng.permutation(n_episodes).tolist()
    n_val    = max(1, int(n_episodes * args.val_fraction))
    val_eps  = sorted(order[:n_val])
    train_eps = sorted(order[n_val:])
    if args.h5_episodes is not None:
        train_eps = train_eps[:args.h5_episodes]
    print(f"Episodes — train: {len(train_eps)}, val: {len(val_eps)}")

    # ------------------------------------------------------------------
    # Build policy
    # ------------------------------------------------------------------
    policy = FlowMatchingPolicyRich(
        action_dim  = ACTION_DIM,
        vocab_sizes = vocab_sizes,
        embed_dim   = args.embed_dim,
        context_dim = args.context_dim,
        hidden_dim  = args.hidden_dim,
        time_emb_dim= args.time_emb_dim,
    ).to(device)
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    optimizer = optim.AdamW(policy.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # Phase 0: Re-simulate H5 demonstrations to get rich state
    # ------------------------------------------------------------------
    print("\n=== Phase 0: Loading H5 demonstrations (re-simulated for rich state) ===")
    dataset = DaggerDataset()
    h5_transitions = load_h5_demos_with_rich_state(
        h5_file         = args.h5_file,
        ep_indices      = train_eps,
        env_builder     = env_builder,
        vocabs          = vocabs,
        max_steps_per_ep= args.max_steps_ep,
        verbose         = args.verbose,
    )
    dataset.add_transitions(h5_transitions)
    print(f"Dataset size after H5 load: {len(dataset)} transitions")

    # Shuffle eval episode lists once (deterministic for reproducibility)
    rng_eval = np.random.default_rng(args.seed + 99)
    train_eps_eval = rng_eval.choice(train_eps, size=min(args.eval_episodes, len(train_eps)), replace=False).tolist()
    val_eps_eval   = rng_eval.choice(val_eps,   size=min(args.eval_episodes, len(val_eps)),   replace=False).tolist()

    def run_eval(tag_prefix: str, global_iter: int) -> None:
        """Evaluate on train and val splits, log to TensorBoard and stdout."""
        print(f"\n--- Evaluation at {tag_prefix} (iter {global_iter}) ---")
        tr = eval_policy(
            env_builder, train_eps_eval, policy, vocabs, device,
            max_steps=args.max_steps_ep, n_ode_steps=args.n_ode_steps,
            max_episodes=args.eval_episodes, h5_file=args.h5_file,
            split_name="train", verbose=args.verbose,
        )
        vl = eval_policy(
            env_builder, val_eps_eval, policy, vocabs, device,
            max_steps=args.max_steps_ep, n_ode_steps=args.n_ode_steps,
            max_episodes=args.eval_episodes, h5_file=args.h5_file,
            split_name="val", verbose=args.verbose,
        )
        writer.add_scalar(f"{tag_prefix}/train_success_rate", tr["success_rate"], global_iter)
        writer.add_scalar(f"{tag_prefix}/val_success_rate",   vl["success_rate"], global_iter)
        policy.train()

    # ------------------------------------------------------------------
    # Pre-train on H5 demonstrations (with periodic eval)
    # ------------------------------------------------------------------
    pretrain_global = 0
    if args.h5_pretrain_iters > 0 and len(dataset) > 0:
        print(f"\nPre-training on H5 demos for {args.h5_pretrain_iters} iters "
              f"(eval every {args.eval_freq} iters) ...")
        remaining = args.h5_pretrain_iters
        chunk_size = args.eval_freq

        while remaining > 0:
            iters_this_chunk = min(chunk_size, remaining)
            avg_loss = train_one_round(
                policy, dataset, optimizer, device,
                n_iters=iters_this_chunk,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            pretrain_global += iters_this_chunk
            remaining       -= iters_this_chunk
            writer.add_scalar("pretrain/loss", avg_loss, pretrain_global)
            print(f"  [pretrain iter {pretrain_global}/{args.h5_pretrain_iters}]  loss={avg_loss:.5f}")
            run_eval("pretrain", pretrain_global)

        torch.save(policy.state_dict(), str(out_dir / "policy_pretrained.pt"))
        print(f"Pre-trained checkpoint saved → {out_dir / 'policy_pretrained.pt'}")

    # ------------------------------------------------------------------
    # DAgger rounds
    # ------------------------------------------------------------------
    global_step = 0

    for dagger_round in range(args.dagger_rounds):
        print(f"\n{'='*60}")
        print(f"DAgger Round {dagger_round + 1}/{args.dagger_rounds}")
        print(f"Dataset size before rollout: {len(dataset)}")

        n_success = 0
        n_expert_steps = 0
        n_policy_steps  = 0

        # Select episodes for this DAgger round (sample from train set)
        round_eps = random.choices(train_eps, k=args.episodes_per_round)

        for ep_num in tqdm(round_eps, desc=f"DAgger round {dagger_round+1}"):
            cam_intr = read_cam_intrinsics_from_h5(args.h5_file, ep_num)
            ep_meta  = read_episode_meta(args.h5_file, ep_num)

            # --- Expert trajectory (from initial state) ---
            try:
                env_exp = env_builder.make_env_for_episode(ep_num, max_steps=args.max_steps_ep)
                expert_transitions = collect_expert_trajectory(
                    env_exp, ep_meta, cam_intr, vocabs,
                    max_steps=args.max_steps_ep,
                    verbose=args.verbose,
                )
                expert_actions = [t["action"] for t in expert_transitions]
                n_expert_steps += len(expert_transitions)
            except Exception as exc:
                if args.verbose:
                    print(f"[DAgger ep {ep_num}] expert failed: {exc}")
                expert_actions = []
            finally:
                try:
                    env_exp.close()
                except Exception:
                    pass

            # --- Policy rollout (from same initial state) ---
            try:
                env_pol = env_builder.make_env_for_episode(ep_num, max_steps=args.max_steps_ep)
                policy_states, outcome = collect_policy_rollout(
                    env_pol, ep_meta, cam_intr, policy, vocabs,
                    device=device,
                    max_steps=args.max_steps_ep,
                    n_ode_steps=args.n_ode_steps,
                )
                n_policy_steps += len(policy_states)
                if outcome == "success":
                    n_success += 1
            except Exception as exc:
                if args.verbose:
                    print(f"[DAgger ep {ep_num}] policy rollout failed: {exc}")
                policy_states = []
            finally:
                try:
                    env_pol.close()
                except Exception:
                    pass

            # --- Aggregate: (policy state, time-aligned expert action) ---
            if policy_states and expert_actions:
                dataset.add_dagger_pairs(policy_states, expert_actions)
            elif expert_transitions:
                # Fallback: add raw expert transitions even without policy states
                dataset.add_transitions(expert_transitions)

        # Data-collection success rate (informational — confounded with exploration)
        collection_sr = n_success / max(args.episodes_per_round, 1)
        print(f"Data-collection success rate: {collection_sr:.2%}  "
              f"({n_success}/{args.episodes_per_round})  "
              f"[NOTE: this is the rate during data collection, not a clean eval]")
        print(f"Expert steps: {n_expert_steps}  Policy steps: {n_policy_steps}")
        print(f"Dataset size after rollout: {len(dataset)}")

        writer.add_scalar("dagger/collection_success_rate", collection_sr, dagger_round)
        writer.add_scalar("dagger/dataset_size", len(dataset), dagger_round)

        # --- Train on aggregated dataset ---
        print(f"Training for {args.iters_per_round} iters ...")
        avg_loss = train_one_round(
            policy, dataset, optimizer, device,
            n_iters=args.iters_per_round,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        global_step += args.iters_per_round
        writer.add_scalar("dagger/train_loss", avg_loss, dagger_round)
        print(f"Round {dagger_round+1} train loss: {avg_loss:.5f}")

        # --- Dedicated success-rate evaluation (clean, separate from data collection) ---
        print(f"\nEvaluating after round {dagger_round+1} ...")
        tr_res = eval_policy(
            env_builder, train_eps_eval, policy, vocabs, device,
            max_steps=args.max_steps_ep, n_ode_steps=args.n_ode_steps,
            max_episodes=args.eval_episodes, h5_file=args.h5_file,
            split_name="train", verbose=args.verbose,
        )
        vl_res = eval_policy(
            env_builder, val_eps_eval, policy, vocabs, device,
            max_steps=args.max_steps_ep, n_ode_steps=args.n_ode_steps,
            max_episodes=args.eval_episodes, h5_file=args.h5_file,
            split_name="val", verbose=args.verbose,
        )
        writer.add_scalar("dagger/train_success_rate", tr_res["success_rate"], dagger_round)
        writer.add_scalar("dagger/val_success_rate",   vl_res["success_rate"], dagger_round)
        print(f"  Train success: {tr_res['n_success']}/{tr_res['n_total']} "
              f"({tr_res['success_rate']:.1%})  |  "
              f"Val success: {vl_res['n_success']}/{vl_res['n_total']} "
              f"({vl_res['success_rate']:.1%})")
        policy.train()

        # Save checkpoint (best by val success rate)
        ckpt_path = out_dir / f"policy_round_{dagger_round+1:03d}.pt"
        torch.save(policy.state_dict(), str(ckpt_path))
        print(f"Checkpoint saved → {ckpt_path}")

    # Final checkpoint
    final_path = out_dir / "policy_final.pt"
    torch.save(policy.state_dict(), str(final_path))
    print(f"\nFinal policy saved → {final_path}")

    # Final evaluation summary
    print("\n=== Final Evaluation ===")
    tr_final = eval_policy(
        env_builder, train_eps_eval, policy, vocabs, device,
        max_steps=args.max_steps_ep, n_ode_steps=args.n_ode_steps,
        max_episodes=args.eval_episodes, h5_file=args.h5_file,
        split_name="train (final)", verbose=args.verbose,
    )
    vl_final = eval_policy(
        env_builder, val_eps_eval, policy, vocabs, device,
        max_steps=args.max_steps_ep, n_ode_steps=args.n_ode_steps,
        max_episodes=args.eval_episodes, h5_file=args.h5_file,
        split_name="val (final)", verbose=args.verbose,
    )
    print(f"\nFinal train success rate: {tr_final['success_rate']:.1%}  "
          f"({tr_final['n_success']}/{tr_final['n_total']})")
    print(f"Final val   success rate: {vl_final['success_rate']:.1%}  "
          f"({vl_final['n_success']}/{vl_final['n_total']})")

    # Save vocabs for inference
    vocab_path = out_dir / "vocabs.pt"
    torch.save({f: vocabs[f].state_dict() for f in RICH_CAT_FIELDS}, str(vocab_path))
    print(f"Vocabs saved → {vocab_path}")

    writer.close()


if __name__ == "__main__":
    main()
