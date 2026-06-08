"""
Subtask-conditioned offline BC with Flow Matching on RoboMME BinFill H5 data.

Architecture choice
-------------------
MSE direct regression: averages multimodal distributions — within "pick up the red
  cube at [105, 118]" a robot can legally approach from multiple angles, so MSE
  predicts an invalid average trajectory.

Diffusion Policy: handles multimodality but requires ~100 denoising steps per
  inference call, which is too slow for 10-20 Hz closed-loop control.

Flow Matching (chosen): OT-straight paths need only ~10 Euler steps at inference
  (10× faster than diffusion) with the same expressiveness. Already validated in
  this repo by bc_binfill_fm.py. Works especially well here because the subgoal
  conditioning narrows each subtask to a near-unimodal sub-distribution (one target
  object, one target pixel), making flow paths short and stable.

Data structure
--------------
Full episodes are split at is_subgoal_boundary transitions into subtask segments.
Each segment carries a FIXED parsed subgoal and a sequence of (state, action) pairs.
Train / val split is at EPISODE level to prevent leakage between segments of the
same episode.

State  : joint_state(7) + eef_state(6) + gripper_state(2) + is_gripper_close(1) = 16-D
Goal   : parsed from grounded_subgoal_online text
           action_type ∈ {pick, put, press}       → nn.Embedding(3, embed_dim)
           color       ∈ {red, green, blue, none}  → nn.Embedding(4, embed_dim)
           pixel_y, pixel_x  ∈ [0, 1]             → raw float (2-D)
Action : joint_action (8-D): 7 absolute joint angles + gripper (-1=close, +1=open)

Training (Conditional Flow Matching — OT linear paths)
  x_1 = normalised ground-truth joint_action
  x_0 ~ N(0, I)
  t   ~ Uniform(0, 1)
  x_t = (1 - t) * x_0 + t * x_1
  u_t = x_1 - x_0          (constant velocity along OT path)
  loss = MSE( v_θ(x_t, t, context), u_t )

Inference (Euler ODE, n_steps steps)
  x ~ N(0, I)
  for k in 0 .. n_steps-1:
      x += v_θ(x, k / n_steps, context) / n_steps
  return de_normalise(x)

Offline evaluation during training:
  - FM loss on the val set (fast, no ODE needed)
  - Action accuracy: fraction of ODE-sampled predictions within L2 < threshold
  - Per action-type breakdown (pick / put / press)

Simulation evaluation is handled by eval_subtask_sim.py (separate script).
"""

import math
import os
import re
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

STATE_DIM  = 16   # joint(7) + eef(6) + gripper(2) + is_gripper_close(1)
ACTION_DIM = 8    # joint_action: 7 joint angles + gripper command

ACTION_TYPE_VOCAB = ["pick", "put", "press"]         # indices 0, 1, 2
COLOR_VOCAB       = ["red", "green", "blue", "none"]  # indices 0, 1, 2, 3
IMG_SIZE          = 256.0   # image resolution used for normalising pixel coords


# ─────────────────────────────────────────────────────────────────────────────
# Subgoal text parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_grounded_subgoal(text: str) -> Dict:
    """
    Parse a grounded_subgoal_online string into structured fields.

    Examples
    --------
    "pick up the 1st red cube at [105, 118]"  → pick,  red,   py=105/256, px=118/256
    "put it into the bin"                      → put,   none,  py=0,       px=0
    "press the button"                         → press, none,  py=0,       px=0

    Pixel convention in RoboMME: [y, x].
    Returns
    -------
    dict with keys: action_type (int), color (int), pixel_y (float),
                    pixel_x (float), raw_text (str)
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    t = text.lower().strip()

    if "pick" in t:
        action_type = 0
    elif any(kw in t for kw in ("put", "place", "bin")):
        action_type = 1
    elif any(kw in t for kw in ("press", "button")):
        action_type = 2
    else:
        action_type = 0

    color = 3  # none
    for i, c in enumerate(("red", "green", "blue")):
        if c in t:
            color = i
            break

    m = re.search(r"\[(\d+),\s*(\d+)\]", t)
    if m:
        pixel_y = float(m.group(1)) / IMG_SIZE
        pixel_x = float(m.group(2)) / IMG_SIZE
    else:
        pixel_y = pixel_x = 0.0

    return {
        "action_type": action_type,
        "color":       color,
        "pixel_y":     pixel_y,
        "pixel_x":     pixel_x,
        "raw_text":    text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# H5 loading + subtask segmentation
# ─────────────────────────────────────────────────────────────────────────────

def _h5_scalar(val) -> str:
    """Decode h5 bytes / numpy scalar to a plain Python str."""
    if isinstance(val, bytes):
        return val.decode("utf-8")
    if isinstance(val, np.ndarray):
        item = val.item()
        return item.decode("utf-8") if isinstance(item, bytes) else str(item)
    return str(val)


def load_subtasks_from_h5(
    h5_path: str,
    num_episodes: Optional[int] = None,
) -> Tuple[List[List[Dict]], List[Dict]]:
    """
    Read the RoboMME H5 file and segment every episode into subtask segments.

    H5 structure assumed:
      record_dataset_BinFill.h5
      ├── episode_1/
      │   ├── setup/   (seed, difficulty, task_goal, camera intrinsics)
      │   └── timestep_1/ ... timestep_T/
      │         ├── obs/    (joint_state, eef_state, gripper_state, is_gripper_close, …)
      │         ├── action/ (joint_action (8,), eef_action, waypoint_action, …)
      │         └── info/   (grounded_subgoal_online, is_subgoal_boundary, is_completed, …)
      └── episode_2/ …

    Segmentation rule
    -----------------
    is_subgoal_boundary=True marks the first timestep of a new subtask.
    We always force a boundary at index 0 (start of episode) so the very first
    subtask is always captured.

    Parameters
    ----------
    h5_path      : path to the H5 file
    num_episodes : cap on episodes to load; None = all

    Returns
    -------
    episodes_subtasks : List[List[dict]]
        Outer list = one entry per episode.
        Inner list = ordered subtask segments for that episode.
        Each subtask dict:
          subgoal_text   : str   raw grounded_subgoal_online at segment start
          subgoal_parsed : dict  {action_type, color, pixel_y, pixel_x, raw_text}
          states         : ndarray (T_sub, 16) float32
          actions        : ndarray (T_sub, 8)  float32
          episode_key    : str
          difficulty     : str
    episode_setups : List[dict]
        One dict per episode: episode_key, seed, difficulty, task_goal.
    """
    episodes_subtasks: List[List[Dict]] = []
    episode_setups:    List[Dict]       = []

    with h5py.File(h5_path, "r") as f:
        ep_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))
        if num_episodes is not None:
            ep_keys = ep_keys[:num_episodes]

        print(f"Loading {len(ep_keys)} episodes from {h5_path} ...")
        for ep_key in tqdm(ep_keys):
            ep    = f[ep_key]
            setup = ep["setup"]

            difficulty = _h5_scalar(setup["difficulty"][()])
            seed       = int(setup["seed"][()])
            tg_raw     = setup["task_goal"][()]
            task_goal  = (
                [_h5_scalar(g) for g in tg_raw]
                if hasattr(tg_raw, "__iter__") and not isinstance(tg_raw, (str, bytes))
                else [_h5_scalar(tg_raw)]
            )

            episode_setups.append({
                "episode_key": ep_key,
                "seed":        seed,
                "difficulty":  difficulty,
                "task_goal":   task_goal,
            })

            ts_keys = sorted(
                [k for k in ep.keys() if k.startswith("timestep_")],
                key=lambda k: int(k.split("_")[-1]),
            )
            if not ts_keys:
                episodes_subtasks.append([])
                continue

            states_all      = []
            actions_all     = []
            sg_texts_all    = []
            is_boundary_all = []

            for ts_key in ts_keys:
                ts  = ep[ts_key]
                obs = ts["obs"]
                act = ts["action"]
                inf = ts["info"]

                joint_state      = obs["joint_state"][()].astype(np.float32)          # (7,)
                eef_state        = obs["eef_state"][()].astype(np.float32)            # (6,)
                gripper_state    = obs["gripper_state"][()].astype(np.float32)        # (2,)
                is_gripper_close = np.array(
                    [float(bool(obs["is_gripper_close"][()]))], dtype=np.float32
                )                                                                      # (1,)
                state = np.concatenate(
                    [joint_state, eef_state, gripper_state, is_gripper_close]
                )                                                                      # (16,)

                joint_action = act["joint_action"][()].astype(np.float32)             # (8,)
                sg_text      = _h5_scalar(inf["grounded_subgoal_online"][()])
                is_boundary  = bool(inf["is_subgoal_boundary"][()])

                states_all.append(state)
                actions_all.append(joint_action)
                sg_texts_all.append(sg_text)
                is_boundary_all.append(is_boundary)

            # ── segment on boundary transitions ──────────────────────────
            T = len(states_all)
            boundary_indices = [i for i, b in enumerate(is_boundary_all) if b]
            if not boundary_indices or boundary_indices[0] != 0:
                boundary_indices = [0] + boundary_indices

            ep_subtasks: List[Dict] = []
            for seg_idx, start in enumerate(boundary_indices):
                end = boundary_indices[seg_idx + 1] if seg_idx + 1 < len(boundary_indices) else T
                if end <= start:
                    continue
                sg_text = sg_texts_all[start]
                ep_subtasks.append({
                    "subgoal_text":   sg_text,
                    "subgoal_parsed": parse_grounded_subgoal(sg_text),
                    "states":         np.array(states_all[start:end],  dtype=np.float32),
                    "actions":        np.array(actions_all[start:end], dtype=np.float32),
                    "episode_key":    ep_key,
                    "difficulty":     difficulty,
                })

            episodes_subtasks.append(ep_subtasks)

    return episodes_subtasks, episode_setups


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SubtaskDataset(Dataset):
    """
    Flat dataset of (state, action_type, color, pixel, action) per timestep.

    States and actions are normalised using per-dim mean/std computed from the
    training set. Pass fit_normalizer=True for training; pass the resulting
    {state,action}_{mean,std} arrays to the validation set constructor.

    The same subgoal fields repeat for every timestep within a segment.
    """

    def __init__(
        self,
        subtask_list:   List[Dict],
        state_mean:     Optional[np.ndarray] = None,
        state_std:      Optional[np.ndarray] = None,
        action_mean:    Optional[np.ndarray] = None,
        action_std:     Optional[np.ndarray] = None,
        fit_normalizer: bool = False,
    ):
        all_states  = np.concatenate([s["states"]  for s in subtask_list], axis=0)
        all_actions = np.concatenate([s["actions"] for s in subtask_list], axis=0)

        if fit_normalizer:
            state_mean  = all_states.mean(0)
            state_std   = all_states.std(0).clip(1e-6)
            action_mean = all_actions.mean(0)
            action_std  = all_actions.std(0).clip(1e-6)

        self.state_mean  = state_mean
        self.state_std   = state_std
        self.action_mean = action_mean
        self.action_std  = action_std

        if state_mean is not None:
            all_states = (all_states - state_mean) / state_std
        if action_mean is not None:
            all_actions = (all_actions - action_mean) / action_std

        # Broadcast per-segment subgoal to every timestep in the segment
        action_types: List[int]            = []
        colors:       List[int]            = []
        pixels:       List[List[float]]    = []
        for st in subtask_list:
            T  = len(st["states"])
            sg = st["subgoal_parsed"]
            action_types.extend([sg["action_type"]] * T)
            colors.extend([sg["color"]] * T)
            pixels.extend([[sg["pixel_y"], sg["pixel_x"]]] * T)

        self.states       = torch.from_numpy(all_states).float()
        self.actions      = torch.from_numpy(all_actions).float()
        self.action_types = torch.tensor(action_types, dtype=torch.long)
        self.colors       = torch.tensor(colors,       dtype=torch.long)
        self.pixels       = torch.tensor(pixels,       dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.action_types[idx],
            self.colors[idx],
            self.pixels[idx],
            self.actions[idx],
        )


def collate_fn(batch):
    states, atypes, colors, pixels, actions = zip(*batch)
    return (
        torch.stack(states),
        torch.stack(atypes),
        torch.stack(colors),
        torch.stack(pixels),
        torch.stack(actions),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal time embedding
# ─────────────────────────────────────────────────────────────────────────────

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t : (B,) float in [0, 1]   →   (B, dim)"""
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Network modules
# ─────────────────────────────────────────────────────────────────────────────

class SubtaskEncoder(nn.Module):
    """
    Encodes (robot_state_16D, subgoal) into a fixed-size context vector.

    Subgoal is represented as:
      - action_type ∈ {0,1,2}  → learned embedding
      - color       ∈ {0,1,2,3}→ learned embedding
      - pixel_y, pixel_x ∈ [0,1] → raw continuous
    """

    def __init__(self, embed_dim: int, context_dim: int, hidden_dim: int):
        super().__init__()
        self.action_type_emb = nn.Embedding(len(ACTION_TYPE_VOCAB), embed_dim)
        self.color_emb       = nn.Embedding(len(COLOR_VOCAB),       embed_dim)
        in_dim = STATE_DIM + 2 * embed_dim + 2   # state + 2 embeddings + pixel(2)
        self.net = nn.Sequential(
            nn.Linear(in_dim,    hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(
        self,
        state:       torch.Tensor,   # (B, 16)
        action_type: torch.Tensor,   # (B,)  long
        color:       torch.Tensor,   # (B,)  long
        pixel:       torch.Tensor,   # (B, 2)
    ) -> torch.Tensor:               # (B, context_dim)
        parts = [
            state,
            self.action_type_emb(action_type),
            self.color_emb(color),
            pixel,
        ]
        return self.net(torch.cat(parts, dim=-1))


class VectorFieldNet(nn.Module):
    """v_θ(x_t, t, context) → vector field  (B, action_dim)"""

    def __init__(self, action_dim: int, context_dim: int, time_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_proj    = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU()
        )
        in_dim = action_dim + time_emb_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim,    hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        x_t:     torch.Tensor,   # (B, action_dim)
        t:       torch.Tensor,   # (B,)  float in [0, 1]
        context: torch.Tensor,   # (B, context_dim)
    ) -> torch.Tensor:
        t_emb = self.time_proj(sinusoidal_embedding(t, self.time_emb_dim))
        return self.net(torch.cat([x_t, t_emb, context], dim=-1))


class SubtaskFlowMatchingPolicy(nn.Module):
    """
    Full subtask-conditioned policy: SubtaskEncoder + VectorFieldNet.

    Action normalisation (mean / std) is stored as buffers so they are saved
    with the checkpoint and automatically restored without needing separate files.
    """

    def __init__(
        self,
        embed_dim:    int = 16,
        context_dim:  int = 256,
        hidden_dim:   int = 256,
        time_emb_dim: int = 64,
    ):
        super().__init__()
        self.action_dim = ACTION_DIM
        self.encoder    = SubtaskEncoder(embed_dim, context_dim, hidden_dim)
        self.vf_net     = VectorFieldNet(ACTION_DIM, context_dim, time_emb_dim, hidden_dim)

        # Action normalisation parameters — set via set_normalizers() after data load
        self.register_buffer("state_mean",  torch.zeros(STATE_DIM))
        self.register_buffer("state_std",   torch.ones(STATE_DIM))
        self.register_buffer("action_mean", torch.zeros(ACTION_DIM))
        self.register_buffer("action_std",  torch.ones(ACTION_DIM))

    def set_normalizers(
        self,
        state_mean:  np.ndarray,
        state_std:   np.ndarray,
        action_mean: np.ndarray,
        action_std:  np.ndarray,
    ) -> None:
        self.state_mean.copy_(torch.from_numpy(state_mean.astype(np.float32)))
        self.state_std.copy_(torch.from_numpy(state_std.astype(np.float32)))
        self.action_mean.copy_(torch.from_numpy(action_mean.astype(np.float32)))
        self.action_std.copy_(torch.from_numpy(action_std.astype(np.float32)))

    def _encode(self, state, action_type, color, pixel):
        return self.encoder(state, action_type, color, pixel)

    def compute_loss(
        self,
        state:       torch.Tensor,   # (B, 16)  normalised
        action_type: torch.Tensor,   # (B,)
        color:       torch.Tensor,   # (B,)
        pixel:       torch.Tensor,   # (B, 2)
        x_1:         torch.Tensor,   # (B, 8)   normalised ground-truth action
    ) -> torch.Tensor:
        """Conditional flow matching loss (OT-linear paths)."""
        B       = x_1.size(0)
        context = self._encode(state, action_type, color, pixel)

        x_0 = torch.randn_like(x_1)
        t   = torch.rand(B, device=x_1.device)

        x_t = (1.0 - t[:, None]) * x_0 + t[:, None] * x_1
        u_t = x_1 - x_0

        v = self.vf_net(x_t, t, context)
        return F.mse_loss(v, u_t)

    @torch.no_grad()
    def sample(
        self,
        state:       torch.Tensor,   # (B, 16) — normalised if from dataset, raw if from env
        action_type: torch.Tensor,   # (B,)
        color:       torch.Tensor,   # (B,)
        pixel:       torch.Tensor,   # (B, 2)
        n_steps:     int  = 10,
        state_is_raw: bool = False,  # set True when feeding raw env obs (will normalise here)
    ) -> torch.Tensor:
        """Euler ODE integration → de-normalised joint_action (B, 8)."""
        if state_is_raw:
            state = (state - self.state_mean) / self.state_std

        context = self._encode(state, action_type, color, pixel)
        x = torch.randn(state.size(0), self.action_dim, device=state.device)
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t = torch.full((state.size(0),), k * dt, device=state.device)
            x = x + self.vf_net(x, t, context) * dt

        return x * self.action_std + self.action_mean   # de-normalise


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    exp_name:            Optional[str] = None
    seed:                int           = 1
    torch_deterministic: bool          = True
    cuda:                bool          = True
    track:               bool          = False
    wandb_project_name:  str           = "ManiSkill"
    wandb_entity:        Optional[str] = None
    save_model:          bool          = True

    # Dataset
    h5_file:      str           = "robomme_data/record_dataset_BinFill.h5"
    """path to the BinFill H5 file"""
    num_episodes: Optional[int] = None
    """cap on episodes to load; None = all"""
    val_fraction: float         = 0.2
    """fraction of EPISODES held out for validation (split before flattening to segments)"""

    # Model architecture
    embed_dim:    int = 16
    """embedding dim for action_type and color"""
    context_dim:  int = 256
    """SubtaskEncoder output / conditioning vector size"""
    hidden_dim:   int = 256
    """MLP hidden layer width"""
    time_emb_dim: int = 64
    """sinusoidal time embedding dim"""

    # Flow-matching inference
    n_inference_steps: int = 10
    """Euler ODE steps at evaluation (more → better quality, slower)"""

    # Training
    total_iters:          int   = 100_000
    batch_size:           int   = 512
    lr:                   float = 3e-4
    num_dataload_workers: int   = 0

    # Logging / offline eval
    log_freq:      int   = 200
    eval_freq:     int   = 1_000
    acc_threshold: float = 0.05
    """L2 threshold (in raw action space) for counting a prediction as accurate"""


# ─────────────────────────────────────────────────────────────────────────────
# Offline evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_offline(
    policy:    SubtaskFlowMatchingPolicy,
    loader:    DataLoader,
    device:    torch.device,
    n_steps:   int,
    threshold: float,
) -> Dict[str, float]:
    """
    FM loss + action-quality metrics on a validation DataLoader.

    All L2 comparisons are in the de-normalised (raw) action space so that
    the threshold is interpretable in physical units (rad / gripper units).
    """
    policy.eval()
    fm_sum = acc_sum = mae_sum = n = 0

    type_acc:   List[float] = [0.0] * len(ACTION_TYPE_VOCAB)
    type_count: List[int]   = [0]   * len(ACTION_TYPE_VOCAB)

    for state, atype, color, pixel, action_norm in loader:
        state       = state.to(device)
        atype       = atype.to(device)
        color       = color.to(device)
        pixel       = pixel.to(device)
        action_norm = action_norm.to(device)

        fm_sum += policy.compute_loss(state, atype, color, pixel, action_norm).item() * len(state)

        pred   = policy.sample(state, atype, color, pixel, n_steps=n_steps)  # de-normalised
        gt_raw = action_norm * policy.action_std + policy.action_mean

        l2 = (pred - gt_raw).norm(dim=-1)
        acc_sum += l2.lt(threshold).float().sum().item()
        mae_sum += l2.sum().item()
        n       += len(state)

        for ti in range(len(ACTION_TYPE_VOCAB)):
            mask = (atype == ti)
            if mask.any():
                type_acc[ti]   += l2[mask].lt(threshold).float().sum().item()
                type_count[ti] += int(mask.sum().item())

    per_type = {
        f"eval_acc_{ACTION_TYPE_VOCAB[ti]}": type_acc[ti] / type_count[ti]
        for ti in range(len(ACTION_TYPE_VOCAB))
        if type_count[ti] > 0
    }

    policy.train()
    return {
        "eval_fm_loss": fm_sum  / max(n, 1),
        "eval_acc":     acc_sum / max(n, 1),
        "eval_mae":     mae_sum / max(n, 1),
        **per_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
    run_name = f"BinFill_subtask__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────
    episodes_subtasks, episode_setups = load_subtasks_from_h5(
        args.h5_file, num_episodes=args.num_episodes
    )
    n_ep      = len(episodes_subtasks)
    all_flat  = [st for ep in episodes_subtasks for st in ep]
    print(f"\nEpisodes loaded     : {n_ep}")
    print(f"Total subtasks      : {len(all_flat)}")
    print(f"Total timesteps     : {sum(len(s['states']) for s in all_flat):,}")
    print(f"Avg subtask length  : {np.mean([len(s['states']) for s in all_flat]):.1f} steps")

    type_dist  = Counter(ACTION_TYPE_VOCAB[s["subgoal_parsed"]["action_type"]] for s in all_flat)
    color_dist = Counter(COLOR_VOCAB[s["subgoal_parsed"]["color"]]             for s in all_flat)
    print(f"Action type dist    : {dict(type_dist)}")
    print(f"Color dist          : {dict(color_dist)}")

    # ── Episode-level train / val split (no leakage) ───────────────────────
    rng   = np.random.default_rng(args.seed)
    order = rng.permutation(n_ep).tolist()
    n_val = max(1, int(n_ep * args.val_fraction))

    val_episodes   = [episodes_subtasks[i] for i in order[:n_val]]
    train_episodes = [episodes_subtasks[i] for i in order[n_val:]]
    val_subtasks   = [st for ep in val_episodes   for st in ep]
    train_subtasks = [st for ep in train_episodes for st in ep]

    print(f"\nTrain — {len(train_episodes)} episodes, {len(train_subtasks)} subtasks, "
          f"{sum(len(s['states']) for s in train_subtasks):,} timesteps")
    print(f"Val   — {len(val_episodes)} episodes, {len(val_subtasks)} subtasks, "
          f"{sum(len(s['states']) for s in val_subtasks):,} timesteps")

    # ── Normaliser (fit on train, apply to val) ───────────────────────────
    train_ds = SubtaskDataset(train_subtasks, fit_normalizer=True)
    val_ds   = SubtaskDataset(
        val_subtasks,
        state_mean  = train_ds.state_mean,
        state_std   = train_ds.state_std,
        action_mean = train_ds.action_mean,
        action_std  = train_ds.action_std,
    )

    print(f"\nState  mean (first 4): {train_ds.state_mean[:4]}")
    print(f"State  std  (first 4): {train_ds.state_std[:4]}")
    print(f"Action mean          : {train_ds.action_mean}")
    print(f"Action std           : {train_ds.action_std}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_dataload_workers, drop_last=True,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 4, shuffle=False,
        num_workers=args.num_dataload_workers,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"),
    )

    # ── Model ──────────────────────────────────────────────────────────────
    policy = SubtaskFlowMatchingPolicy(
        embed_dim    = args.embed_dim,
        context_dim  = args.context_dim,
        hidden_dim   = args.hidden_dim,
        time_emb_dim = args.time_emb_dim,
    ).to(device)
    policy.set_normalizers(
        train_ds.state_mean, train_ds.state_std,
        train_ds.action_mean, train_ds.action_std,
    )

    enc_p  = sum(p.numel() for p in policy.encoder.parameters())
    vfn_p  = sum(p.numel() for p in policy.vf_net.parameters())
    tot_p  = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"\nModel params: {tot_p:,}  (encoder={enc_p:,}, vf_net={vfn_p:,})")
    print(f"ODE inference steps: {args.n_inference_steps}")

    optimizer = optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-6)

    # ── Save checkpoint dir + normaliser stats ─────────────────────────────
    ckpt_dir = f"runs/{run_name}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    # Normaliser stats also stored in the checkpoint buffers, but save .npy
    # for convenient loading in eval_subtask_sim.py without needing the model
    np.save(f"{ckpt_dir}/state_mean.npy",  train_ds.state_mean)
    np.save(f"{ckpt_dir}/state_std.npy",   train_ds.state_std)
    np.save(f"{ckpt_dir}/action_mean.npy", train_ds.action_mean)
    np.save(f"{ckpt_dir}/action_std.npy",  train_ds.action_std)

    # ── Logging ────────────────────────────────────────────────────────────
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name, entity=args.wandb_entity,
            sync_tensorboard=True, config=vars(args),
            name=run_name, save_code=True, group="SubtaskBC-BinFill",
        )

    # ── Training loop ──────────────────────────────────────────────────────
    best_eval_fm = float("inf")
    iteration    = 0
    start_time   = time.time()

    hdr = (f"{'Iter':>7}  {'FM_loss':>8}  {'Tr_acc':>7}"
           f"  {'Eval_FM':>9}  {'Eval_acc':>8}"
           f"  {'pick':>6}  {'put':>6}  {'press':>6}  {'s':>5}")
    print(f"\nTraining for {args.total_iters} iters (batch={args.batch_size})")
    print(hdr)
    print("-" * len(hdr))

    policy.train()
    while iteration < args.total_iters:
        for state, atype, color, pixel, action in train_loader:
            if iteration >= args.total_iters:
                break

            state  = state.to(device)
            atype  = atype.to(device)
            color  = color.to(device)
            pixel  = pixel.to(device)
            action = action.to(device)

            loss = policy.compute_loss(state, atype, color, pixel, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % args.log_freq == 0:
                writer.add_scalar("train/fm_loss", loss.item(), iteration)

            if iteration % args.eval_freq == 0:
                eval_m = evaluate_offline(
                    policy, val_loader, device,
                    args.n_inference_steps, args.acc_threshold,
                )
                with torch.no_grad():
                    pred_tr = policy.sample(state, atype, color, pixel,
                                            n_steps=args.n_inference_steps)
                    gt_tr   = action * policy.action_std + policy.action_mean
                    tr_acc  = ((pred_tr - gt_tr).norm(dim=-1) < args.acc_threshold).float().mean().item()

                elapsed = time.time() - start_time
                print(
                    f"{iteration:>7d}"
                    f"  {loss.item():>8.5f}"
                    f"  {tr_acc:>7.4f}"
                    f"  {eval_m['eval_fm_loss']:>9.5f}"
                    f"  {eval_m['eval_acc']:>8.4f}"
                    f"  {eval_m.get('eval_acc_pick',  float('nan')):>6.3f}"
                    f"  {eval_m.get('eval_acc_put',   float('nan')):>6.3f}"
                    f"  {eval_m.get('eval_acc_press', float('nan')):>6.3f}"
                    f"  {elapsed:>4.0f}s"
                )
                for k, v in eval_m.items():
                    writer.add_scalar(f"eval/{k}", v, iteration)
                writer.add_scalar("train/accuracy", tr_acc, iteration)

                if eval_m["eval_fm_loss"] < best_eval_fm and args.save_model:
                    best_eval_fm = eval_m["eval_fm_loss"]
                    torch.save(
                        {"policy_state_dict": policy.state_dict(), "args": vars(args)},
                        f"{ckpt_dir}/best_eval_fm_loss.pt",
                    )
                    print(f"  → best eval_fm_loss={best_eval_fm:.5f}. Saved.")

            if args.save_model and iteration > 0 and iteration % 10_000 == 0:
                torch.save(
                    {"policy_state_dict": policy.state_dict(), "args": vars(args)},
                    f"{ckpt_dir}/iter_{iteration:07d}.pt",
                )

            iteration += 1

    # ── Final eval + checkpoint ────────────────────────────────────────────
    print("\n--- Final offline evaluation ---")
    final_m = evaluate_offline(
        policy, val_loader, device, args.n_inference_steps, args.acc_threshold
    )
    for k, v in final_m.items():
        print(f"  {k}: {v:.6f}")
        writer.add_scalar(f"eval/{k}", v, iteration)

    if args.save_model:
        torch.save(
            {"policy_state_dict": policy.state_dict(), "args": vars(args)},
            f"{ckpt_dir}/final.pt",
        )
        print(f"Final checkpoint saved to {ckpt_dir}/final.pt")

    writer.close()
    if args.track:
        import wandb
        wandb.finish()
