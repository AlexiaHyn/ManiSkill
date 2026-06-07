"""
Flow Matching BC on the RoboMME BinFill h5 dataset.

Data loading is shared with bc_binfill.py (same obs, same categorical encoding).

Architecture
------------
  ObsEncoder      : (cont_obs 37-D, cat embeddings) → context vector (context_dim)
  VectorFieldNet  : (noisy_action, sinusoidal_time_emb, context) → vector field

Training  — Conditional Flow Matching (CFM) with optimal-transport paths
  x_1  = ground-truth action
  x_0  ~ N(0, I)
  t    ~ Uniform(0, 1)
  x_t  = (1 - t) * x_0 + t * x_1          # linear interpolation
  u_t  = x_1 - x_0                         # constant target vector field
  loss = MSE( v_θ(x_t, t, context), u_t )

Inference — Euler ODE integration
  x_0  ~ N(0, I)
  for each step k  in  0 … n_steps-1:
      t   = k / n_steps
      x  += v_θ(x, t, context) * (1 / n_steps)
  return x   ← predicted action

Metrics printed every --eval_freq iterations:
  train_fm_loss  : CFM loss on current mini-batch
  eval_fm_loss   : CFM loss on the full validation set (fast, no ODE)
  eval_acc       : % of ODE-sampled actions within L2 threshold of ground truth
  eval_mae       : mean absolute error of ODE-sampled actions vs ground truth
"""

import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Reuse data layer from bc_binfill.py (same directory)
sys.path.insert(0, os.path.dirname(__file__))
from bc_binfill import (
    ACTION_DIM, CAT_FIELDS, CONT_OBS_DIM,
    BinFillDataset, Vocab, collate_fn, load_all_episodes,
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    save_model: bool = True

    # Dataset
    h5_file: str = "record_dataset_BinFill.h5"
    action_key: str = "eef_action"
    """eef_action (7-D), joint_action (8-D), waypoint_action (7-D)"""
    val_fraction: float = 0.2
    num_demos: Optional[int] = None
    normalize_states: bool = False

    # Model
    embed_dim: int = 16
    """embedding dimension for each categorical field"""
    context_dim: int = 256
    """ObsEncoder output / conditioning vector size"""
    hidden_dim: int = 256
    """width of both ObsEncoder and VectorFieldNet MLPs"""
    time_emb_dim: int = 64
    """sinusoidal time embedding dimension"""

    # Flow matching inference
    n_inference_steps: int = 10
    """Euler ODE steps at inference (more = better quality, slower)"""

    # Training
    total_iters: int = 100_000
    batch_size: int = 512
    lr: float = 3e-4
    num_dataload_workers: int = 0

    # Logging / eval
    log_freq: int = 100
    eval_freq: int = 500
    acc_threshold: float = 0.05
    """L2 threshold for counting a sampled action as accurate"""


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t   : (B,) float in [0, 1]
    out : (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None]          # (B, half)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# ObsEncoder
# ---------------------------------------------------------------------------

class ObsEncoder(nn.Module):
    """
    Encodes (continuous obs, categorical ids) into a fixed-size context vector
    that conditions the vector field network.
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
            field: nn.Embedding(vocab_sizes[field], embed_dim)
            for field in CAT_FIELDS
        })
        input_dim = CONT_OBS_DIM + len(CAT_FIELDS) * embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(self, cont: torch.Tensor, cat: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [cont] + [self.embeddings[f](cat[f]) for f in CAT_FIELDS]
        return self.net(torch.cat(parts, dim=-1))   # (B, context_dim)


# ---------------------------------------------------------------------------
# Vector field network
# ---------------------------------------------------------------------------

class VectorFieldNet(nn.Module):
    """
    Predicts the flow matching vector field  v_θ(x_t, t, context).

    input  : x_t (action_dim) || time_emb (time_emb_dim) || context (context_dim)
    output : v   (action_dim)
    """

    def __init__(
        self,
        action_dim: int,
        context_dim: int,
        time_emb_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        # project time sinusoid to a learned embedding
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(),
        )
        input_dim = action_dim + time_emb_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,       # (B, action_dim)
        t: torch.Tensor,          # (B,)  float in [0, 1]
        context: torch.Tensor,    # (B, context_dim)
    ) -> torch.Tensor:
        t_emb = self.time_proj(sinusoidal_embedding(t, self.time_emb_dim))
        inp   = torch.cat([x_t, t_emb, context], dim=-1)
        return self.net(inp)                          # (B, action_dim)


# ---------------------------------------------------------------------------
# Full policy
# ---------------------------------------------------------------------------

class FlowMatchingPolicy(nn.Module):
    def __init__(
        self,
        action_dim: int,
        vocab_sizes: Dict[str, int],
        embed_dim: int,
        context_dim: int,
        hidden_dim: int,
        time_emb_dim: int,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.encoder    = ObsEncoder(vocab_sizes, embed_dim, context_dim, hidden_dim)
        self.vf_net     = VectorFieldNet(action_dim, context_dim, time_emb_dim, hidden_dim)

    # ------------------------------------------------------------------
    # Training: CFM loss
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        cont: torch.Tensor,           # (B, 37)
        cat: Dict[str, torch.Tensor], # each (B,)
        x_1: torch.Tensor,            # (B, action_dim)  ground-truth action
    ) -> torch.Tensor:
        B = x_1.size(0)
        context = self.encoder(cont, cat)                     # (B, context_dim)

        x_0 = torch.randn_like(x_1)                          # (B, action_dim)
        t   = torch.rand(B, device=x_1.device)               # (B,)

        # linear interpolation
        t_bcast = t[:, None]
        x_t = (1.0 - t_bcast) * x_0 + t_bcast * x_1         # (B, action_dim)

        # constant target vector field along the OT path
        u_t = x_1 - x_0                                       # (B, action_dim)

        v_pred = self.vf_net(x_t, t, context)                # (B, action_dim)
        return F.mse_loss(v_pred, u_t)

    # ------------------------------------------------------------------
    # Inference: Euler ODE integration
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        cont: torch.Tensor,
        cat: Dict[str, torch.Tensor],
        n_steps: int = 10,
    ) -> torch.Tensor:
        """Returns predicted action (B, action_dim) via Euler integration."""
        context = self.encoder(cont, cat)                     # (B, context_dim)
        x = torch.randn(cont.size(0), self.action_dim, device=cont.device)
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t = torch.full((cont.size(0),), k * dt, device=cont.device)
            v = self.vf_net(x, t, context)
            x = x + v * dt
        return x


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def action_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> float:
    return ((pred - target).norm(dim=-1) < threshold).float().mean().item()


@torch.no_grad()
def evaluate_on_dataset(
    policy: FlowMatchingPolicy,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    n_steps: int,
) -> Dict[str, float]:
    policy.eval()
    fm_loss = acc = mae = n = 0
    for cont, cat, action in loader:
        cont   = cont.to(device)
        cat    = {f: v.to(device) for f, v in cat.items()}
        action = action.to(device)

        # CFM loss (cheap — no ODE needed)
        fm_loss += policy.compute_loss(cont, cat, action).item() * len(cont)

        # ODE sample for action quality metrics
        pred = policy.sample(cont, cat, n_steps=n_steps)
        acc += (pred - action).norm(dim=-1).lt(threshold).float().sum().item()
        mae += (pred - action).abs().mean(dim=-1).sum().item()
        n   += len(cont)

    policy.train()
    return {
        "eval_fm_loss": fm_loss / n,
        "eval_acc":     acc     / n,
        "eval_mae":     mae     / n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
    run_name = f"BinFill__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")

    action_dim = ACTION_DIM[args.action_key]

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    vocabs: Dict[str, Vocab] = {field: Vocab() for field in CAT_FIELDS}
    all_records = load_all_episodes(args.h5_file, args.action_key, args.num_demos, vocabs)

    print("\nVocabulary sizes:")
    for field, vocab in vocabs.items():
        print(f"  {field}: {len(vocab)} tokens")

    # ------------------------------------------------------------------
    # Print one example to verify input construction
    # ------------------------------------------------------------------
    ex   = all_records[0]
    cont = ex["cont_obs"]
    print("\n" + "=" * 60)
    print("EXAMPLE TIMESTEP INPUT (record index 0)")
    print("=" * 60)
    print("\n[Continuous obs — 37-D]")
    idx = 0
    for name, dim in [
        ("eef_state", 6), ("joint_state", 7), ("gripper_state", 2),
        ("is_gripper_close", 1), ("front_camera_intrinsic", 9),
        ("wrist_camera_intrinsic", 9), ("is_completed", 1),
        ("is_subgoal_boundary", 1), ("is_video_demo", 1),
    ]:
        print(f"  {name} ({dim}-D): {cont[idx: idx + dim]}")
        idx += dim
    print("\n[Categorical fields]")
    id2tok = {field: {v: k for k, v in vocabs[field]._tok2id.items()} for field in CAT_FIELDS}
    for field in CAT_FIELDS:
        fid  = ex["cat_ids"][field]
        text = id2tok[field].get(fid, "<UNK>")
        print(f"  {field}: id={fid}  string={text!r}")
    print(f"\n[Action — {len(ex['action'])}-D  ({args.action_key})]")
    print(f"  {ex['action']}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------
    rng   = np.random.default_rng(args.seed)
    order = rng.permutation(len(all_records)).tolist()
    n_val = max(1, int(len(all_records) * args.val_fraction))

    train_ds = BinFillDataset(
        [all_records[i] for i in order[n_val:]],
        normalize_states=args.normalize_states,
    )
    val_ds = BinFillDataset(
        [all_records[i] for i in order[:n_val]],
        normalize_states=args.normalize_states,
        obs_mean=train_ds.obs_mean,
        obs_std=train_ds.obs_std,
    )
    print(f"Train transitions: {len(train_ds):,}   Val transitions: {len(val_ds):,}")

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

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    vocab_sizes = {field: len(vocabs[field]) for field in CAT_FIELDS}
    policy = FlowMatchingPolicy(
        action_dim  = action_dim,
        vocab_sizes = vocab_sizes,
        embed_dim   = args.embed_dim,
        context_dim = args.context_dim,
        hidden_dim  = args.hidden_dim,
        time_emb_dim= args.time_emb_dim,
    ).to(device)

    total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    enc_params   = sum(p.numel() for p in policy.encoder.parameters())
    vf_params    = sum(p.numel() for p in policy.vf_net.parameters())
    print(f"\nModel params: {total_params:,}  "
          f"(encoder={enc_params:,}, vector_field_net={vf_params:,})")
    print(f"ODE inference steps: {args.n_inference_steps}")

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   sync_tensorboard=True, config=vars(args), name=run_name,
                   save_code=True, group="FlowMatching-BinFill")

    best_eval_loss = float("inf")
    iteration      = 0
    start_time     = time.time()

    print(f"\nTraining for {args.total_iters} iterations  "
          f"(batch={args.batch_size}, action_key={args.action_key})")
    print(f"{'Iter':>7}  {'TrainFMLoss':>12}  {'TrainAcc':>9}  "
          f"{'EvalFMLoss':>11}  {'EvalAcc':>8}  {'EvalMAE':>8}  {'Time':>6}")
    print("-" * 78)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    policy.train()
    while iteration < args.total_iters:
        for cont, cat, action in train_loader:
            if iteration >= args.total_iters:
                break

            cont   = cont.to(device)
            cat    = {f: v.to(device) for f, v in cat.items()}
            action = action.to(device)

            loss = policy.compute_loss(cont, cat, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % args.log_freq == 0:
                # Quick train accuracy: one ODE sample on current batch
                with torch.no_grad():
                    pred     = policy.sample(cont, cat, n_steps=args.n_inference_steps)
                    tr_acc   = action_accuracy(pred, action, args.acc_threshold)
                writer.add_scalar("train/fm_loss",  loss.item(), iteration)
                writer.add_scalar("train/accuracy", tr_acc,      iteration)

            if iteration % args.eval_freq == 0:
                eval_m = evaluate_on_dataset(
                    policy, val_loader, device, args.acc_threshold, args.n_inference_steps
                )
                with torch.no_grad():
                    pred   = policy.sample(cont, cat, n_steps=args.n_inference_steps)
                    tr_acc = action_accuracy(pred, action, args.acc_threshold)
                elapsed = time.time() - start_time
                print(
                    f"{iteration:>7d}  "
                    f"{loss.item():>12.6f}  "
                    f"{tr_acc:>9.4f}  "
                    f"{eval_m['eval_fm_loss']:>11.6f}  "
                    f"{eval_m['eval_acc']:>8.4f}  "
                    f"{eval_m['eval_mae']:>8.6f}  "
                    f"{elapsed:>5.0f}s"
                )
                for k, v in eval_m.items():
                    writer.add_scalar(f"eval/{k}", v, iteration)

                if eval_m["eval_fm_loss"] < best_eval_loss and args.save_model:
                    best_eval_loss = eval_m["eval_fm_loss"]
                    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
                    torch.save(policy.state_dict(),
                               f"runs/{run_name}/checkpoints/best_eval_loss.pt")
                    print(f"  -> new best eval_fm_loss={best_eval_loss:.6f}. Checkpoint saved.")

            iteration += 1

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    print("\n--- Final evaluation on validation set ---")
    final_m = evaluate_on_dataset(
        policy, val_loader, device, args.acc_threshold, args.n_inference_steps
    )
    for k, v in final_m.items():
        print(f"  {k}: {v:.6f}")
        writer.add_scalar(f"eval/{k}", v, iteration)

    if args.save_model:
        os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
        torch.save(policy.state_dict(), f"runs/{run_name}/checkpoints/final.pt")
        print(f"Final model saved to runs/{run_name}/checkpoints/final.pt")

    writer.close()
    if args.track:
        import wandb
        wandb.finish()
