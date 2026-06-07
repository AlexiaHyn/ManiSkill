"""
Behavioral Cloning on the RoboMME BinFill h5 dataset.

No live environment is needed. Evaluation is done on a held-out 20% split
of episodes from the same h5 file.

Continuous obs (37-D):
  eef_state(6) + joint_state(7) + gripper_state(2) + is_gripper_close(1)
  + front_camera_intrinsic(9) + wrist_camera_intrinsic(9)
  + is_completed(1) + is_subgoal_boundary(1) + is_video_demo(1)

Categorical obs (embedded, then concatenated):
  difficulty      (from setup)  -- e.g. "easy" / "medium" / "hard"
  task_goal       (from setup)  -- episode-level goal string
  simple_subgoal  (from info)   -- current subgoal string per timestep

Text vocabularies are built from the loaded episodes at startup.
Each categorical field gets its own nn.Embedding (embed_dim=16).

Metrics printed every --eval_freq iterations:
  train_loss / train_acc  -- on the current mini-batch
  eval_loss  / eval_acc   -- on the full held-out validation set
  eval_mae                -- mean absolute error per action dim (val set)
"""

import json
import os
import random
import time
from dataclasses import dataclass
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
    h5_file: str = "robomme_data/record_dataset_BinFill.h5"
    """path to the BinFill h5 file"""
    action_key: str = "eef_action"
    """which stored action to imitate: eef_action (7-D), joint_action (8-D), waypoint_action (7-D)"""
    val_fraction: float = 0.2
    """fraction of episodes held out for validation"""
    num_demos: Optional[int] = None
    """cap on number of episodes to load; None = all"""
    normalize_states: bool = False
    """normalise continuous observations to zero mean / unit std"""

    # Model
    embed_dim: int = 16
    """embedding dimension for each categorical field"""
    hidden_dim: int = 256
    """MLP hidden layer width"""

    # Training
    total_iters: int = 100_000
    batch_size: int = 512
    lr: float = 3e-4
    num_dataload_workers: int = 0

    # Logging / eval
    log_freq: int = 100
    eval_freq: int = 500
    acc_threshold: float = 0.05
    """L2 distance threshold for counting a prediction as accurate"""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_DIM = {
    "eef_action": 7,
    "joint_action": 8,
    "waypoint_action": 7,
}

CONT_OBS_DIM = 37   # see module docstring breakdown above

# Categorical field names stored in each sample dict
CAT_FIELDS = ["difficulty", "task_goal", "simple_subgoal"]


# ---------------------------------------------------------------------------
# Vocabulary helper
# ---------------------------------------------------------------------------

class Vocab:
    """Maps string tokens to integer IDs (0 = unknown / padding)."""

    def __init__(self):
        self._tok2id: Dict[str, int] = {}

    def add(self, token: str) -> None:
        if token not in self._tok2id:
            self._tok2id[token] = len(self._tok2id) + 1  # 0 reserved for UNK

    def __call__(self, token: str) -> int:
        return self._tok2id.get(token, 0)

    def __len__(self) -> int:
        return len(self._tok2id) + 1  # +1 for UNK slot


def _decode(raw) -> str:
    """Decode bytes or string h5 scalar to a plain str."""
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_continuous_obs(ts, setup_cont: np.ndarray) -> np.ndarray:
    """Build the 37-D continuous obs vector for one timestep."""
    eef   = ts["obs"]["eef_state"][()].astype(np.float32)          # (6,)
    joint = ts["obs"]["joint_state"][()].astype(np.float32)        # (7,)
    grip  = ts["obs"]["gripper_state"][()].astype(np.float32)      # (2,)
    close = np.array([float(ts["obs"]["is_gripper_close"][()])],
                     dtype=np.float32)                              # (1,)

    is_completed      = np.array([float(ts["info"]["is_completed"][()])],
                                 dtype=np.float32)                  # (1,)
    is_subgoal_bound  = np.array([float(ts["info"]["is_subgoal_boundary"][()])],
                                 dtype=np.float32)                  # (1,)
    is_video_demo     = np.array([float(ts["info"]["is_video_demo"][()])],
                                 dtype=np.float32)                  # (1,)

    # setup_cont = [front_cam_intrinsic(9), wrist_cam_intrinsic(9)] = 18-D
    return np.concatenate([eef, joint, grip, close,
                           setup_cont,
                           is_completed, is_subgoal_bound, is_video_demo])


def _read_setup_cont(ep) -> np.ndarray:
    """18-D continuous part of setup: flattened camera intrinsics."""
    front = ep["setup"]["front_camera_intrinsic"][()].astype(np.float32).ravel()  # (9,)
    wrist = ep["setup"]["wrist_camera_intrinsic"][()].astype(np.float32).ravel()  # (9,)
    return np.concatenate([front, wrist])


def _read_setup_cats(ep) -> Dict[str, str]:
    return {
        "difficulty": _decode(ep["setup"]["difficulty"][()]),
        "task_goal":  _decode(ep["setup"]["task_goal"][0]
                              if ep["setup"]["task_goal"].shape[0] > 0
                              else ep["setup"]["task_goal"][()]),
    }


def _read_info_cats(ts) -> Dict[str, str]:
    return {
        "simple_subgoal": _decode(ts["info"]["simple_subgoal"][()]),
    }


def load_all_episodes(
    h5_file: str,
    action_key: str,
    num_demos: Optional[int],
    vocabs: Dict[str, Vocab],
) -> List[dict]:
    """
    Load every episode and return a list of per-timestep dicts:
        cont_obs   : (37,) float32 np array
        cat_ids    : dict {field: int}
        action     : (action_dim,) float32 np array
    Also populates vocabs in-place.
    """
    records = []
    with h5py.File(h5_file, "r") as f:
        ep_keys = sorted(f.keys())
        if num_demos is not None:
            ep_keys = ep_keys[:num_demos]

        print(f"Loading {len(ep_keys)} episodes from {h5_file} ...")
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

            # update vocabs with setup cats
            for field, val in setup_cats.items():
                vocabs[field].add(val)

            for ts_key in timesteps:
                ts = ep[ts_key]

                # action — skip if it's a Group (shouldn't happen for chosen key)
                act_node = ts["action"][action_key]
                if isinstance(act_node, h5py.Group):
                    continue
                action = act_node[()].astype(np.float32)

                cont_obs = _read_continuous_obs(ts, setup_cont)

                info_cats = _read_info_cats(ts)
                for field, val in info_cats.items():
                    vocabs[field].add(val)

                cat_ids = {
                    field: vocabs[field](val)
                    for field, val in {**setup_cats, **info_cats}.items()
                }

                records.append({
                    "cont_obs": cont_obs,
                    "cat_ids":  cat_ids,
                    "action":   action,
                })

    print(f"  Total transitions: {len(records):,}")
    return records


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BinFillDataset(Dataset):
    def __init__(
        self,
        records: List[dict],
        normalize_states: bool = False,
        obs_mean: Optional[np.ndarray] = None,
        obs_std:  Optional[np.ndarray] = None,
    ):
        cont = np.stack([r["cont_obs"] for r in records], axis=0).astype(np.float32)

        if normalize_states:
            if obs_mean is None:
                obs_mean = cont.mean(axis=0, keepdims=True)
                obs_std  = cont.std(axis=0, keepdims=True) + 1e-8
            cont = (cont - obs_mean) / obs_std

        self.cont    = torch.from_numpy(cont)
        self.actions = torch.from_numpy(
            np.stack([r["action"] for r in records]).astype(np.float32)
        )
        # categorical ids: one tensor per field
        self.cat = {
            field: torch.tensor([r["cat_ids"][field] for r in records], dtype=torch.long)
            for field in CAT_FIELDS
        }
        self.obs_mean = obs_mean
        self.obs_std  = obs_std

    def __len__(self):
        return len(self.cont)

    def __getitem__(self, idx):
        return (
            self.cont[idx],
            {field: self.cat[field][idx] for field in CAT_FIELDS},
            self.actions[idx],
        )


def collate_fn(batch):
    cont    = torch.stack([b[0] for b in batch])
    cat     = {field: torch.stack([b[1][field] for b in batch]) for field in CAT_FIELDS}
    actions = torch.stack([b[2] for b in batch])
    return cont, cat, actions


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """
    MLP that takes continuous obs + embedded categorical fields and predicts action.
    input_dim = CONT_OBS_DIM + len(CAT_FIELDS) * embed_dim
    """

    def __init__(
        self,
        action_dim: int,
        vocab_sizes: Dict[str, int],
        embed_dim: int = 16,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            field: nn.Embedding(vocab_sizes[field], embed_dim)
            for field in CAT_FIELDS
        })
        input_dim = CONT_OBS_DIM + len(CAT_FIELDS) * embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, cont: torch.Tensor, cat: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [cont] + [self.embeddings[f](cat[f]) for f in CAT_FIELDS]
        return self.net(torch.cat(parts, dim=-1))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def action_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> float:
    return ((pred - target).norm(dim=-1) < threshold).float().mean().item()


@torch.no_grad()
def evaluate_on_dataset(
    actor: nn.Module, loader: DataLoader,
    device: torch.device, threshold: float,
) -> Dict[str, float]:
    actor.eval()
    total_loss = total_acc = total_mae = n = 0
    for cont, cat, action in loader:
        cont   = cont.to(device)
        cat    = {f: v.to(device) for f, v in cat.items()}
        action = action.to(device)
        pred   = actor(cont, cat)

        total_loss += F.mse_loss(pred, action, reduction="sum").item()
        total_acc  += (pred - action).norm(dim=-1).lt(threshold).float().sum().item()
        total_mae  += (pred - action).abs().mean(dim=-1).sum().item()
        n          += len(cont)

    actor.train()
    return {
        "eval_loss": total_loss / n,
        "eval_acc":  total_acc  / n,
        "eval_mae":  total_mae  / n,
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
    # Load data and build vocabularies in one pass
    # ------------------------------------------------------------------
    vocabs: Dict[str, Vocab] = {field: Vocab() for field in CAT_FIELDS}
    all_records = load_all_episodes(args.h5_file, args.action_key, args.num_demos, vocabs)

    print("\nVocabulary sizes:")
    for field, vocab in vocabs.items():
        print(f"  {field}: {len(vocab)} tokens")

    # ------------------------------------------------------------------
    # Print one example timestep to verify what goes into the model
    # ------------------------------------------------------------------
    ex = all_records[0]
    cont = ex["cont_obs"]
    print("\n" + "=" * 60)
    print("EXAMPLE TIMESTEP INPUT (record index 0)")
    print("=" * 60)

    print("\n[Continuous obs — 37-D]")
    idx = 0
    labels = [
        ("eef_state",              6),
        ("joint_state",            7),
        ("gripper_state",          2),
        ("is_gripper_close",       1),
        ("front_camera_intrinsic", 9),
        ("wrist_camera_intrinsic", 9),
        ("is_completed",           1),
        ("is_subgoal_boundary",    1),
        ("is_video_demo",          1),
    ]
    for name, dim in labels:
        vals = cont[idx: idx + dim]
        print(f"  {name} ({dim}-D): {vals}")
        idx += dim

    print("\n[Categorical fields — integer vocab IDs + decoded strings]")
    id2tok = {field: {v: k for k, v in vocabs[field]._tok2id.items()}
              for field in CAT_FIELDS}
    for field in CAT_FIELDS:
        fid  = ex["cat_ids"][field]
        text = id2tok[field].get(fid, "<UNK>")
        print(f"  {field}: id={fid}  string={text!r}")

    print(f"\n[Action — {len(ex['action'])}-D  ({args.action_key})]")
    print(f"  {ex['action']}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Train / val split by shuffling record indices
    # ------------------------------------------------------------------
    rng   = np.random.default_rng(args.seed)
    order = rng.permutation(len(all_records)).tolist()
    n_val = max(1, int(len(all_records) * args.val_fraction))

    train_records = [all_records[i] for i in order[n_val:]]
    val_records   = [all_records[i] for i in order[:n_val]]

    train_ds = BinFillDataset(train_records, normalize_states=args.normalize_states)
    val_ds   = BinFillDataset(
        val_records,
        normalize_states=args.normalize_states,
        obs_mean=train_ds.obs_mean,
        obs_std=train_ds.obs_std,
    )
    print(f"\nTrain transitions: {len(train_ds):,}   Val transitions: {len(val_ds):,}")

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
    actor = Actor(action_dim, vocab_sizes,
                  embed_dim=args.embed_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    print(f"Actor params: {total_params:,}  |  "
          f"input_dim={CONT_OBS_DIM + len(CAT_FIELDS)*args.embed_dim}")

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
                   save_code=True, group="BehaviorCloning-BinFill")

    best_eval_loss = float("inf")
    iteration      = 0
    start_time     = time.time()

    print(f"\nTraining for {args.total_iters} iterations  "
          f"(batch={args.batch_size}, action_key={args.action_key})")
    print(f"{'Iter':>7}  {'TrainLoss':>10}  {'TrainAcc':>9}  "
          f"{'EvalLoss':>9}  {'EvalAcc':>8}  {'EvalMAE':>8}  {'Time':>6}")
    print("-" * 72)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    actor.train()
    while iteration < args.total_iters:
        for cont, cat, action in train_loader:
            if iteration >= args.total_iters:
                break

            cont   = cont.to(device)
            cat    = {f: v.to(device) for f, v in cat.items()}
            action = action.to(device)

            pred = actor(cont, cat)
            loss = F.mse_loss(pred, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % args.log_freq == 0:
                with torch.no_grad():
                    tr_acc = action_accuracy(pred, action, args.acc_threshold)
                writer.add_scalar("train/loss",     loss.item(), iteration)
                writer.add_scalar("train/accuracy", tr_acc,      iteration)

            if iteration % args.eval_freq == 0:
                eval_m = evaluate_on_dataset(actor, val_loader, device, args.acc_threshold)
                with torch.no_grad():
                    tr_acc = action_accuracy(pred, action, args.acc_threshold)
                elapsed = time.time() - start_time
                print(
                    f"{iteration:>7d}  "
                    f"{loss.item():>10.6f}  "
                    f"{tr_acc:>9.4f}  "
                    f"{eval_m['eval_loss']:>9.6f}  "
                    f"{eval_m['eval_acc']:>8.4f}  "
                    f"{eval_m['eval_mae']:>8.6f}  "
                    f"{elapsed:>5.0f}s"
                )
                for k, v in eval_m.items():
                    writer.add_scalar(f"eval/{k}", v, iteration)

                if eval_m["eval_loss"] < best_eval_loss and args.save_model:
                    best_eval_loss = eval_m["eval_loss"]
                    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
                    torch.save(actor.state_dict(),
                               f"runs/{run_name}/checkpoints/best_eval_loss.pt")
                    print(f"  -> new best eval_loss={best_eval_loss:.6f}. Checkpoint saved.")

            iteration += 1

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    print("\n--- Final evaluation on validation set ---")
    final_m = evaluate_on_dataset(actor, val_loader, device, args.acc_threshold)
    for k, v in final_m.items():
        print(f"  {k}: {v:.6f}")
        writer.add_scalar(f"eval/{k}", v, iteration)

    if args.save_model:
        os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
        torch.save(actor.state_dict(), f"runs/{run_name}/checkpoints/final.pt")
        print(f"Final model saved to runs/{run_name}/checkpoints/final.pt")

    writer.close()
    if args.track:
        import wandb
        wandb.finish()
