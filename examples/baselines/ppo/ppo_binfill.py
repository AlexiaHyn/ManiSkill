"""
Offline PPO trained on the RoboMME BinFill h5 dataset.

Data comes from record_dataset_BinFill.h5 instead of a live environment.
State obs = [eef_state(6), joint_state(7), gripper_state(2), is_gripper_close(1)] = 16-D.
Rewards are derived from info.is_subgoal_boundary and info.is_completed (sparse).
Log-probs are recomputed under the current policy at the start of every iteration so
the PPO importance-sampling ratio stays valid.
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Hyper-parameters
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
    """which stored action to use: eef_action (7-D), joint_action (8-D), waypoint_action (7-D)"""
    subgoal_reward: float = 1.0
    """reward given when is_subgoal_boundary=True"""
    completion_reward: float = 5.0
    """reward given when is_completed=True"""

    # Algorithm
    total_timesteps: int = 10_000_000
    learning_rate: float = 3e-4
    num_envs: int = 64
    """number of virtual parallel trajectories sampled per iteration"""
    num_steps: int = 50
    """timesteps collected per virtual env per iteration"""
    gamma: float = 0.8
    gae_lambda: float = 0.9
    num_minibatches: int = 32
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.1
    reward_scale: float = 1.0
    anneal_lr: bool = False
    log_freq: int = 10
    """log to tensorboard every N iterations"""

    # filled at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

OBS_DIM = 16   # eef(6) + joint(7) + gripper(2) + is_close(1)

ACTION_DIM = {
    "eef_action": 7,
    "joint_action": 8,
    "waypoint_action": 7,
}


class BinFillDataset:
    """
    Loads the RoboMME BinFill h5 file and exposes a flat array of transitions.

    Arrays stored (all float32, on CPU):
        obs     : (N, OBS_DIM)
        actions : (N, action_dim)
        rewards : (N,)
        dones   : (N,)   1.0 at last step of each episode

    episode_ends : sorted list of indices where episodes end (for boundary detection).
    """

    def __init__(self, h5_file: str, action_key: str = "eef_action",
                 subgoal_reward: float = 1.0, completion_reward: float = 5.0):
        self.action_key = action_key
        all_obs, all_actions, all_rewards, all_dones = [], [], [], []
        self.episode_ends = []
        cursor = 0

        print(f"Loading dataset from {h5_file} ...")
        with h5py.File(h5_file, "r") as f:
            episodes = sorted(f.keys())
            print(f"  Found {len(episodes)} episodes")
            for ep_key in episodes:
                ep = f[ep_key]
                timesteps = sorted(
                    [k for k in ep.keys() if k.startswith("timestep_")],
                    key=lambda x: int(x.split("_")[1]),
                )
                if not timesteps:
                    continue

                ep_obs, ep_act, ep_rew, ep_done = [], [], [], []
                for ts_key in timesteps:
                    ts = ep[ts_key]

                    # --- observation ---
                    eef   = ts["obs"]["eef_state"][()].astype(np.float32)        # (6,)
                    joint = ts["obs"]["joint_state"][()].astype(np.float32)      # (7,)
                    grip  = ts["obs"]["gripper_state"][()].astype(np.float32)    # (2,)
                    close = np.array(
                        [float(ts["obs"]["is_gripper_close"][()])], dtype=np.float32
                    )                                                              # (1,)
                    ep_obs.append(np.concatenate([eef, joint, grip, close]))

                    # --- action ---
                    act = ts["action"][action_key][()].astype(np.float32)
                    ep_act.append(act)

                    # --- reward ---
                    is_subgoal   = bool(ts["info"]["is_subgoal_boundary"][()])
                    is_completed = bool(ts["info"]["is_completed"][()])
                    reward = (subgoal_reward if is_subgoal else 0.0) + \
                             (completion_reward if is_completed else 0.0)
                    ep_rew.append(np.float32(reward))

                    # done when task completed mid-episode
                    ep_done.append(np.float32(1.0 if is_completed else 0.0))

                # last step of the episode is always a boundary
                ep_done[-1] = np.float32(1.0)

                T = len(ep_obs)
                all_obs.append(np.stack(ep_obs))
                all_actions.append(np.stack(ep_act))
                all_rewards.append(np.array(ep_rew, dtype=np.float32))
                all_dones.append(np.array(ep_done, dtype=np.float32))

                cursor += T
                self.episode_ends.append(cursor - 1)  # index of last step

        self.obs     = torch.from_numpy(np.concatenate(all_obs,     axis=0))  # (N, 16)
        self.actions = torch.from_numpy(np.concatenate(all_actions, axis=0))  # (N, action_dim)
        self.rewards = torch.from_numpy(np.concatenate(all_rewards, axis=0))  # (N,)
        self.dones   = torch.from_numpy(np.concatenate(all_dones,   axis=0))  # (N,)
        self.N = len(self.obs)
        self.episode_ends_set = set(self.episode_ends)
        print(f"  Total transitions: {self.N}")

    def sample_batch(self, num_envs: int, num_steps: int, device: torch.device):
        """
        Sample `num_envs` random starting indices and collect `num_steps` transitions
        from each, wrapping at episode boundaries.

        Returns tensors of shape (num_steps, num_envs, *) on `device`.
        """
        obs_buf     = torch.zeros(num_steps, num_envs, OBS_DIM,                     device=device)
        action_buf  = torch.zeros(num_steps, num_envs, self.actions.shape[1],        device=device)
        reward_buf  = torch.zeros(num_steps, num_envs,                               device=device)
        done_buf    = torch.zeros(num_steps, num_envs,                               device=device)

        # Pick random starting positions
        starts = torch.randint(0, self.N, (num_envs,))

        for env_i, start in enumerate(starts.tolist()):
            idx = start
            for step in range(num_steps):
                obs_buf[step, env_i]    = self.obs[idx]
                action_buf[step, env_i] = self.actions[idx]
                reward_buf[step, env_i] = self.rewards[idx]
                done_buf[step, env_i]   = self.dones[idx]

                # advance; wrap to next episode start if at episode end
                if self.dones[idx] == 1.0:
                    # find next valid start (cycle through dataset)
                    idx = (idx + 1) % self.N
                else:
                    idx = (idx + 1) % self.N

        return obs_buf, action_buf, reward_buf, done_buf


# ---------------------------------------------------------------------------
# Policy network  (identical architecture to ppo.py)
# ---------------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)),     nn.Tanh(),
            layer_init(nn.Linear(256, 256)),     nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),       nn.Tanh(),
            layer_init(nn.Linear(256, 256)),            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.critic(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size     = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
    run_name = f"BinFill__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")

    # --- dataset ---
    dataset = BinFillDataset(
        args.h5_file,
        action_key=args.action_key,
        subgoal_reward=args.subgoal_reward,
        completion_reward=args.completion_reward,
    )
    action_dim = ACTION_DIM[args.action_key]

    # --- agent ---
    agent = Agent(OBS_DIM, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- logging ---
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   sync_tensorboard=True, config=vars(args), name=run_name,
                   save_code=True, group="PPO-BinFill")

    # --- storage buffers ---
    obs_buf     = torch.zeros(args.num_steps, args.num_envs, OBS_DIM,     device=device)
    action_buf  = torch.zeros(args.num_steps, args.num_envs, action_dim,  device=device)
    logprob_buf = torch.zeros(args.num_steps, args.num_envs,              device=device)
    reward_buf  = torch.zeros(args.num_steps, args.num_envs,              device=device)
    done_buf    = torch.zeros(args.num_steps, args.num_envs,              device=device)
    value_buf   = torch.zeros(args.num_steps, args.num_envs,              device=device)

    global_step = 0
    start_time  = time.time()

    print(f"num_iterations={args.num_iterations}  batch={args.batch_size}  "
          f"minibatch={args.minibatch_size}  update_epochs={args.update_epochs}")

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ------------------------------------------------------------------
        # 1. Sample a batch of transitions from the h5 dataset
        # ------------------------------------------------------------------
        sampled_obs, sampled_act, sampled_rew, sampled_done = dataset.sample_batch(
            args.num_envs, args.num_steps, device
        )

        obs_buf    = sampled_obs
        action_buf = sampled_act
        reward_buf = sampled_rew * args.reward_scale
        done_buf   = sampled_done

        # ------------------------------------------------------------------
        # 2. Compute log-probs and values under the CURRENT policy
        #    (this makes the PPO ratio valid for this iteration)
        # ------------------------------------------------------------------
        agent.eval()
        with torch.no_grad():
            flat_obs = obs_buf.reshape(-1, OBS_DIM)
            flat_act = action_buf.reshape(-1, action_dim)
            _, flat_logprob, _, flat_value = agent.get_action_and_value(flat_obs, flat_act)
            logprob_buf = flat_logprob.reshape(args.num_steps, args.num_envs)
            value_buf   = flat_value.reshape(args.num_steps, args.num_envs)

        global_step += args.batch_size

        # ------------------------------------------------------------------
        # 3. GAE advantage estimation
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Bootstrap from the last obs in each virtual env
            last_obs = obs_buf[-1]                                  # (num_envs, obs_dim)
            next_value = agent.get_value(last_obs).reshape(1, -1)  # (1, num_envs)

            advantages = torch.zeros_like(reward_buf)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - done_buf[t]
                    nextvalues    = next_value
                else:
                    next_not_done = 1.0 - done_buf[t + 1]
                    nextvalues    = value_buf[t + 1]
                delta = reward_buf[t] + args.gamma * nextvalues * next_not_done - value_buf[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
                )
            returns = advantages + value_buf

        # ------------------------------------------------------------------
        # 4. Flatten for minibatch updates
        # ------------------------------------------------------------------
        b_obs       = obs_buf.reshape(-1, OBS_DIM)
        b_actions   = action_buf.reshape(-1, action_dim)
        b_logprobs  = logprob_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns   = returns.reshape(-1)
        b_values    = value_buf.reshape(-1)

        # ------------------------------------------------------------------
        # 5. PPO update
        # ------------------------------------------------------------------
        agent.train()
        b_inds     = np.arange(args.batch_size)
        clipfracs  = []
        update_time = time.time()

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end    = start + args.minibatch_size
                mb_idx = b_inds[start:end]

                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb_idx], b_actions[mb_idx]
                )
                logratio = new_logprob - b_logprobs[mb_idx]
                ratio    = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl     = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_adv = b_advantages[mb_idx]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # policy loss
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()

                # value loss
                new_value = new_value.view(-1)
                if args.clip_vloss:
                    v_unclipped = (new_value - b_returns[mb_idx]) ** 2
                    v_clipped   = b_values[mb_idx] + torch.clamp(
                        new_value - b_values[mb_idx], -args.clip_coef, args.clip_coef
                    )
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_returns[mb_idx]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time

        # ------------------------------------------------------------------
        # 6. Logging
        # ------------------------------------------------------------------
        if iteration % args.log_freq == 0 or iteration == 1:
            y_pred  = b_values.cpu().numpy()
            y_true  = b_returns.cpu().numpy()
            var_y   = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            sps = int(global_step / (time.time() - start_time))
            print(f"Iter {iteration:5d} | step {global_step:9d} | "
                  f"pg={pg_loss.item():.4f}  v={v_loss.item():.4f}  "
                  f"ent={entropy_loss.item():.4f}  kl={approx_kl.item():.4f}  "
                  f"SPS={sps}")

            writer.add_scalar("charts/learning_rate",    optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS",              sps,                             global_step)
            writer.add_scalar("losses/policy_loss",      pg_loss.item(),                  global_step)
            writer.add_scalar("losses/value_loss",       v_loss.item(),                   global_step)
            writer.add_scalar("losses/entropy",          entropy_loss.item(),             global_step)
            writer.add_scalar("losses/approx_kl",        approx_kl.item(),               global_step)
            writer.add_scalar("losses/old_approx_kl",    old_approx_kl.item(),           global_step)
            writer.add_scalar("losses/clipfrac",         np.mean(clipfracs),             global_step)
            writer.add_scalar("losses/explained_variance", explained_var,                global_step)
            writer.add_scalar("time/update_time",        update_time,                    global_step)
            writer.add_scalar("data/mean_reward",        reward_buf.mean().item(),       global_step)
            writer.add_scalar("data/mean_done",          done_buf.mean().item(),         global_step)

    # ------------------------------------------------------------------
    # Save final checkpoint
    # ------------------------------------------------------------------
    if args.save_model:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    writer.close()
