import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium import spaces

from agents.replay_buffer import ReplayBuffer
from agents.base import BaseAgent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# Observation utilities (GENERIC)
# ---------------------------------------------------------------------
def flat_obs_dim(obs_space):
    if isinstance(obs_space, spaces.Box):
        return int(np.prod(obs_space.shape))

    elif isinstance(obs_space, spaces.Dict):
        dim = 0
        for space in obs_space.spaces.values():
            if isinstance(space, spaces.Box):
                dim += int(np.prod(space.shape))
            elif isinstance(space, spaces.Discrete):
                dim += 1
            else:
                raise ValueError(f"Unsupported subspace: {space}")
        return dim

    else:
        raise ValueError(f"Unsupported obs space: {obs_space}")


def flatten_obs(obs):
    if isinstance(obs, dict):
        out = []
        for k in sorted(obs.keys()):
            v = obs[k]
            if np.isscalar(v):
                out.append(np.array([v], dtype=np.float32))  # 👈 FIX
            else:
                out.append(np.asarray(v, dtype=np.float32).ravel())
        return np.concatenate(out)
    else:
        return np.asarray(obs, dtype=np.float32).ravel()



# ---------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------
# DQN Agent (GENERIC)
# ---------------------------------------------------------------------

class DQNAgent(BaseAgent):
    def __init__(
        self,
        obs_space,
        action_space,
        gamma=0.99,
        lr=1e-3,
        buffer_size=int(1e6),
        batch_size=256,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=150000,
        target_update_freq=1000,
        hidden_sizes=(256, 256),
        device=DEVICE,
        **kwargs,
    ):
        assert isinstance(action_space, spaces.Discrete), \
            "DQN requires a discrete action space"

        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # --- Exploration ---
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.total_steps = 0

        # --- Dimensions ---
        self.act_dim = action_space.n
        self.obs_dim = flat_obs_dim(obs_space)

        # --- Networks ---
        self.q = QNetwork(self.obs_dim, self.act_dim, hidden_sizes).to(device)
        self.q_target = QNetwork(self.obs_dim, self.act_dim, hidden_sizes).to(device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=lr)

        # --- Replay buffer ---
        self.replay_buffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            act_dim=1,
            size=buffer_size,
        )

    # -----------------------------------------------------------------
    # Interaction
    # -----------------------------------------------------------------

    def act(self, obs, explore=True):
        obs_vec = flatten_obs(obs)

        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)

        obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_vals = self.q(obs_t)

        return int(torch.argmax(q_vals, dim=1).item())


    def store(self, obs, action, reward, next_obs, done):
        self.replay_buffer.store(
            flatten_obs(obs),
            np.array([action], dtype=np.float32),
            reward,
            flatten_obs(next_obs),
            done,
        )


    # -----------------------------------------------------------------
    # Learning
    # -----------------------------------------------------------------

    def update(self):
        if self.replay_buffer.size < self.batch_size:
            return None

        self.total_steps += 1

        batch = self.replay_buffer.sample_batch(self.batch_size)

        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        acts = torch.tensor(batch["acts"], dtype=torch.long, device=self.device).squeeze(-1)
        rews = torch.tensor(batch["rews"], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        done = torch.tensor(batch["done"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_q = self.q_target(next_obs).max(dim=1)[0]
            target = rews + self.gamma * (1.0 - done) * next_q

        q_vals = self.q(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(q_vals, target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.total_steps % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        # --- Linear epsilon decay ---
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - self.total_steps / self.epsilon_decay,
        )

        # Return metrics for logging
        return {
            "loss": loss.item(),
            "q_value_mean": q_vals.mean().item(),
            "epsilon": self.epsilon,
        }

    def reset(self):
        pass

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.q.eval()
