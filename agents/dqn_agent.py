import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from collections import deque
import random
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Utilities
# ============================================================

def flatten_obs(obs):
    """Flatten dict or array observation into 1D float32 numpy array."""
    if isinstance(obs, dict):
        return np.concatenate(
            [np.asarray(v, dtype=np.float32).ravel() for v in obs.values()]
        )
    else:
        return np.asarray(obs, dtype=np.float32).ravel()


def obs_dim_from_space(space):
    if isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    elif isinstance(space, spaces.Dict):
        return sum(int(np.prod(s.shape)) for s in space.spaces.values())
    else:
        raise ValueError(f"Unsupported obs space: {space}")


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, obs, act, rew, next_obs, done):
        self.buffer.append((obs, act, rew, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = zip(*batch)
        return (
            torch.tensor(obs, dtype=torch.float32, device=DEVICE),
            torch.tensor(act, dtype=torch.long, device=DEVICE),
            torch.tensor(rew, dtype=torch.float32, device=DEVICE),
            torch.tensor(next_obs, dtype=torch.float32, device=DEVICE),
            torch.tensor(done, dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Q Network
# ============================================================

network_size = 64
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, network_size),
            nn.ReLU(),
            nn.Linear(network_size, network_size),
            nn.ReLU(),
            nn.Linear(network_size, act_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# DQN Agent
# ============================================================

class DQNAgent:
    def __init__(
        self,
        obs_space,
        action_space,
        gamma=0.99,
        lr=0.001,
        buffer_size=int(1e6),
        batch_size=256,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=750_000,
        target_update_freq=1000,
        **kwargs,
    ):
        assert isinstance(action_space, spaces.Discrete)

        self.obs_dim = obs_dim_from_space(obs_space)
        self.act_dim = action_space.n

        self.q = QNetwork(self.obs_dim, self.act_dim).to(DEVICE)
        self.q_target = QNetwork(self.obs_dim, self.act_dim).to(DEVICE)
        self.q_target.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=lr)

        self.replay = ReplayBuffer(buffer_size)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start

        self.total_steps = 0

    # --------------------------------------------------------

    def act(self, obs, explore=True):
        obs_vec = flatten_obs(obs)
        obs_t = torch.tensor(obs_vec, device=DEVICE).unsqueeze(0)

        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)

        with torch.no_grad():
            q = self.q(obs_t)
        return int(q.argmax(dim=1).item())

    # --------------------------------------------------------

    def store(self, obs, act, rew, next_obs, done):
        self.replay.store(
            flatten_obs(obs),
            act,
            rew,
            flatten_obs(next_obs),
            done,
        )

    # --------------------------------------------------------

    def update(self):
        if len(self.replay) < self.batch_size:
            return None

        self.total_steps += 1

        obs, act, rew, next_obs, done = self.replay.sample(self.batch_size)


        with torch.no_grad():
            next_q = self.q_target(next_obs).max(dim=1)[0]
            target = rew + self.gamma * (1.0 - done) * next_q

        q = self.q(obs).gather(1, act.unsqueeze(1)).squeeze(1)

        loss = nn.functional.mse_loss(q, target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # if self.total_steps % self.target_update_freq == 0:
        tau = 0.001
        for param, target_param in zip(self.q.parameters(), self.q_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Linear epsilon decay
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - self.total_steps / self.epsilon_decay,
        )

        return {
            "loss": loss.item(),
            "q_value_mean": q.mean().item(),
            "epsilon": self.epsilon,
        }

    # --------------------------------------------------------

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location=DEVICE))
        self.q_target.load_state_dict(self.q.state_dict())
