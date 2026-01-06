import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium import spaces

from agents.networks import Actor, Critic
from agents.replay_buffer import ReplayBuffer
from agents.base import BaseAgent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO move to utils
def flat_obs_dim(obs_space):
    if isinstance(obs_space, spaces.Box):
        return int(np.prod(obs_space.shape))
    elif isinstance(obs_space, spaces.Dict):
        return sum(int(np.prod(s.shape)) for s in obs_space.spaces.values())
    else:
        raise ValueError(f"Unsupported observation space: {obs_space}")

# TODO move to utils
def _flatten_obs(obs):
    if isinstance(obs, dict):
        # deterministic ordering
        return np.concatenate(
            [np.asarray(obs[k]).ravel() for k in sorted(obs.keys())]
        )
    return np.asarray(obs).ravel()


class TD3Agent(BaseAgent):
    def __init__(
        self,
        obs_space,
        action_space,
        gamma=0.99,
        tau=0.005,
        lr=1e-3,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        buffer_size=int(1e6),
        batch_size=256,
        device=DEVICE,
        **kwargs
    ):
        assert isinstance(action_space, spaces.Box), \
            "TD3 requires a continuous (Box) action space"

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size

        # --- Infer action properties ---
        self.act_dim = int(np.prod(action_space.shape))
        self.act_limit = float(action_space.high[0])

        # --- Infer observation dimension ---
        self.obs_dim = flat_obs_dim(obs_space)

        # --- Networks ---
        self.actor = Actor(self.obs_dim, self.act_dim, self.act_limit).to(device)
        self.actor_target = Actor(self.obs_dim, self.act_dim, self.act_limit).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(self.obs_dim, self.act_dim).to(device)
        self.critic2 = Critic(self.obs_dim, self.act_dim).to(device)
        self.critic1_target = Critic(self.obs_dim, self.act_dim).to(device)
        self.critic2_target = Critic(self.obs_dim, self.act_dim).to(device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )

        self.replay_buffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            size=buffer_size,
        )

        self.total_it = 0


    def act(self, obs, explore=True):
        obs = _flatten_obs(obs)
        obs_t = torch.tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]

        if explore:
            action += np.random.normal(0, self.policy_noise, size=action.shape)

        return np.clip(action, -self.act_limit, self.act_limit)


    def store(self, obs, action, reward, next_obs, done):
        self.replay_buffer.store(
            _flatten_obs(obs),
            action,
            reward,
            _flatten_obs(next_obs),
            done,
        )


    def update(self):
        if self.replay_buffer.size < self.batch_size:
            return

        self.total_it += 1
        batch = self.replay_buffer.sample_batch(self.batch_size)

        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        acts = torch.tensor(batch["acts"], dtype=torch.float32, device=self.device)
        rews = torch.tensor(batch["rews"], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        done = torch.tensor(batch["done"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            noise = (
                torch.randn_like(acts) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_actions = (
                self.actor_target(next_obs) + noise
            ).clamp(-self.act_limit, self.act_limit)

            target_q1 = self.critic1_target(next_obs, next_actions)
            target_q2 = self.critic2_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)

            backup = rews + self.gamma * (1 - done) * target_q

        q1 = self.critic1(obs, acts)
        q2 = self.critic2(obs, acts)

        critic_loss = ((q1 - backup) ** 2 + (q2 - backup) ** 2).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

    def _soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.q.eval()
