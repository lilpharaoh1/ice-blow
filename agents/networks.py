import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            act = activation if i < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = MLP([obs_dim, 256, 256, act_dim], output_activation=nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q = MLP([obs_dim + act_dim, 256, 256, 1])

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1)).squeeze(-1)
