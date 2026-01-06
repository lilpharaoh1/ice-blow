from agents.random_agent import RandomAgent
from agents.keyboard_agent import KeyboardAgent
from agents.dqn_agent import DQNAgent
from agents.td3_agent import TD3Agent


AGENT_REGISTRY = {
    "random": RandomAgent,
    "keyboard": KeyboardAgent,
    "dqn": DQNAgent,
    "td3": TD3Agent,
}


def make_agent(cfg, env):
    agent_type = cfg.agent["type"]

    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent_cls = AGENT_REGISTRY[agent_type]

    return agent_cls(
        cfg=cfg,
        action_space=env.action_space,
        obs_space=env.observation_space,
        env_type=cfg.env.get("type", "discrete"),
    )
