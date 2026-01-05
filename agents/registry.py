from agents.random_agent import RandomAgent
from agents.keyboard_agent import KeyboardAgent


AGENT_REGISTRY = {
    "random": RandomAgent,
    "keyboard": KeyboardAgent,
}


def make_agent(agent_cfg, env):
    agent_type = agent_cfg["type"]

    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent_cls = AGENT_REGISTRY[agent_type]

    return agent_cls(
        action_space=env.action_space,
        obs_space=env.observation_space,
        env_type=agent_cfg.get("env_type", "discrete"),
        **agent_cfg,
    )
