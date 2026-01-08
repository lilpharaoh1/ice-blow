from agents.random_agent import RandomAgent
from agents.keyboard_agent import KeyboardAgent
from agents.dqn_agent import DQNAgent
from agents.td3_agent import TD3Agent
from agents.pathfinding_agent import PathfindingAgent
from agents.assisted_pathfinding_agent import AssistedPathfindingAgent
from agents.pathfinding_discrete_agent import PathfindingDiscreteAgent
from agents.assisted_pathfinding_discrete_agent import AssistedPathfindingDiscreteAgent


AGENT_REGISTRY = {
    "random": RandomAgent,
    "keyboard": KeyboardAgent,
    "dqn": DQNAgent,
    "td3": TD3Agent,
    # Gridworld pathfinding
    "pathfinding": PathfindingAgent,
    "assisted_pathfinding": AssistedPathfindingAgent,
    # Discrete (continuous position) pathfinding
    "pathfinding_discrete": PathfindingDiscreteAgent,
    "assisted_pathfinding_discrete": AssistedPathfindingDiscreteAgent,
}


def make_agent(cfg, env):
    agent_type = cfg.agent["type"]

    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent_cls = AGENT_REGISTRY[agent_type]

    env_type = cfg.env["type"]

    # Extract agent-specific parameters from config (exclude 'type')
    agent_kwargs = {k: v for k, v in cfg.agent.items() if k != "type"}

    # Extract blow parameters and map to agent parameter names
    blow_kwargs = {}
    if hasattr(cfg, 'blow') and cfg.blow:
        blow_config = cfg.blow
        # Map config names to agent parameter names
        if 'width' in blow_config:
            blow_kwargs['blow_width'] = blow_config['width']
        if 'warning_duration' in blow_config:
            blow_kwargs['warning_duration'] = blow_config['warning_duration']
        if 'num_lines' in blow_config:
            blow_kwargs['num_blow_lines'] = blow_config['num_lines']
        if 'active_duration' in blow_config:
            blow_kwargs['active_duration'] = blow_config['active_duration']
        if 'interval' in blow_config:
            blow_kwargs['blow_interval'] = blow_config['interval']

    # Extract env parameters that agents might need
    env_kwargs = {}
    if hasattr(cfg, 'env') and cfg.env:
        env_config = cfg.env
        if 'grid_size' in env_config:
            env_kwargs['grid_size'] = env_config['grid_size']
            env_kwargs['world_size'] = env_config['grid_size']  # Alias for discrete env
        if 'dt' in env_config:
            env_kwargs['dt'] = env_config['dt']
        if 'friction' in env_config:
            env_kwargs['friction'] = env_config['friction']
        if 'vel_scale' in env_config:
            env_kwargs['vel_scale'] = env_config['vel_scale']
        if 'max_vel' in env_config:
            env_kwargs['max_vel'] = env_config['max_vel']

    # Merge all kwargs - agent_kwargs takes priority if there are conflicts
    all_kwargs = {**env_kwargs, **blow_kwargs, **agent_kwargs}

    return agent_cls(
        action_space=env.action_space,
        obs_space=env.observation_space,
        env_type=env_type,
        **all_kwargs,
    )
