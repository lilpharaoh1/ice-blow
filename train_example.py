import argparse
import pygame
import numpy as np

from run_utils import setup_run_list
from envs.ice_blow_discrete import IceBlowDiscreteEnv
from envs.ice_blow_continuous import IceBlowContinuousEnv
from render import IceBlowRenderer
from agents.registry import make_agent


def make_env(run):
    blow = run.blow

    common_kwargs = dict(
        blow_interval=blow["interval"],
        warning_duration=blow["warning_duration"],
        active_duration=blow["active_duration"],
        blow_width=blow["width"],
        num_blow_lines=blow["num_lines"],
    )

    if run.env["type"] == "discrete":
        return IceBlowDiscreteEnv(
            grid_size=run.env["grid_size"],
            **common_kwargs,
        )
    else:
        return IceBlowContinuousEnv(**common_kwargs)


def run_single(run):
    print(f"=== Running {run.run_name} ===")

    np.random.seed(run.seed)

    env = make_env(run)
    agent = make_agent(run.agent, env)

    obs, _ = env.reset(seed=run.seed)

    renderer = IceBlowRenderer()
    steps = run.run["steps"]

    for _ in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close()
                return

        # Placeholder agent logic
        action = agent.act(obs)
        if action is None:
            continue

        obs, reward, done, _, _ = env.step(action)

        if isinstance(obs, dict):
            agent_pos = obs["agent"]
            goal_pos = obs["goal"]
        else:
            agent_pos = obs[:2]
            goal_pos = obs[4:6]

        renderer.render(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            blow_phase=env.blow_phase,
            blow_axis=env.blow_axis,
            blow_centers=env.blow_centers,
            blow_width=env.blow_width,
            world_size=env.world_size,
        )

        if done:
            obs, _ = env.reset()
            agent.reset()

    renderer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runfile", type=str, required=True)
    args = parser.parse_args()

    run_list = setup_run_list(args.runfile)

    for run in run_list:
        run_single(run)


if __name__ == "__main__":
    main()
