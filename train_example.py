# train_example.py
import time
import numpy as np
import pygame

from envs.ice_blow_discrete import IceBlowDiscreteEnv
from envs.ice_blow_continuous import IceBlowContinuousEnv
from render import IceBlowRenderer


def run_env(env, steps=500):
    renderer = IceBlowRenderer()
    obs, _ = env.reset()

    for step in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close()
                return

        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)

        if isinstance(obs, dict):
            agent_pos = obs["agent"]
            goal_pos = obs["goal"]
        else:
            agent_pos = obs[0:2]
            goal_pos = obs[4:6]

        renderer.render(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            blow_phase=env.blow_phase,
            blow_axis=env.blow_axis,
            blow_coord=env.blow_coord,
            world_size=env.world_size,
        )



        if reward != 0:
            print(f"Step {step}, reward: {reward}")

        if done:
            print("Agent died. Resetting.")
            obs, _ = env.reset()
            time.sleep(0.5)

    renderer.close()


if __name__ == "__main__":
    print("Running DISCRETE environment")
    env = IceBlowDiscreteEnv(grid_size=10)
    run_env(env)

    print("Running CONTINUOUS environment")
    env = IceBlowContinuousEnv()
    run_env(env)

