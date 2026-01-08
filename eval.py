import argparse
import pygame
import numpy as np

from run_utils import setup_run_list
from agents.registry import make_agent
from render import IceBlowRenderer
from train import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runfile", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (optional for non-trainable agents)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()

    run = setup_run_list(args.runfile)[0]

    env = make_env(run)
    agent = make_agent(run, env)

    # Only load checkpoint if provided and agent supports it
    if args.checkpoint is not None:
        if hasattr(agent, 'load'):
            agent.load(args.checkpoint)
        else:
            print(f"Warning: Agent {type(agent).__name__} has no load method, ignoring checkpoint")
    elif hasattr(agent, 'load'):
        # Trainable agent without checkpoint - warn user
        print(f"Note: No checkpoint provided for trainable agent {type(agent).__name__}")

    renderer = None
    if args.render:
        pygame.init()
        renderer = IceBlowRenderer(fps=args.fps)

    returns = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=run.seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            if args.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

            action = agent.act(obs, explore=False)
            obs, reward, done, _, _ = env.step(action)
            ep_return += reward

            if args.render:
                renderer.render(
                    agent_pos=obs["agent"] if isinstance(obs, dict) else obs[:2],
                    goal_pos=obs["goal"] if isinstance(obs, dict) else obs[4:6],
                    blow_phase=env.blow_phase,
                    blow_axis=env.blow_axis,
                    blow_centers=env.blow_centers,
                    blow_width=env.blow_width,
                    world_size=env.world_size,
                )
                renderer.step()

        returns.append(ep_return)
        print(f"Episode {ep}: return = {ep_return:.2f}")

    print("\nEvaluation summary")
    print(f"Mean return: {np.mean(returns):.2f}")
    print(f"Std return:  {np.std(returns):.2f}")

    if args.render:
        pygame.quit()


if __name__ == "__main__":
    main()
