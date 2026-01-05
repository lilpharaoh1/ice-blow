import argparse
import pygame
import numpy as np

from run_utils import setup_run_list
from agents.registry import make_agent
from render import IceBlowRenderer

from envs.ice_blow_discrete import IceBlowDiscreteEnv
from envs.ice_blow_continuous import IceBlowContinuousEnv

from training.trajectory import Trajectory
from training.logger import Logger


# ---------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------

def make_env(run):
    blow_cfg = run.blow

    common_kwargs = dict(
        blow_interval=blow_cfg["interval"],
        warning_duration=blow_cfg["warning_duration"],
        active_duration=blow_cfg["active_duration"],
        blow_width=blow_cfg["width"],
        num_blow_lines=blow_cfg["num_lines"],
    )

    if run.env["type"] == "discrete":
        return IceBlowDiscreteEnv(
            grid_size=run.env["grid_size"],
            **common_kwargs,
        )
    elif run.env["type"] == "continuous":
        return IceBlowContinuousEnv(
            grid_size=run.env["grid_size"],
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown env type: {run.env['type']}")


# ---------------------------------------------------------------------
# Single run (train or eval)
# ---------------------------------------------------------------------

def run_single(run, render=False, fps=30, eval_mode=False):
    print(f"\n=== Running {run.run_name} | eval={eval_mode} ===")

    np.random.seed(run.seed)

    env = make_env(run)
    agent = make_agent(run, env)

    logger = Logger(run.output_path)
    trajectory = Trajectory()

    renderer = None
    if render:
        pygame.init()
        renderer = IceBlowRenderer(fps=fps)

    obs, _ = env.reset(seed=run.seed)

    episode_reward = 0.0
    episode_length = 0
    episode_idx = 0

    max_steps = run.run["steps"]

    for step in range(max_steps):

        # --------------------------------------------------------------
        # Handle window events (only if rendering)
        # --------------------------------------------------------------
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

        # --------------------------------------------------------------
        # Agent action
        # --------------------------------------------------------------
        action = agent.act(obs)

        # Allow no-op actions (keyboard agent)
        if action is None:
            if render:
                renderer.step()
            continue

        next_obs, reward, done, _, info = env.step(action)

        trajectory.add(obs, action, reward, done)

        episode_reward += reward
        episode_length += 1

        # --------------------------------------------------------------
        # Rendering
        # --------------------------------------------------------------
        if render:
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

        obs = next_obs

        # --------------------------------------------------------------
        # Episode termination
        # --------------------------------------------------------------
        if done:
            logger.log_episode({
                "episode": episode_idx,
                "reward": episode_reward,
                "length": episode_length,
                "seed": run.seed,
                "eval": eval_mode,
            })

            if not eval_mode:
                agent.observe(trajectory)
                agent.update()

            trajectory.clear()
            agent.reset()

            obs, _ = env.reset()

            episode_reward = 0.0
            episode_length = 0
            episode_idx += 1

    logger.flush()

    if render:
        pygame.quit()


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runfile", type=str, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args()

    run_list = setup_run_list(args.runfile)

    for run in run_list:
        run_single(
            run,
            render=args.render,
            fps=args.fps,
            eval_mode=args.eval,
        )


if __name__ == "__main__":
    main()
