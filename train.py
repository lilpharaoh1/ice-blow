import os
import argparse
import pygame
import numpy as np

from run_utils import setup_run_list
from agents.registry import make_agent
from render import IceBlowRenderer

from envs.ice_blow_gridworld import IceBlowGridworldEnv
from envs.ice_blow_discrete import IceBlowDiscreteEnv
from envs.ice_blow_continuous import IceBlowContinuousEnv

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

    env_kwargs = {
        k: v for k, v in run.env.items()
    }

    if run.env["type"] == "gridworld":
        return IceBlowGridworldEnv(
            # grid_size=run.env["grid_size"],
            **common_kwargs,
            **env_kwargs
        )
    elif run.env["type"] == "discrete":
        return IceBlowDiscreteEnv(
            # grid_size=run.env["grid_size"],
            **common_kwargs,
        )
    elif run.env["type"] == "continuous":
        return IceBlowContinuousEnv(
            # grid_size=run.env["grid_size"],
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown env type: {run.env['type']}")


# ---------------------------------------------------------------------
# Single run (train or eval)
# ---------------------------------------------------------------------

def run_single(
    run,
    render=False,
    fps=30,
    load_path=None,
    save_dir="checkpoints",
):
    print(f"\n=== Running {run.run_name} ===")

    np.random.seed(run.seed)

    env = make_env(run)
    agent = make_agent(run, env)

    if load_path is not None:
        print(f"Loading agent weights from {load_path}")
        agent.load(load_path)


    logger = Logger(run.output_path)

    renderer = None
    if render:
        pygame.init()
        renderer = IceBlowRenderer(fps=fps)

    obs, _ = env.reset(seed=run.seed)
    global_step = 0 

    episode_reward = 0.0
    episode_length = 0
    episode_idx = 0

    avg_reward = 0
    avg_decay = 0.1
    
    print_every = 10000 if run.run["steps"] > 10000 else 1000
    max_step = run.run["steps"]
    for step in range(max_step):
        if step % print_every == 0:
            print(f"Step {step} / {max_step}")
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

        action = agent.act(obs, explore=True)

        if action is None:
            if render:
                renderer.step()
            continue

        next_obs, reward, done, _, info = env.step(action)

        agent.store(obs, action, reward, next_obs, done)

        episode_reward += reward
        episode_length += 1

        global_step += 1   # ← ADD THIS

        # try:
        if getattr(agent, "batch_size", False) and global_step > agent.batch_size * 5:
            metrics = agent.update()

            # Log training metrics every 1000 steps
            if metrics and global_step % 1000 == 0:
                logger.log_training_step(global_step, metrics)
                print(f"Step {global_step}: Loss={metrics['loss']:.4f}, Reward={avg_reward:.4f}, Q-value={metrics['q_value_mean']:.4f}, Epsilon={metrics['epsilon']:.4f}")
        # except:
        #     pass # not trainable agent


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

        if done:
            logger.log_episode({
                "episode": episode_idx,
                "reward": episode_reward,
                "length": episode_length,
                "seed": run.seed,
            })

            obs, _ = env.reset()
            avg_reward = episode_reward * avg_decay + avg_reward * (1-avg_decay)

            episode_reward = 0.0
            episode_length = 0
            episode_idx += 1

    if callable(getattr(agent, "save", None)):
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, f"{run.run_name}.pt")
        agent.save(ckpt_path)
        print(f"Saved model to {ckpt_path}")


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
    parser.add_argument("--load", type=str, default=None,
                    help="Path to agent checkpoint (.pt) to load")
    parser.add_argument("--save-dir", type=str, default="checkpoints")


    args = parser.parse_args()

    run_list = setup_run_list(args.runfile)

    for run in run_list:
        run_single(
            run,
            render=args.render,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
