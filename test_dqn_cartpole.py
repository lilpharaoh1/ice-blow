import gymnasium as gym
import numpy as np
import torch

from agents.dqn_agent import DQNAgent


def main():
    env = gym.make("CartPole-v1")

    agent = DQNAgent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=100_000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=50_000,
        target_update_freq=1000,
    )

    obs, _ = env.reset(seed=0)

    episode_reward = 0
    episode = 0
    rewards = []

    for step in range(200_000):
        action = agent.act(obs, explore=True)

        next_obs_raw, reward, terminated, truncated, _ = env.step(action)
        next_obs = {"state": next_obs_raw}
        done = terminated or truncated

        agent.store(obs, action, reward, next_obs, done)

        if step > 1000:
            agent.update()

        obs = next_obs
        episode_reward += reward

        if done:
            rewards.append(episode_reward)

            if episode % 10 == 0:
                avg = np.mean(rewards[-10:])
                print(
                    f"Episode {episode:4d} | "
                    f"Reward {episode_reward:4.0f} | "
                    f"Avg(10) {avg:6.1f} | "
                    f"Eps {agent.epsilon:.3f}"
                )

            obs_raw, _ = env.reset()
            obs = {"state": obs_raw}
            episode_reward = 0
            episode += 1

        # Stop if solved
        if len(rewards) > 100 and np.mean(rewards[-100:]) > 475:
            print("🎉 CartPole solved!")
            break

    env.close()


if __name__ == "__main__":
    main()
