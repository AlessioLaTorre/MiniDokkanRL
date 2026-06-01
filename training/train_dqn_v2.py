import os
import numpy as np

from env.mini_dokkan_env import MiniDokkanEnv
from agents.dqn_agent_v2 import DQNAgentV2


def train_dqn_v2(
    num_episodes=1500,
    seed=42,
    save_path="models/dqn_v2_30k.keras",
):
    """
    Train DQN v2 with replay buffer and target network.
    """
    env = MiniDokkanEnv(render_mode=None, seed=seed)

    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    agent = DQNAgentV2(
        obs_dim=obs_dim,
        action_dim=action_dim,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_size=50_000,
        batch_size=64,
        target_update_freq=250,
    )

    episode_returns = []
    episode_lengths = []
    win_history = []
    loss_history = []

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False

        episode_return = 0.0
        episode_length = 0
        episode_losses = []
        won = False

        while not terminated and not truncated:
            action = agent.act(
                obs=obs,
                info=info,
                env=env,
                training=True,
            )

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.remember(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done,
            )

            loss = agent.update()

            if loss is not None:
                episode_losses.append(loss)

            obs = next_obs
            episode_return += reward
            episode_length += 1

            if info.get("all_phases_cleared", False):
                won = True

        agent.decay_epsilon()

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        win_history.append(float(won))
        loss_history.append(np.mean(episode_losses) if episode_losses else 0.0)

        if episode % 50 == 0:
            avg_return = np.mean(episode_returns[-50:])
            avg_win_rate = np.mean(win_history[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            avg_loss = np.mean(loss_history[-50:])

            print(
                f"Episode {episode:4d} | "
                f"Return: {avg_return:7.3f} | "
                f"Win rate: {avg_win_rate * 100:5.1f}% | "
                f"Length: {avg_length:5.2f} | "
                f"Loss: {avg_loss:8.4f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Buffer: {len(agent.replay_buffer)}"
            )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)

    print(f"\nTraining completed. Model saved to: {save_path}")

    return agent, {
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "win_history": win_history,
        "loss_history": loss_history,
    }


if __name__ == "__main__":
    train_dqn_v2(num_episodes=10000)
