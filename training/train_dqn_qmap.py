import os
import numpy as np

from env.mini_dokkan_env import MiniDokkanEnv
from agents.dqn_qmap_agent import DQNQMapAgent


def train_dqn_qmap(
    num_episodes=1500,
    seed=42,
    board_size=4,
    save_path="models/dqn_qmap_10k.keras",
):
    """
    Train DQN Q-map with replay buffer and target network.

    This training loop intentionally follows the same structure used by
    DQNAgentV2, so that logs and learning curves are directly comparable.

    Differences from DQN v2:
    - the environment uses structured observations;
    - the board is processed by a CNN;
    - the network outputs a Q-map with shape:
          board_size x board_size x num_units
      instead of a fixed flat vector.
    """
    env = MiniDokkanEnv(
        render_mode=None,
        seed=seed,
        board_size=board_size,
        obs_mode="dict",
    )

    # ------------------------------------------------------------
    # Infer observation dimensions from one reset.
    # The board size can change, but global_dim and unit_dim are
    # independent from the board size.
    # ------------------------------------------------------------
    obs, info = env.reset(seed=seed)

    global_dim = int(obs["global"].shape[0])
    unit_dim = int(obs["units"].shape[1])
    num_orb_types = int(obs["board"].shape[-1])
    num_units = int(obs["units"].shape[0])

    agent = DQNQMapAgent(
        global_dim=global_dim,
        unit_dim=unit_dim,
        num_orb_types=num_orb_types,
        num_units=num_units,
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

            update_info = agent.update()

            if update_info is not None:
                episode_losses.append(update_info["loss"])

            obs = next_obs
            episode_return += reward
            episode_length += 1

            if info.get("all_phases_cleared", False):
                won = True

        agent.decay_epsilon()

        episode_returns.append(float(episode_return))
        episode_lengths.append(int(episode_length))
        win_history.append(float(won))
        loss_history.append(
            float(np.mean(episode_losses)) if episode_losses else 0.0
        )

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
    train_dqn_qmap(
        num_episodes=10000,
        seed=42,
        board_size=4,
        save_path="models/dqn_qmap_10k.keras",
    )