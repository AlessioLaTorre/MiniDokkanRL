import os
import csv
import numpy as np

from env.mini_dokkan_env import MiniDokkanEnv
from agents.dqn_agent import DQNAgent


def evaluate_agent(agent, num_episodes=100, seed=10_000):
    """
    Evaluate the agent without exploration.

    Returns average return, win rate and average episode length.
    """
    returns = []
    wins = []
    lengths = []

    for episode in range(num_episodes):
        env = MiniDokkanEnv(render_mode=None, seed=seed + episode)
        obs, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False

        episode_return = 0.0
        episode_length = 0
        won = False

        while not terminated and not truncated:
            action = agent.act(obs, training=False)

            obs, reward, terminated, truncated, info = env.step(action)

            episode_return += reward
            episode_length += 1

            if info.get("all_phases_cleared", False):
                won = True

        returns.append(episode_return)
        wins.append(float(won))
        lengths.append(episode_length)

    return {
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "win_rate": float(np.mean(wins)),
        "avg_length": float(np.mean(lengths)),
    }


def save_training_row(csv_path, row):
    """
    Append one row to the training CSV.
    Creates the file and header if it does not exist.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def train_minimal_dqn(
    num_episodes=10_000,
    seed=42,
    save_path="models/minimal_dqn_latest.keras",
    best_save_path="models/minimal_dqn_best.keras",
    csv_path="results/minimal_dqn_training.csv",
    eval_every=500,
    eval_episodes=100,
):
    """
    Train minimal DQN and save training/evaluation metrics to CSV.

    This version is useful for fair comparison with DQN Replay:
    same number of training episodes, same environment, same evaluation protocol.
    """
    env = MiniDokkanEnv(render_mode=None, seed=seed)

    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    agent = DQNAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
    )

    episode_returns = []
    win_history = []
    loss_history = []
    length_history = []

    best_eval_return = -float("inf")
    best_eval_win_rate = 0.0

    # Avoid appending to an old file by mistake.
    if os.path.exists(csv_path):
        os.remove(csv_path)

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False

        episode_return = 0.0
        episode_losses = []
        episode_length = 0
        won = False

        while not terminated and not truncated:
            action = agent.act(obs, training=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            loss = agent.update(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )

            obs = next_obs
            episode_return += reward
            episode_length += 1

            if loss is not None:
                episode_losses.append(loss)

            if info.get("all_phases_cleared", False):
                won = True

        agent.decay_epsilon()

        mean_loss = float(np.mean(episode_losses)) if episode_losses else 0.0

        episode_returns.append(float(episode_return))
        win_history.append(float(won))
        loss_history.append(mean_loss)
        length_history.append(int(episode_length))

        if episode % 50 == 0:
            avg_return = float(np.mean(episode_returns[-50:]))
            avg_win_rate = float(np.mean(win_history[-50:]))
            avg_loss = float(np.mean(loss_history[-50:]))
            avg_length = float(np.mean(length_history[-50:]))

            row = {
                "episode": episode,
                "train_return": avg_return,
                "train_win_rate": avg_win_rate,
                "train_length": avg_length,
                "train_loss": avg_loss,
                "epsilon": float(agent.epsilon),
                "eval_return": "",
                "eval_std_return": "",
                "eval_win_rate": "",
                "eval_length": "",
                "is_best": 0,
            }

            print(
                f"Episode {episode:5d} | "
                f"Train Return: {avg_return:7.3f} | "
                f"Train Win: {avg_win_rate * 100:5.1f}% | "
                f"Length: {avg_length:5.2f} | "
                f"Loss: {avg_loss:8.4f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

            save_training_row(csv_path, row)

        if episode % eval_every == 0:
            eval_results = evaluate_agent(
                agent=agent,
                num_episodes=eval_episodes,
                seed=10_000,
            )

            is_best = 0

            if eval_results["avg_return"] > best_eval_return:
                best_eval_return = eval_results["avg_return"]
                best_eval_win_rate = eval_results["win_rate"]
                is_best = 1

                os.makedirs(os.path.dirname(best_save_path), exist_ok=True)
                agent.save(best_save_path)

            eval_row = {
                "episode": episode,
                "train_return": "",
                "train_win_rate": "",
                "train_length": "",
                "train_loss": "",
                "epsilon": float(agent.epsilon),
                "eval_return": eval_results["avg_return"],
                "eval_std_return": eval_results["std_return"],
                "eval_win_rate": eval_results["win_rate"],
                "eval_length": eval_results["avg_length"],
                "is_best": is_best,
            }

            print(
                f"[EVAL] Episode {episode:5d} | "
                f"Return: {eval_results['avg_return']:7.3f} ± "
                f"{eval_results['std_return']:.3f} | "
                f"Win: {eval_results['win_rate'] * 100:5.1f}% | "
                f"Length: {eval_results['avg_length']:5.2f}"
            )

            if is_best:
                print(
                    f"New best minimal DQN saved to: {best_save_path} | "
                    f"Return: {best_eval_return:.3f} | "
                    f"Win: {best_eval_win_rate * 100:.1f}%"
                )

            save_training_row(csv_path, eval_row)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)

    print("\nTraining completed.")
    print(f"Latest model saved to: {save_path}")
    print(f"Best model saved to:   {best_save_path}")
    print(f"CSV saved to:          {csv_path}")

    return agent, {
        "episode_returns": episode_returns,
        "win_history": win_history,
        "loss_history": loss_history,
        "length_history": length_history,
        "best_eval_return": best_eval_return,
        "best_eval_win_rate": best_eval_win_rate,
    }


if __name__ == "__main__":
    train_minimal_dqn(
        num_episodes=10_000,
        eval_every=500,
        eval_episodes=100,
    )