import os
import json
import numpy as np

from env.mini_dokkan_env import MiniDokkanEnv
from agents import ReinforceBaselineAgent


def evaluate_policy(
    agent,
    num_episodes=100,
    seed=42,
):
    """
    Evaluate the current policy without exploration.

    The same fixed seed range is used at every evaluation checkpoint,
    so improvements over time are meaningfully comparable.
    """
    episode_returns = []
    episode_lengths = []
    wins = []
    phases_reached = []
    damages_dealt = []
    damages_taken = []

    for episode in range(num_episodes):
        env = MiniDokkanEnv(seed=seed + episode)
        obs, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False

        ep_return = 0.0
        ep_length = 0
        ep_damage_dealt = 0.0
        ep_damage_taken = 0.0
        max_phase_reached = 0
        won = False

        while not terminated and not truncated:
            action = agent.act(
                obs=obs,
                info=info,
                env=env,
                training=False,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            ep_return += reward
            ep_length += 1

            ep_damage_dealt += info.get("damage_dealt", 0.0)
            ep_damage_taken += info.get("damage_taken", 0.0)
            max_phase_reached = max(
                max_phase_reached,
                info.get("phase_after", info.get("phase", 0)),
            )

            if info.get("all_phases_cleared", False):
                won = True

        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        wins.append(float(won))
        phases_reached.append(max_phase_reached)
        damages_dealt.append(ep_damage_dealt)
        damages_taken.append(ep_damage_taken)

    return {
        "avg_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "win_rate": float(np.mean(wins)),
        "avg_length": float(np.mean(episode_lengths)),
        "avg_phase_reached": float(np.mean(phases_reached)),
        "avg_damage_dealt": float(np.mean(damages_dealt)),
        "avg_damage_taken": float(np.mean(damages_taken)),
    }


def train_reinforce_baseline(
    num_episodes=5000,
    seed=42,
    eval_every=100,
    eval_episodes=100,
    episodes_per_update=10,
    save_prefix="models/reinforce_baseline10k",
    metrics_path="results/reinforce_baseline_metrics.json",
):
    """
    Train REINFORCE with learned baseline.

    Besides normal training statistics, the function performs a stable
    evaluation every `eval_every` episodes on fixed evaluation seeds.
    """
    env = MiniDokkanEnv(seed=seed)

    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    agent = ReinforceBaselineAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        gamma=0.99,
        policy_learning_rate=5e-4,
        value_learning_rate=1e-3,
        entropy_coef=0.001,
    )

    history = {
        "train_episode_return": [],
        "train_episode_length": [],
        "train_win": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "eval": [],
        "avg_raw_advantage": [],
        "std_raw_advantage": [],
        "max_action_prob": []
    }

    batch_states = []
    batch_actions = []
    batch_returns = []

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=seed + episode)

        states = []
        actions = []
        rewards = []

        terminated = False
        truncated = False

        episode_return = 0.0
        episode_length = 0
        won = False

        while not terminated and not truncated:
            action = agent.act(
                obs=obs,
                info=info,
                env=env,
                training=True,
            )

            next_obs, reward, terminated, truncated, info = env.step(action)

            states.append(obs)
            actions.append(action)
            rewards.append(reward)

            obs = next_obs
            episode_return += reward
            episode_length += 1

            if info.get("all_phases_cleared", False):
                won = True

        returns = agent.compute_returns(rewards)

        batch_states.extend(states)
        batch_actions.extend(actions)
        batch_returns.extend(returns)

        update_info = None

        if episode % episodes_per_update == 0:
            update_info = agent.update_from_episode(
                states=batch_states,
                actions=batch_actions,
                returns=batch_returns,
            )

            batch_states = []
            batch_actions = []
            batch_returns = []
        if update_info is not None:
            history["train_episode_return"].append(float(episode_return))
            history["train_episode_length"].append(int(episode_length))
            history["train_win"].append(float(won))
            history["policy_loss"].append(update_info["policy_loss"])
            history["value_loss"].append(update_info["value_loss"])
            history["entropy"].append(update_info["entropy"])
            history["avg_raw_advantage"].append(update_info["avg_raw_advantage"])
            history["std_raw_advantage"].append(update_info["std_raw_advantage"])
            history["max_action_prob"].append(update_info["max_action_prob"])

        if episode % 100 == 0:
            avg_return = np.mean(history["train_episode_return"][-100:])
            avg_win_rate = np.mean(history["train_win"][-100:])
            avg_length = np.mean(history["train_episode_length"][-100:])
            avg_policy_loss = np.mean(history["policy_loss"][-10:]) if history["policy_loss"] else 0.0
            avg_value_loss = np.mean(history["value_loss"][-10:]) if history["value_loss"] else 0.0
            avg_entropy = np.mean(history["entropy"][-10:]) if history["entropy"] else 0.0
            avg_raw_advantage = np.mean(history["avg_raw_advantage"][-10:]) if history["avg_raw_advantage"] else 0.0
            avg_std_raw_advantage = np.mean(history["std_raw_advantage"][-10:]) if history["std_raw_advantage"] else 0.0
            avg_max_action_prob = np.mean(history["max_action_prob"][-10:]) if history["max_action_prob"] else 0.0


            print(
                f"Episode {episode:5d} | "
                f"Train Return: {avg_return:7.3f} | "
                f"Train Win: {avg_win_rate * 100:5.1f}% | "
                f"Length: {avg_length:5.2f} | "
                f"P-Loss: {avg_policy_loss:8.4f} | "
                f"V-Loss: {avg_value_loss:8.4f} | "
                f"Entropy: {avg_entropy:7.4f} | "
                f"avg_raw_advantage: {avg_raw_advantage:7.4f} | "
                f"std_raw_advantage: {avg_std_raw_advantage:7.4f} | "
                f"max_action_prob: {avg_max_action_prob:7.4f}"
            )

        if episode % eval_every == 0:
            eval_results = evaluate_policy(
                agent=agent,
                num_episodes=eval_episodes,
                seed=10_000,
            )

            eval_record = {
                "episode": int(episode),
                **eval_results,
            }
            history["eval"].append(eval_record)

            print(
                f"[EVAL] Episode {episode:5d} | "
                f"Return: {eval_results['avg_return']:7.3f} ± "
                f"{eval_results['std_return']:.3f} | "
                f"Win: {eval_results['win_rate'] * 100:5.1f}% | "
                f"Phase: {eval_results['avg_phase_reached']:.2f} | "
                f"Length: {eval_results['avg_length']:.2f}"
            )

    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    agent.save(save_prefix)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed.")
    print(f"Models saved with prefix: {save_prefix}")
    print(f"Metrics saved to: {metrics_path}")

    return agent, history


if __name__ == "__main__":
    train_reinforce_baseline(
        num_episodes=10000,
        eval_every=500,
        eval_episodes=100,
        episodes_per_update=10
    )