import os
import numpy as np

from env.mini_dokkan_env import MiniDokkanEnv
from agents.reinforce_map_agent import ReinforceMapAgent


def train_reinforce_map(
    num_episodes=1500,
    seed=42,
    board_size=4,
    batch_size=64,
    save_path="models/reinforce_map_10k",
):
    """
    Train REINFORCE Q-map with learned value baseline.

    This training loop follows the same logging style used by the previous
    agents, so curves are easier to compare.

    Main difference from old REINFORCE:
    - updates are done on batches of transitions;
    - batch_size is set to 64 by default, like DQN Replay;
    - the policy outputs a dynamic map instead of a fixed flat vector.

    Important:
        This is still on-policy REINFORCE.
        It does not use replay memory like DQN.
    """
    env = MiniDokkanEnv(
        render_mode=None,
        seed=seed,
        board_size=board_size,
        obs_mode="dict",
    )

    obs, info = env.reset(seed=seed)

    global_dim = int(obs["global"].shape[0])
    unit_dim = int(obs["units"].shape[1])
    num_orb_types = int(obs["board"].shape[-1])
    num_units = int(obs["units"].shape[0])

    agent = ReinforceMapAgent(
        global_dim=global_dim,
        unit_dim=unit_dim,
        num_orb_types=num_orb_types,
        num_units=num_units,
        gamma=0.99,
        policy_learning_rate=1e-4,
        value_learning_rate=5e-4,
        entropy_coef=0.001,
        batch_size=batch_size,
        normalize_advantages=True,
    )

    episode_returns = []
    episode_lengths = []
    win_history = []

    policy_loss_history = []
    value_loss_history = []
    entropy_history = []
    max_action_prob_history = []

    # ------------------------------------------------------------
    # Pending on-policy transitions.
    # These are updated in batches of batch_size transitions.
    # ------------------------------------------------------------
    pending_states = []
    pending_actions = []
    pending_returns = []

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False

        states = []
        actions = []
        rewards = []

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
            rewards.append(float(reward))

            obs = next_obs
            episode_return += float(reward)
            episode_length += 1

            if info.get("all_phases_cleared", False):
                won = True

        # ------------------------------------------------------------
        # Compute Monte Carlo returns only after the full episode ended.
        # ------------------------------------------------------------
        returns = agent.compute_returns(rewards)

        pending_states.extend(states)
        pending_actions.extend(actions)
        pending_returns.extend(list(returns))

        episode_policy_losses = []
        episode_value_losses = []
        episode_entropies = []
        episode_max_probs = []

        # ------------------------------------------------------------
        # Update using batches of on-policy transitions.
        # Batch size is the same default used by DQN Replay: 64.
        # ------------------------------------------------------------
        while len(pending_states) >= batch_size:
            batch_states = pending_states[:batch_size]
            batch_actions = pending_actions[:batch_size]
            batch_returns = pending_returns[:batch_size]

            update_info = agent.update_from_batch(
                states=batch_states,
                actions=batch_actions,
                returns=batch_returns,
            )

            episode_policy_losses.append(update_info["policy_loss"])
            episode_value_losses.append(update_info["value_loss"])
            episode_entropies.append(update_info["entropy"])
            episode_max_probs.append(update_info["max_action_prob"])

            pending_states = pending_states[batch_size:]
            pending_actions = pending_actions[batch_size:]
            pending_returns = pending_returns[batch_size:]

        episode_returns.append(float(episode_return))
        episode_lengths.append(int(episode_length))
        win_history.append(float(won))

        policy_loss_history.append(
            float(np.mean(episode_policy_losses))
            if episode_policy_losses
            else 0.0
        )

        value_loss_history.append(
            float(np.mean(episode_value_losses))
            if episode_value_losses
            else 0.0
        )

        entropy_history.append(
            float(np.mean(episode_entropies))
            if episode_entropies
            else 0.0
        )

        max_action_prob_history.append(
            float(np.mean(episode_max_probs))
            if episode_max_probs
            else 0.0
        )

        if episode % 50 == 0:
            avg_return = np.mean(episode_returns[-50:])
            avg_win_rate = np.mean(win_history[-50:])
            avg_length = np.mean(episode_lengths[-50:])

            avg_policy_loss = np.mean(policy_loss_history[-50:])
            avg_value_loss = np.mean(value_loss_history[-50:])
            avg_entropy = np.mean(entropy_history[-50:])
            avg_max_prob = np.mean(max_action_prob_history[-50:])

            print(
                f"Episode {episode:4d} | "
                f"Return: {avg_return:7.3f} | "
                f"Win rate: {avg_win_rate * 100:5.1f}% | "
                f"Length: {avg_length:5.2f} | "
                f"Policy loss: {avg_policy_loss:8.4f} | "
                f"Value loss: {avg_value_loss:8.4f} | "
                f"Entropy: {avg_entropy:7.4f} | "
                f"Max prob: {avg_max_prob:6.3f} | "
                f"Pending: {len(pending_states):3d}"
            )

    # ------------------------------------------------------------
    # Optional final update with remaining transitions.
    # This uses a smaller final batch only at the end of training.
    # ------------------------------------------------------------
    if len(pending_states) > 0:
        update_info = agent.update_from_batch(
            states=pending_states,
            actions=pending_actions,
            returns=pending_returns,
        )

        print(
            "\nFinal partial update | "
            f"Policy loss: {update_info['policy_loss']:.4f} | "
            f"Value loss: {update_info['value_loss']:.4f} | "
            f"Entropy: {update_info['entropy']:.4f}"
        )

    save_dir = os.path.dirname(save_path)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    agent.save(save_path)

    print(f"\nTraining completed. Model saved to prefix: {save_path}")

    return agent, {
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "win_history": win_history,
        "policy_loss_history": policy_loss_history,
        "value_loss_history": value_loss_history,
        "entropy_history": entropy_history,
        "max_action_prob_history": max_action_prob_history,
    }


if __name__ == "__main__":
    train_reinforce_map(
        num_episodes=10000,
        seed=42,
        board_size=4,
        batch_size=64,
        save_path="models/reinforce_map_10k",
    )