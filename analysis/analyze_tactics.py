import argparse
from collections import Counter

import numpy as np

from env.mini_dokkan_env import MiniDokkanEnv, UNIT_NAMES, ORB_NAMES, OrbType
from evaluation.agents_registry import build_agent, available_agents


def get_type_relation(unit_type, boss_type):
    """
    Return the type relation between selected unit and boss.

    Type wheel:
    STR > PHY > INT > TEQ > AGL > STR
    """
    advantage = {
        "STR": "PHY",
        "PHY": "INT",
        "INT": "TEQ",
        "TEQ": "AGL",
        "AGL": "STR",
    }

    if advantage[unit_type] == boss_type:
        return "ADVANTAGE"

    if advantage[boss_type] == unit_type:
        return "DISADVANTAGE"

    return "NEUTRAL"


def classify_action_style(info):
    """
    Rough tactical interpretation of the action.

    Aggressive:
        action mainly gives damage bonus.

    Defensive:
        action mainly gives healing, defense or boss attack reduction.

    Balanced:
        neither component clearly dominates.
    """
    effects = info["orb_effects"]

    offensive_score = effects.get("damage_bonus", 0.0)

    defensive_score = (
        effects.get("defense_bonus", 0.0)
        + effects.get("boss_attack_reduction", 0.0)
        + effects.get("dodge_bonus", 0.0)
        + effects.get("heal_amount", 0.0) / 100.0
    )

    if offensive_score > defensive_score:
        return "AGGRESSIVE"

    if defensive_score > offensive_score:
        return "DEFENSIVE"

    return "BALANCED"


def percentage(counter, key):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return 100.0 * counter[key] / total


def analyze_agent_tactics(
    agent_name,
    num_episodes=200,
    seed=20_000,
    board_size=4,
):
    """
    Evaluate tactical tendencies of one agent over many episodes.

    For DQN Q-map and reinforce_map:
        - use structured observations;
        - allow variable board sizes.

    For the other agents:
        - use the original flat observation format.
    """

    qmap_agents = {"dqn_qmap", "reinforce_map"}
    obs_mode = "dict" if agent_name in qmap_agents else "flat"

    base_env = MiniDokkanEnv(
        seed=seed,
        board_size=board_size,
        obs_mode=obs_mode,
    )

    agent = build_agent(agent_name, base_env)

    unit_usage = Counter()
    selected_orb_types = Counter()
    collected_orb_types = Counter()
    type_relations = Counter()
    action_styles = Counter()
    phase_clear_counts = Counter()

    total_returns = []
    wins = 0
    total_steps = 0

    for episode in range(num_episodes):
        env = MiniDokkanEnv(
            seed=seed + episode,
            board_size=board_size,
            obs_mode=obs_mode,
        )

        obs, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False
        episode_return = 0.0

        while not terminated and not truncated:
            action = agent.act(
                obs=obs,
                info=info,
                env=env,
                training=False,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            episode_return += reward
            total_steps += 1

            # --------------------------------------------------------
            # Unit chosen.
            # --------------------------------------------------------
            unit_usage[info["unit_name"]] += 1

            # --------------------------------------------------------
            # Type of the selected orb.
            # This is dynamic because board_size can now change.
            # --------------------------------------------------------
            row, col = info["orb_position"]
            selected_orb_id = int(info["board_before"][row, col])
            selected_orb_name = ORB_NAMES[OrbType(selected_orb_id)]
            selected_orb_types[selected_orb_name] += 1

            # --------------------------------------------------------
            # All collected orb colors.
            # --------------------------------------------------------
            for orb_name, count in info["orb_counts"].items():
                collected_orb_types[orb_name] += count

            # --------------------------------------------------------
            # Type relation unit vs boss.
            # --------------------------------------------------------
            unit_type = info["unit_type"]
            boss_type = UNIT_NAMES[info["boss_type_before"]]
            relation = get_type_relation(unit_type, boss_type)
            type_relations[relation] += 1

            # --------------------------------------------------------
            # Aggressive / defensive / balanced.
            # --------------------------------------------------------
            style = classify_action_style(info)
            action_styles[style] += 1

            # --------------------------------------------------------
            # Cleared phases.
            # --------------------------------------------------------
            if info["phase_cleared"]:
                phase_clear_counts[info["phase_before"] + 1] += 1

            if info.get("all_phases_cleared", False):
                wins += 1

        total_returns.append(episode_return)

    print("=" * 70)
    print(f"TACTICAL ANALYSIS - {agent_name}")
    print("=" * 70)
    print(f"Board size:          {board_size}x{board_size}")
    print(f"Observation mode:    {obs_mode}")
    print(f"Episodes:            {num_episodes}")
    print(f"Total actions:       {total_steps}")
    print(f"Average return:      {np.mean(total_returns):.3f}")
    print(f"Return std:          {np.std(total_returns):.3f}")
    print(f"Win rate:            {100.0 * wins / num_episodes:.1f}%")
    print()

    print("Unit usage:")
    for key, value in unit_usage.most_common():
        print(f"  {key:20s}: {value:5d} ({percentage(unit_usage, key):5.1f}%)")
    print()

    print("Selected orb type:")
    for key, value in selected_orb_types.most_common():
        print(f"  {key:5s}: {value:5d} ({percentage(selected_orb_types, key):5.1f}%)")
    print()

    print("Collected orb totals:")
    for key, value in collected_orb_types.most_common():
        print(f"  {key:5s}: {value:5d} ({percentage(collected_orb_types, key):5.1f}%)")
    print()

    print("Type relation of chosen unit:")
    for key, value in type_relations.most_common():
        print(f"  {key:15s}: {value:5d} ({percentage(type_relations, key):5.1f}%)")
    print()

    print("Action style:")
    for key, value in action_styles.most_common():
        print(f"  {key:12s}: {value:5d} ({percentage(action_styles, key):5.1f}%)")
    print()

    print("Phase clears:")
    for phase in [1, 2, 3]:
        print(f"  Phase {phase}: {phase_clear_counts[phase]} clears")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=available_agents(),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=4,
        help="Square board size. Example: 4 means 4x4, 5 means 5x5.",
    )

    args = parser.parse_args()

    analyze_agent_tactics(
        agent_name=args.agent,
        num_episodes=args.episodes,
        seed=args.seed,
        board_size=args.board_size,
    )


if __name__ == "__main__":
    main()