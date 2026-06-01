import argparse
import os
import shutil

from env.mini_dokkan_env import MiniDokkanEnv, UNIT_NAMES
from evaluation.agents_registry import build_agent, available_agents
from utils.visualization import (
    render_initial_frame,
    render_step_frame,
    make_gif,
)


def run_visual_episode(agent_name, seed=42):
    """
    Run one episode with the selected agent and save:
    - one PNG frame per turn;
    - one final GIF.
    """
    env = MiniDokkanEnv(render_mode=None, seed=seed)
    agent = build_agent(agent_name, env)

    output_dir = f"results/visual_{agent_name}_frames"
    gif_path = f"results/visual_{agent_name}.gif"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False
    total_reward = 0.0
    frame_paths = []

    # ------------------------------------------------------------
    # Frame 0: initial state.
    # ------------------------------------------------------------
    initial_frame_path = os.path.join(output_dir, "frame_000.png")

    render_initial_frame(
        board=env.board.copy(),
        player_hp=env.player_hp,
        player_max_hp=env.player_max_hp,
        boss_hp=env.current_boss_hp,
        boss_max_hp=env.current_boss["max_hp"],
        phase=env.current_phase + 1,
        total_phases=len(env.bosses),
        boss_name=env.current_boss["name"],
        boss_type=UNIT_NAMES[env.current_boss["type"]],
        next_boss_attacks=env.next_boss_attacks,
        save_path=initial_frame_path,
    )

    frame_paths.append(initial_frame_path)

    # ------------------------------------------------------------
    # Episode loop.
    # ------------------------------------------------------------
    while not terminated and not truncated:
        action = agent.act(
            obs=obs,
            info=info,
            env=env,
            training=False,
        )

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        frame_path = os.path.join(
            output_dir,
            f"frame_{info['turn']:03d}.png",
        )

        render_step_frame(
            board=info["board_before"],
            player_hp=info["player_hp"],
            player_max_hp=env.player_max_hp,

            boss_hp_before=info["boss_hp_before"],
            boss_hp_after=info["boss_hp_after"],
            boss_max_hp=info["boss_max_hp_before"],

            phase=info["phase_before"] + 1,
            total_phases=len(env.bosses),
            turn=info["turn"],

            selected_orb=info["orb_position"],
            collected_positions=info["collected_positions"],

            damage_dealt=info["damage_dealt"],
            damage_taken=info["damage_taken"],
            healed=info["healed"],
            reward=info["reward"],

            unit_name=info["unit_name"],
            unit_type=info["unit_type"],

            boss_name=info["boss_name_before"],
            boss_type=UNIT_NAMES[info["boss_type_before"]],
            next_boss_attacks_before=info["next_boss_attacks_before"],

            orb_counts=info["orb_counts"],
            orb_effects=info["orb_effects"],
            dodged_attacks=info["dodged_attacks"],
            phase_cleared=info["phase_cleared"],
            boss_attack_reduction=info["boss_attack_reduction"],

            save_path=frame_path,
        )

        frame_paths.append(frame_path)

    make_gif(frame_paths, gif_path, duration=1000)

    print("=" * 60)
    print(f"Agent: {agent_name}")
    print(f"Seed: {seed}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"GIF saved to: {gif_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=available_agents(),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    run_visual_episode(
        agent_name=args.agent,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()