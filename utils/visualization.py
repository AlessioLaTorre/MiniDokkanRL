import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


ORB_COLORS = {
    "STR": "#e74c3c",   # red
    "AGL": "#3498db",   # blue
    "TEQ": "#2ecc71",   # green
    "INT": "#9b59b6",   # purple
    "PHY": "#f1c40f",   # yellow
    "RNB": "#f2f2f2",   # white rainbow base
}

ORB_TEXT_COLORS = {
    "STR": "white",
    "AGL": "white",
    "TEQ": "black",
    "INT": "white",
    "PHY": "black",
    "RNB": "black",
}

ORB_LABELS = {
    0: "STR",
    1: "AGL",
    2: "TEQ",
    3: "INT",
    4: "PHY",
    5: "RNB",
}

def _get_type_relation(unit_type, boss_type):
    """
    Return a readable description of type relation between selected unit and boss.

    The simplified type wheel is:
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
        return "ADVANTAGE (x1.5)"
    if advantage[boss_type] == unit_type:
        return "DISADVANTAGE (x0.75)"
    return "NEUTRAL (x1.0)"


def _build_legend_text():
    """
    Build a compact legend text shown in the bottom-right area.
    """
    return (
        "Legend\n"
        "------\n"
        "Advantage:\n"
        "AGL > STR > PHY > INT > TEQ > AGL ...\n"
        "------\n"
        "Orb colors:\n"
        "  STR = damage bonus\n"
        "  AGL = dodge bonus\n"
        "  TEQ = defense bonus\n"
        "  INT = boss ATK debuff\n"
        "  PHY = heal\n"
        "  RNB = small general bonus\n\n"
        "Statistics:\n"
        "  Damage dealt = damage to boss\n"
        "  Damage taken = damage received\n"
        "  Dodged attacks = boss hits avoided\n"
        "  Healed = HP restored this turn\n"
        "  Reward = RL reward for this action\n\n"
        "Board markers:\n"
        "  Black border = selected orb\n"
        "  Orange border = collected orbs\n"
        "  Rainbow orb = white cell with stripes"
    )

def _draw_hp_bar(ax, x, y, width, height, value, max_value, label):
    ratio = 0.0 if max_value <= 0 else max(0.0, min(1.0, value / max_value))

    ax.text(x, y + height + 0.04, f"{label}: {value:.1f}/{max_value:.1f}", fontsize=10)
    ax.add_patch(Rectangle((x, y), width, height, edgecolor="black", facecolor="none"))
    ax.add_patch(Rectangle((x, y), width * ratio, height, edgecolor="none", facecolor="#2ecc71"))


def _draw_boss_hp_bar(ax, x, y, width, height, before, after, max_value):
    before_ratio = 0.0 if max_value <= 0 else max(0.0, min(1.0, before / max_value))
    after_ratio = 0.0 if max_value <= 0 else max(0.0, min(1.0, after / max_value))

    ax.text(
        x,
        y + height + 0.04,
        f"Boss HP: {before:.1f} -> {after:.1f}/{max_value:.1f}",
        fontsize=10,
    )

    ax.add_patch(Rectangle((x, y), width, height, edgecolor="black", facecolor="none"))
    ax.add_patch(Rectangle((x, y), width * before_ratio, height, edgecolor="none", facecolor="#e74c3c"))
    ax.add_patch(Rectangle((x, y), width * after_ratio, height, edgecolor="none", facecolor="#27ae60"))


def _draw_board(ax, board, selected_orb=None, collected_positions=None):
    board = np.array(board)
    board_size = board.shape[0]
    collected_positions = set(collected_positions or [])

    ax.set_xlim(0, board_size)
    ax.set_ylim(0, board_size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for r in range(board_size):
        for c in range(board_size):
            orb_id = int(board[r, c])
            orb_name = ORB_LABELS[orb_id]

            x = c
            y = board_size - 1 - r

            rect = Rectangle(
                (x, y),
                1,
                1,
                facecolor=ORB_COLORS[orb_name],
                edgecolor="black",
                linewidth=1.3,
            )
            ax.add_patch(rect)

            # Rainbow visual: diagonal colored stripes.
            if orb_name == "RNB":
                stripe_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6"]
                for i, color in enumerate(stripe_colors):
                    ax.plot(
                        [x + 0.08 + i * 0.17, x + 0.28 + i * 0.17],
                        [y + 0.10, y + 0.90],
                        color=color,
                        linewidth=2.2,
                    )

            ax.text(
                x + 0.5,
                y + 0.5,
                orb_name,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=ORB_TEXT_COLORS[orb_name],
            )

            # Collected orbs: thick orange border.
            if (r, c) in collected_positions:
                ax.add_patch(
                    Rectangle(
                        (x + 0.06, y + 0.06),
                        0.88,
                        0.88,
                        facecolor="none",
                        edgecolor="#ff7f00",
                        linewidth=4.5,
                    )
                )

    # Selected orb: black outer border.
    if selected_orb is not None:
        sr, sc = selected_orb
        ax.add_patch(
            Rectangle(
                (sc + 0.015, board_size - 1 - sr + 0.015),
                0.97,
                0.97,
                facecolor="none",
                edgecolor="black",
                linewidth=5,
            )
        )


def render_initial_frame(
    board,
    player_hp,
    player_max_hp,
    boss_hp,
    boss_max_hp,
    phase,
    total_phases,
    boss_name,
    boss_type,
    next_boss_attacks,
    save_path,
):
    """
    Render the initial state of an episode before the first action.
    """
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])

    ax_board = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

    ax_board.set_title(f"Initial State | Phase {phase}/{total_phases}", fontsize=14)
    _draw_board(ax_board, board)

    ax_info.axis("off")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)

    _draw_hp_bar(ax_info, 0.02, 0.78, 0.85, 0.045, player_hp, player_max_hp, "Player HP")
    _draw_boss_hp_bar(ax_info, 0.02, 0.64, 0.85, 0.045, boss_hp, boss_hp, boss_max_hp)

    text = (
        f"Boss: {boss_name}\n"
        f"Boss type: {boss_type}\n"
        f"Next attacks: {[round(a, 1) for a in next_boss_attacks]}\n"
        f"Total attack: {sum(next_boss_attacks):.1f}\n\n"
        f"No action has been taken yet."
    )

    ax_info.text(
        0.02,
        0.50,
        text,
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def render_step_frame(
    board,
    player_hp,
    player_max_hp,
    boss_hp_before,
    boss_hp_after,
    boss_max_hp,
    phase,
    total_phases,
    turn,
    selected_orb,
    collected_positions,
    damage_dealt,
    damage_taken,
    healed,
    reward,
    unit_name,
    unit_type,
    boss_name,
    boss_type,
    next_boss_attacks_before,
    orb_counts=None,
    orb_effects=None,
    dodged_attacks=0,
    phase_cleared=False,
    boss_attack_reduction=0.0,
    save_path=None,
):
    """
    Render a single step of the environment.

    The board shown is the board before the action.
    The selected orb and all collected orbs are highlighted.
    """
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.2])

    ax_board = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

    title = f"Turn {turn} | Phase {phase}/{total_phases}"
    if phase_cleared:
        title += " | PHASE CLEARED"

    ax_board.set_title(title, fontsize=14)
    _draw_board(
        ax_board,
        board=board,
        selected_orb=selected_orb,
        collected_positions=collected_positions,
    )

    ax_info.axis("off")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)

    # HP bars
    _draw_hp_bar(ax_info, 0.02, 0.90, 0.86, 0.035, player_hp, player_max_hp, "Player HP")
    _draw_boss_hp_bar(
        ax_info,
        0.02,
        0.79,
        0.86,
        0.035,
        boss_hp_before,
        boss_hp_after,
        boss_max_hp,
    )

    orb_counts = orb_counts or {}
    orb_effects = orb_effects or {}

    orb_counts_text = ", ".join([f"{k}:{v}" for k, v in orb_counts.items() if v > 0])
    if not orb_counts_text:
        orb_counts_text = "none"

    type_relation = _get_type_relation(unit_type, boss_type)

    action_text = (
        f"Boss: {boss_name} ({boss_type})\n"
        f"Boss attacks: {[round(a, 1) for a in next_boss_attacks_before]}\n"
        f"Boss attack reduction: {boss_attack_reduction:.2f}\n\n"
        f"Action:\n"
        f"  Unit: {unit_name} ({unit_type})\n"
        f"  Type relation: {type_relation}\n"
        f"  Selected orb: {selected_orb}\n"
        f"  Collected orbs: {len(collected_positions)}\n"
        f"  Orb counts: {orb_counts_text}\n\n"
        f"Results:\n"
        f"  Damage dealt: {damage_dealt:.1f}\n"
        f"  Damage taken: {damage_taken:.1f}\n"
        f"  Dodged attacks: {dodged_attacks}\n"
        f"  Healed: {healed:.1f}\n"
        f"  Reward: {reward:.3f}\n\n"
        f"Orb effects:\n"
        f"  DMG +{orb_effects.get('damage_bonus', 0.0):.2f}\n"
        f"  Dodge +{orb_effects.get('dodge_bonus', 0.0):.2f}\n"
        f"  DEF red. {orb_effects.get('defense_bonus', 0.0):.2f}\n"
        f"  Boss ATK debuff +{orb_effects.get('boss_attack_reduction', 0.0):.2f}\n"
        f"  Heal {orb_effects.get('heal_amount', 0.0):.1f}\n"
        f"  Rainbow +{orb_effects.get('rainbow_bonus', 0.0):.2f}"
    )

    legend_text = _build_legend_text()

    # Left text block (action/results)
    ax_info.text(
        0.02,
        0.70,
        action_text,
        ha="left",
        va="top",
        fontsize=10.5,
        family="monospace",
    )

    ax_info.text(
        0.59,
        0.64,
        legend_text,
        ha="left",
        va="top",
        fontsize=9.5,
        family="monospace",
    )

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=130)
        plt.close(fig)
    else:
        plt.show()

def make_gif(frame_paths, output_path, duration=900):
    """
    Create a GIF from a list of saved frame paths.
    """
    if not frame_paths:
        raise ValueError("No frames provided.")

    images = [Image.open(path) for path in frame_paths]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )