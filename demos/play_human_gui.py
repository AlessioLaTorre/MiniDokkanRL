import tkinter as tk
from tkinter import ttk, messagebox

from env.mini_dokkan_env import (
    MiniDokkanEnv,
    ORB_NAMES,
    UNIT_NAMES,
    OrbType,
)


ORB_COLORS = {
    "STR": "#e74c3c",
    "AGL": "#3498db",
    "TEQ": "#2ecc71",
    "INT": "#9b59b6",
    "PHY": "#f1c40f",
    "RNB": "#eeeeee",
}

ORB_TEXT_COLORS = {
    "STR": "white",
    "AGL": "white",
    "TEQ": "black",
    "INT": "white",
    "PHY": "black",
    "RNB": "black",
}


class HumanPlayGUI:
    """
    Small Tkinter GUI to manually play MiniDokkanEnv.

    Interaction:
    1. Select one unit from the team.
    2. Click one orb cell.
    3. The environment executes the corresponding action.

    A preview panel shows what would happen if the hovered orb
    were selected with the currently selected unit.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("MiniDokkan - Human Play")
        self.root.geometry("1220x760")
        self.root.minsize(1150, 700)

        self.env = MiniDokkanEnv(render_mode=None, seed=42)
        self.obs, self.info = self.env.reset(seed=42)

        self.selected_unit_idx = 0
        self.terminated = False
        self.truncated = False
        self.total_reward = 0.0

        self.board_buttons = []
        self.unit_buttons = []

        self._build_layout()
        self._refresh_all()

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------

    def _build_layout(self):
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill="both", expand=True)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=False, padx=(0, 16))

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="left", fill="both", expand=True)

        # ---------------- Board area ----------------
        board_title = ttk.Label(
            left_frame,
            text="Board",
            font=("Arial", 18, "bold"),
        )
        board_title.pack(pady=(0, 10))

        self.board_frame = ttk.Frame(left_frame)
        self.board_frame.pack()

        for r in range(self.env.board_size):
            row_buttons = []

            for c in range(self.env.board_size):
                button = tk.Button(
                    self.board_frame,
                    width=11,
                    height=5,
                    font=("Arial", 13, "bold"),
                    relief="raised",
                    bd=3,
                    command=lambda row=r, col=c: self._on_orb_click(row, col),
                )

                button.grid(row=r, column=c, padx=5, pady=5)

                button.bind(
                    "<Enter>",
                    lambda event, row=r, col=c: self._show_action_preview(row, col),
                )

                row_buttons.append(button)

            self.board_buttons.append(row_buttons)

        # ---------------- Legend ----------------
        legend_frame = ttk.LabelFrame(left_frame, text="Orb Legend", padding=10)
        legend_frame.pack(fill="x", pady=(18, 0))

        legend_text = (
            "STR  → damage bonus\n"
            "AGL  → dodge bonus\n"
            "TEQ  → defense bonus\n"
            "INT  → permanent boss ATK debuff\n"
            "PHY  → healing\n"
            "RNB  → small general bonus"
        )

        ttk.Label(
            legend_frame,
            text=legend_text,
            justify="left",
            font=("Consolas", 10),
        ).pack(anchor="w")

        # ---------------- Right upper panel ----------------
        top_right = ttk.Frame(right_frame)
        top_right.pack(fill="x")

        self.state_frame = ttk.LabelFrame(top_right, text="Current State", padding=10)
        self.state_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.state_label = ttk.Label(
            self.state_frame,
            text="",
            justify="left",
            font=("Consolas", 11),
        )
        self.state_label.pack(anchor="w")

        control_frame = ttk.LabelFrame(top_right, text="Controls", padding=10)
        control_frame.pack(side="left", fill="y")

        ttk.Button(
            control_frame,
            text="Reset Episode",
            command=self._reset_episode,
        ).pack(fill="x", pady=(0, 8))

        ttk.Button(
            control_frame,
            text="Quit",
            command=self.root.destroy,
        ).pack(fill="x")

        # ---------------- Team panel ----------------
        team_frame = ttk.LabelFrame(right_frame, text="Choose Unit", padding=10)
        team_frame.pack(fill="x", pady=(12, 12))

        for idx, unit in enumerate(self.env.team):
            button = tk.Button(
                team_frame,
                anchor="w",
                justify="left",
                font=("Consolas", 11),
                padx=10,
                pady=8,
                command=lambda unit_idx=idx: self._select_unit(unit_idx),
            )
            button.pack(fill="x", pady=4)
            self.unit_buttons.append(button)

        # ---------------- Bottom two panels ----------------
        bottom_right = ttk.Frame(right_frame)
        bottom_right.pack(fill="both", expand=True)

        preview_frame = ttk.LabelFrame(bottom_right, text="Action Preview", padding=10)
        preview_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.preview_label = ttk.Label(
            preview_frame,
            text="Select a unit, then hover over an orb.",
            justify="left",
            font=("Consolas", 10),
        )
        self.preview_label.pack(anchor="nw")

        log_frame = ttk.LabelFrame(bottom_right, text="Last Action Result", padding=10)
        log_frame.pack(side="left", fill="both", expand=True)

        self.log_label = ttk.Label(
            log_frame,
            text="No action taken yet.",
            justify="left",
            font=("Consolas", 10),
        )
        self.log_label.pack(anchor="nw")

    # ------------------------------------------------------------------
    # Refresh methods
    # ------------------------------------------------------------------

    def _refresh_all(self):
        self._refresh_board()
        self._refresh_team()
        self._refresh_state()

    def _refresh_board(self):
        for r in range(self.env.board_size):
            for c in range(self.env.board_size):
                orb = OrbType(self.env.board[r, c])
                orb_name = ORB_NAMES[orb]

                label = orb_name
                if orb_name == "RNB":
                    label = "RNB\n✦"

                button = self.board_buttons[r][c]
                button.configure(
                    text=label,
                    bg=ORB_COLORS[orb_name],
                    fg=ORB_TEXT_COLORS[orb_name],
                    activebackground=ORB_COLORS[orb_name],
                    activeforeground=ORB_TEXT_COLORS[orb_name],
                    state="disabled" if self.terminated or self.truncated else "normal",
                )

    def _refresh_team(self):
        for idx, unit in enumerate(self.env.team):
            selected_marker = "▶ " if idx == self.selected_unit_idx else "  "

            text = (
                f"{selected_marker}[{idx}] {unit['name']:<15} "
                f"Type={UNIT_NAMES[unit['type']]:<3}   "
                f"ATK={unit['atk']:<5.1f}   "
                f"DEF={unit['def']:<5.1f}   "
                f"Dodge={unit['dodge']:.2f}"
            )

            relief = "sunken" if idx == self.selected_unit_idx else "raised"

            self.unit_buttons[idx].configure(
                text=text,
                relief=relief,
                bd=3,
                state="disabled" if self.terminated or self.truncated else "normal",
            )

    def _refresh_state(self):
        if self.env.current_phase < len(self.env.bosses):
            boss = self.env.current_boss

            state_text = (
                f"Turn: {self.env.turn}\n"
                f"Phase: {self.env.current_phase + 1}/{len(self.env.bosses)}\n\n"
                f"Player HP: {self.env.player_hp:.1f}/{self.env.player_max_hp:.1f}\n\n"
                f"Boss: {boss['name']}\n"
                f"Boss Type: {UNIT_NAMES[boss['type']]}\n"
                f"Boss HP: {self.env.current_boss_hp:.1f}/{boss['max_hp']:.1f}\n"
                f"Boss ATK reduction: {boss['attack_reduction']:.2f}\n\n"
                f"Next boss attacks: "
                f"{[round(a, 1) for a in self.env.next_boss_attacks]}\n"
                f"Total incoming attack: {sum(self.env.next_boss_attacks):.1f}\n\n"
                f"Total reward: {self.total_reward:.3f}"
            )
        else:
            state_text = (
                f"Turn: {self.env.turn}\n"
                f"All phases cleared!\n\n"
                f"Player HP: {self.env.player_hp:.1f}/{self.env.player_max_hp:.1f}\n"
                f"Total reward: {self.total_reward:.3f}"
            )

        self.state_label.configure(text=state_text)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _select_unit(self, unit_idx):
        if self.terminated or self.truncated:
            return

        self.selected_unit_idx = unit_idx
        self._refresh_team()
        self.preview_label.configure(
            text="Unit selected. Hover over an orb to preview the action."
        )

    def _on_orb_click(self, row, col):
        if self.terminated or self.truncated:
            return

        orb_idx = row * self.env.board_size + col
        action = self.selected_unit_idx * self.env.num_cells + orb_idx

        self.obs, reward, self.terminated, self.truncated, self.info = self.env.step(action)
        self.total_reward += reward

        self._refresh_all()
        self._refresh_log(self.info, reward)

        if self.info.get("all_phases_cleared", False):
            messagebox.showinfo(
                "Victory",
                f"You cleared all phases!\nTotal reward: {self.total_reward:.3f}",
            )

        elif self.env.player_hp <= 0:
            messagebox.showinfo(
                "Defeat",
                f"You were defeated.\nTotal reward: {self.total_reward:.3f}",
            )

        elif self.truncated:
            messagebox.showinfo(
                "Episode Finished",
                f"Maximum turns reached.\nTotal reward: {self.total_reward:.3f}",
            )

    def _reset_episode(self):
        self.obs, self.info = self.env.reset(seed=42)
        self.selected_unit_idx = 0
        self.terminated = False
        self.truncated = False
        self.total_reward = 0.0

        self.log_label.configure(text="No action taken yet.")
        self.preview_label.configure(
            text="Select a unit, then hover over an orb."
        )

        self._refresh_all()

    def _show_action_preview(self, row, col):
        if self.terminated or self.truncated:
            return

        selected_unit = self.env.team[self.selected_unit_idx]

        collected_positions = self.env._get_connected_orbs(row, col)
        collected_orbs = [
            self.env.board[r, c]
            for r, c in collected_positions
        ]

        total_orbs = len(collected_orbs)
        matching_orbs = self.env._count_matching_orbs(
            collected_orbs,
            selected_unit["type"],
        )

        rainbow_orbs = sum(
            1
            for orb in collected_orbs
            if OrbType(orb) == OrbType.RAINBOW
        )

        orb_counts = self.env._count_orb_types(collected_orbs)
        orb_effects = self.env._compute_orb_effects(orb_counts)

        estimated_damage = self.env._compute_player_damage(
            unit=selected_unit,
            total_orbs=total_orbs,
            matching_orbs=matching_orbs,
            rainbow_orbs=rainbow_orbs,
            orb_effects=orb_effects,
        )

        readable_counts = {
            ORB_NAMES[key]: value
            for key, value in orb_counts.items()
            if value > 0
        }

        preview_text = (
            f"Preview cell: ({row}, {col})\n"
            f"Selected unit: {selected_unit['name']} "
            f"({UNIT_NAMES[selected_unit['type']]})\n\n"
            f"Collected positions: {collected_positions}\n"
            f"Total orbs: {total_orbs}\n"
            f"Matching orbs: {matching_orbs}\n"
            f"Rainbow orbs: {rainbow_orbs}\n"
            f"Orb counts: {readable_counts}\n\n"
            f"Estimated immediate damage: {estimated_damage:.1f}\n\n"
            f"Orb effects:\n"
            f"  Damage bonus: {orb_effects['damage_bonus']:.2f}\n"
            f"  Dodge bonus: {orb_effects['dodge_bonus']:.2f}\n"
            f"  Defense bonus: {orb_effects['defense_bonus']:.2f}\n"
            f"  Boss ATK debuff: {orb_effects['boss_attack_reduction']:.2f}\n"
            f"  Heal amount: {orb_effects['heal_amount']:.1f}\n"
            f"  Rainbow bonus: {orb_effects['rainbow_bonus']:.2f}"
        )

        self.preview_label.configure(text=preview_text)

    def _refresh_log(self, info, reward):
        log_text = (
            f"Unit used: {info['unit_name']} ({info['unit_type']})\n"
            f"Selected orb: {info['orb_position']}\n"
            f"Collected positions: {info['collected_positions']}\n"
            f"Orb counts: {info['orb_counts']}\n\n"
            f"Damage dealt: {info['damage_dealt']:.1f}\n"
            f"Damage taken: {info['damage_taken']:.1f}\n"
            f"Dodged attacks: {info['dodged_attacks']}\n"
            f"Healed: {info['healed']:.1f}\n"
            f"Boss ATK reduction: {info['boss_attack_reduction']:.2f}\n"
            f"Reward: {reward:.3f}"
        )

        if info["phase_cleared"]:
            log_text += "\n\nPHASE CLEARED!"

        if info["all_phases_cleared"]:
            log_text += "\nALL PHASES CLEARED!"

        self.log_label.configure(text=log_text)


def main():
    root = tk.Tk()
    app = HumanPlayGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
