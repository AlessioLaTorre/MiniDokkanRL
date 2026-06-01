import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import IntEnum


class OrbType(IntEnum):
    STR = 0
    AGL = 1
    TEQ = 2
    INT = 3
    PHY = 4
    RAINBOW = 5


class UnitType(IntEnum):
    STR = 0
    AGL = 1
    TEQ = 2
    INT = 3
    PHY = 4


ORB_NAMES = {
    OrbType.STR: "STR",
    OrbType.AGL: "AGL",
    OrbType.TEQ: "TEQ",
    OrbType.INT: "INT",
    OrbType.PHY: "PHY",
    OrbType.RAINBOW: "RNB",
}

UNIT_NAMES = {
    UnitType.STR: "STR",
    UnitType.AGL: "AGL",
    UnitType.TEQ: "TEQ",
    UnitType.INT: "INT",
    UnitType.PHY: "PHY",
}


class MiniDokkanEnv(gym.Env):
    """
    MiniDokkanEnv

    A custom Gymnasium-like environment inspired by turn-based orb-collection battles.

    The agent controls a team of 3 units.
    At each turn, it chooses:
        - which unit to use;
        - which board cell to select.

    Action space:
        Discrete(48) = 3 units * 16 cells

    Observation:
        Flattened normalized vector containing:
        - board one-hot encoding;
        - player HP;
        - current boss HP;
        - boss type;
        - next boss attack;
        - current phase;
        - turn;
        - unit stats and roles.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        board_size=4,
        max_turns=30,
        render_mode=None,
        seed=None,
    ):
        super().__init__()

        self.board_size = board_size
        self.num_cells = board_size * board_size
        self.num_orb_types = 6
        self.num_unit_types = 5
        self.num_units = 3
        self.num_roles = 3

        self.max_turns = max_turns
        self.render_mode = render_mode

        self.rng = np.random.default_rng(seed)

        # Action: choose one of 3 units and one of 16 board cells.
        self.action_space = spaces.Discrete(self.num_units * self.num_cells)

        # Observation size:
        # board one-hot: 16 * 6 = 96
        # player hp: 1
        # boss hp: 1
        # boss type one-hot: 5
        # next boss attack: 1
        # phase: 1
        # turn: 1
        # units: 3 * (type one-hot 5 + atk 1 + def 1 + dodge 1 + role one-hot 3) = 33
        obs_dim = (
                self.num_cells * self.num_orb_types  # board one-hot
                + 1  # player HP
                + 1  # boss HP
                + self.num_unit_types  # boss type one-hot
                + 1  # total next boss attack
                + 1  # boss attack reduction
                + 1  # phase
                + 1  # turn
                + self.num_units * (
                        self.num_unit_types  # unit type one-hot
                        + 1  # unit ATK
                        + 1  # unit DEF
                        + 1  # unit dodge
                        + self.num_roles  # unit role one-hot
                )
        ) # 96 + 1 + 1 + 5 + 1 + 1 + 1 + 1 + 33 = 140

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.player_max_hp = 300.0

        self.team_template = [
            {
                "name": "Tank",
                "type": UnitType.STR,
                "atk": 45.0,
                "def": 35.0,
                "dodge": 0.05,
                "role": 0,
            },
            {
                "name": "Damage Dealer",
                "type": UnitType.AGL,
                "atk": 75.0,
                "def": 12.0,
                "dodge": 0.10,
                "role": 1,
            },
            {
                "name": "Healer",
                "type": UnitType.TEQ,
                "atk": 50.0,
                "def": 20.0,
                "dodge": 0.08,
                "role": 2,
            },
        ]

        self.boss_phase_configs = [
            {
                "name": "Phase 1 Boss",
                "hp_range": (220.0, 280.0),
                "atk_range": (35.0, 45.0),
                "num_attacks_range": (1, 1),
            },
            {
                "name": "Phase 2 Boss",
                "hp_range": (330.0, 430.0),
                "atk_range": (45.0, 60.0),
                "num_attacks_range": (1, 2),
            },
            {
                "name": "Final Boss",
                "hp_range": (500.0, 650.0),
                "atk_range": (60.0, 80.0),
                "num_attacks_range": (2, 3),
            },
        ]

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.turn = 0
        self.current_phase = 0
        self.player_hp = self.player_max_hp

        self.team = [unit.copy() for unit in self.team_template]
        self.bosses = [self._generate_boss(i) for i in range(len(self.boss_phase_configs))]

        self.current_boss = self.bosses[self.current_phase].copy()
        self.current_boss_hp = self.current_boss["max_hp"]
        self.next_boss_attacks = self._sample_boss_attacks()

        self.board = self._generate_board()

        self.last_info = {}

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """
        Execute one environment step.

        The agent selects:
        - one unit from the team;
        - one orb cell from the board.

        The environment then:
        - collects connected orbs;
        - computes orb effects;
        - applies damage to the current boss;
        - checks whether the current phase is cleared;
        - applies boss damage if the boss survives;
        - applies healing;
        - replaces collected orbs;
        - computes reward and returns the next observation.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # ------------------------------------------------------------
        # Snapshot before the action.
        # ------------------------------------------------------------
        phase_before = self.current_phase
        boss_before = self.current_boss.copy()
        boss_hp_before = float(self.current_boss_hp)
        boss_max_hp_before = float(self.current_boss["max_hp"])
        board_before = self.board.copy()
        next_boss_attacks_before = list(self.next_boss_attacks)

        self.turn += 1

        # ------------------------------------------------------------
        # Decode action.
        # action = unit_idx * num_cells + orb_idx
        # ------------------------------------------------------------
        unit_idx, orb_idx = self._decode_action(action)
        selected_unit = self.team[unit_idx]

        row = orb_idx // self.board_size
        col = orb_idx % self.board_size

        # ------------------------------------------------------------
        # Collect orbs and compute orb-related quantities.
        # ------------------------------------------------------------
        collected_positions = self._get_connected_orbs(row, col)
        collected_orbs = [self.board[r, c] for r, c in collected_positions]

        total_orbs = len(collected_orbs)
        matching_orbs = self._count_matching_orbs(collected_orbs, selected_unit["type"])
        rainbow_orbs = sum(1 for orb in collected_orbs if OrbType(orb) == OrbType.RAINBOW)

        orb_counts = self._count_orb_types(collected_orbs)
        orb_effects = self._compute_orb_effects(orb_counts)

        # ------------------------------------------------------------
        # Player attacks the current boss.
        # ------------------------------------------------------------
        damage_dealt = self._compute_player_damage(
            unit=selected_unit,
            total_orbs=total_orbs,
            matching_orbs=matching_orbs,
            rainbow_orbs=rainbow_orbs,
            orb_effects=orb_effects,
        )

        self.current_boss_hp = max(0.0, self.current_boss_hp - damage_dealt)
        boss_hp_after = float(self.current_boss_hp)

        phase_cleared = self.current_boss_hp <= 0.0
        all_phases_cleared = False

        damage_taken = 0.0
        dodged_attacks = 0

        # ------------------------------------------------------------
        # If the boss survives, it attacks.
        # If the boss dies, move to the next phase.
        # ------------------------------------------------------------
        if phase_cleared:
            self.current_phase += 1

            if self.current_phase >= len(self.bosses):
                all_phases_cleared = True
            else:
                self.current_boss = self.bosses[self.current_phase].copy()
                self.current_boss_hp = self.current_boss["max_hp"]
                self.next_boss_attacks = self._sample_boss_attacks()
        else:
            damage_taken, dodged_attacks = self._compute_boss_damage(
                selected_unit=selected_unit,
                matching_orbs=matching_orbs,
                orb_effects=orb_effects,
            )
            self.player_hp = max(0.0, self.player_hp - damage_taken)

        # ------------------------------------------------------------
        # Healing from PHY orbs and healer passive.
        # Healing is applied after boss damage.
        # ------------------------------------------------------------
        healed = orb_effects["heal_amount"]

        if selected_unit["role"] == 2 and matching_orbs >= 3:
            healed += 0.05 * self.player_max_hp

        self.player_hp = min(self.player_max_hp, self.player_hp + healed)

        # ------------------------------------------------------------
        # Replace collected orbs.
        # ------------------------------------------------------------
        self._replace_collected_orbs(collected_positions)

        # ------------------------------------------------------------
        # Sample next boss attacks for the next turn.
        # Important: if we already sampled attacks for a new phase above,
        # this overwrites them once, which is okay but redundant.
        # To avoid double sampling, only sample here if the phase was not cleared.
        # ------------------------------------------------------------
        if not phase_cleared and not all_phases_cleared and self.player_hp > 0:
            self.next_boss_attacks = self._sample_boss_attacks()

        terminated = all_phases_cleared or self.player_hp <= 0.0
        truncated = self.turn >= self.max_turns and not terminated

        # ------------------------------------------------------------
        # Reward uses boss_max_hp_before, because damage was dealt to
        # the boss that existed before the action.
        # ------------------------------------------------------------
        reward = self._compute_reward(
            damage_dealt=damage_dealt,
            damage_taken=damage_taken,
            boss_max_hp=boss_max_hp_before,
            phase_cleared=phase_cleared,
            all_phases_cleared=all_phases_cleared,
            player_dead=self.player_hp <= 0.0,
        )

        # ------------------------------------------------------------
        # Logging info.
        # These values are for debugging, rendering and evaluation.
        # They are not directly used by the agent.
        # ------------------------------------------------------------
        active_boss_attack_reduction = (
            float(self.current_boss["attack_reduction"])
            if not all_phases_cleared
            else 0.0
        )

        self.last_info = {
            "turn": int(self.turn),

            # Phase and boss before the action.
            "phase": int(phase_before),
            "phase_before": int(phase_before),
            "phase_after": int(self.current_phase),
            "boss_name_before": boss_before["name"],
            "boss_type_before": boss_before["type"],
            "boss_hp_before": float(boss_hp_before),
            "boss_hp_after": float(boss_hp_after),
            "boss_max_hp_before": float(boss_max_hp_before),
            "boss_attack_reduction": active_boss_attack_reduction,
            "next_boss_attacks_before": [float(a) for a in next_boss_attacks_before],

            # Board/action.
            "board_before": board_before,
            "unit_idx": int(unit_idx),
            "unit_name": selected_unit["name"],
            "unit_type": UNIT_NAMES[selected_unit["type"]],
            "orb_idx": int(orb_idx),
            "orb_position": (int(row), int(col)),
            "collected_positions": [(int(r), int(c)) for r, c in collected_positions],

            # Orb stats.
            "total_orbs": int(total_orbs),
            "matching_orbs": int(matching_orbs),
            "rainbow_orbs": int(rainbow_orbs),
            "orb_counts": {ORB_NAMES[k]: int(v) for k, v in orb_counts.items()},
            "orb_effects": {k: float(v) for k, v in orb_effects.items()},

            # Step results.
            "damage_dealt": float(damage_dealt),
            "damage_taken": float(damage_taken),
            "dodged_attacks": int(dodged_attacks),
            "healed": float(healed),
            "phase_cleared": bool(phase_cleared),
            "all_phases_cleared": bool(all_phases_cleared),
            "player_hp": float(self.player_hp),
            "boss_hp": float(boss_hp_after),
            "reward": float(reward),
        }

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        print("=" * 60)
        print(f"Turn: {self.turn}")

        if self.current_phase < len(self.bosses):
            print(
                f"Phase: {self.current_phase + 1}/{len(self.bosses)} | "
                f"Boss: {self.current_boss['name']} | "
                f"Boss Type: {UNIT_NAMES[self.current_boss['type']]}"
            )
            print(
                f"Player HP: {self.player_hp:.1f}/{self.player_max_hp:.1f} | "
                f"Boss HP: {self.current_boss_hp:.1f}/{self.current_boss['max_hp']:.1f} | "
                f"Next Boss Attacks: {[round(a, 1) for a in self.next_boss_attacks]} | "
                f"Total: {sum(self.next_boss_attacks):.1f}"
            )
        else:
            print("All phases cleared!")

        print("\nBoard:")
        for r in range(self.board_size):
            row = []
            for c in range(self.board_size):
                row.append(f"{ORB_NAMES[OrbType(self.board[r, c])]:>3}")
            print(" ".join(row))

        if self.last_info:
            print("\nLast action:")
            print(
                f"Selected Unit: {self.last_info['unit_name']} | "
                f"Selected Orb: {self.last_info['orb_position']} | "
                f"Collected: {self.last_info['total_orbs']} "
                f"(matching: {self.last_info['matching_orbs']}, "
                f"rainbow: {self.last_info['rainbow_orbs']})"
            )
            print(
                f"Damage Dealt: {self.last_info['damage_dealt']:.2f} | "
                f"Damage Taken: {self.last_info['damage_taken']:.2f} | "
                f"Dodged Attacks: {self.last_info['dodged_attacks']} | "
                f"Healed: {self.last_info['healed']:.2f} | "
                f"Reward: {self.last_info['reward']:.3f}"
            )
        print("=" * 60)

    def _decode_action(self, action):
        unit_idx = action // self.num_cells
        orb_idx = action % self.num_cells
        return unit_idx, orb_idx

    def _generate_board(self):
        return self.rng.integers(
            low=0,
            high=self.num_orb_types,
            size=(self.board_size, self.board_size),
            dtype=np.int64,
        )

    def _replace_collected_orbs(self, positions):
        for r, c in positions:
            self.board[r, c] = self.rng.integers(0, self.num_orb_types)

    def _get_connected_orbs(self, row, col):
        """
        Returns connected orbs starting from selected cell.

        Rule:
        - If selected orb is not rainbow:
            collect connected orbs of same color plus rainbow orbs.
        - If selected orb is rainbow:
            collect connected rainbow orbs only.

        Connectivity: up, down, left, right.
        """

        selected_orb = OrbType(self.board[row, col])
        visited = set()
        stack = [(row, col)]

        def is_compatible(orb):
            orb = OrbType(orb)

            if selected_orb == OrbType.RAINBOW:
                return orb == OrbType.RAINBOW

            return orb == selected_orb or orb == OrbType.RAINBOW

        while stack:
            r, c = stack.pop()

            if (r, c) in visited:
                continue

            if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                continue

            if not is_compatible(self.board[r, c]):
                continue

            visited.add((r, c))

            neighbors = [
                (r - 1, c),
                (r + 1, c),
                (r, c - 1),
                (r, c + 1),
            ]

            for nr, nc in neighbors:
                if (nr, nc) not in visited:
                    stack.append((nr, nc))

        return list(visited)

    def _count_matching_orbs(self, collected_orbs, unit_type):
        matching = 0

        for orb in collected_orbs:
            orb = OrbType(orb)

            if orb == OrbType.RAINBOW:
                matching += 1
            elif int(orb) == int(unit_type):
                matching += 1

        return matching

    def _compute_player_damage(self, unit, total_orbs, matching_orbs, rainbow_orbs, orb_effects):
        """
            Compute the damage dealt by the selected unit to the current boss.

            Damage depends on:
            - the unit attack stat;
            - the number of collected orbs;
            - matching orbs;
            - rainbow orbs;
            - type advantage/disadvantage;
            - tactical orb effects, especially STR and RAINBOW;
            - the Damage Dealer passive.

            Args:
                unit: selected unit dictionary.
                total_orbs: total number of collected orbs.
                matching_orbs: number of collected orbs matching the selected unit type.
                rainbow_orbs: number of collected rainbow orbs.
                orb_effects: dict produced by _compute_orb_effects().

            Returns:
                float damage value.
        """
        base_damage = unit["atk"]

        orb_multiplier = 1.0 + 0.12 * total_orbs
        matching_multiplier = 1.0 + 0.06 * matching_orbs
        rainbow_multiplier = 1.0 + 0.03 * rainbow_orbs
        type_multiplier = self._type_multiplier(
            attacker_type=unit["type"],
            defender_type=self.current_boss["type"],
        )

        damage = (
            base_damage
            * orb_multiplier
            * matching_multiplier
            * rainbow_multiplier
            * type_multiplier
        )
        damage *= 1.0 + orb_effects["damage_bonus"]
        damage *= 1.0 + orb_effects["rainbow_bonus"]

        # Passive: Damage Dealer gets bonus damage with at least 4 collected orbs.
        if unit["role"] == 1 and total_orbs >= 4:
            damage *= 1.40

        return float(damage)


    def _sample_boss_attacks(self):
        attacks = []

        for _ in range(self.current_boss["num_attacks"]):
            noise = self.rng.uniform(0.85, 1.15)
            attacks.append(float(self.current_boss["base_atk"] * noise))

        return attacks

    def _compute_boss_damage(self, selected_unit, matching_orbs, orb_effects):
        """
            Compute the total damage received from all boss attacks in the current turn.

            Boss damage depends on:
            - the list of sampled boss attacks;
            - type advantage/disadvantage;
            - selected unit defense;
            - AGL orb effect, which increases dodge chance;
            - TEQ orb effect, which reduces incoming damage;
            - INT orb effect, which reduces boss attack;
            - RAINBOW orb effect, which slightly improves defense;
            - Tank passive.

            Args:
                selected_unit: unit chosen by the agent.
                matching_orbs: number of orbs matching the selected unit type.
                orb_effects: dict produced by _compute_orb_effects().

            Returns:
                tuple:
                    total_damage: total damage taken during this turn.
                    dodged_attacks: number of boss attacks dodged.
        """
        total_damage = 0.0
        dodged_attacks = 0

        dodge_chance = selected_unit["dodge"] + orb_effects["dodge_bonus"]
        dodge_chance = min(dodge_chance, 0.60)

        defense_reduction = orb_effects["defense_bonus"] + orb_effects["rainbow_bonus"]
        defense_reduction = min(defense_reduction, 0.60)

        attack_reduction = orb_effects["boss_attack_reduction"]
        attack_reduction = min(attack_reduction, 0.50)
        self.current_boss["attack_reduction"] = min(0.50, attack_reduction + self.current_boss["attack_reduction"])

        for attack in self.next_boss_attacks:
            if self.rng.random() < dodge_chance:
                dodged_attacks += 1
                continue

            raw_damage = attack

            type_multiplier = self._type_multiplier(
                attacker_type=self.current_boss["type"],
                defender_type=selected_unit["type"],
            )

            raw_damage *= type_multiplier
            reduced_damage = max(1.0, raw_damage - selected_unit["def"])

            reduced_damage *= 1.0 - defense_reduction

            if selected_unit["role"] == 0 and matching_orbs >= 3:
                reduced_damage *= 0.70

            total_damage += reduced_damage

        total_damage -= total_damage * self.current_boss["attack_reduction"]

        return float(total_damage), dodged_attacks

    def _type_multiplier(self, attacker_type, defender_type):
        """
        Simplified type wheel:

        STR > PHY > INT > TEQ > AGL > STR

        Advantage: 1.5
        Disadvantage: 0.75
        Neutral: 1.0
        """

        advantage = {
            UnitType.STR: UnitType.PHY,
            UnitType.PHY: UnitType.INT,
            UnitType.INT: UnitType.TEQ,
            UnitType.TEQ: UnitType.AGL,
            UnitType.AGL: UnitType.STR,
        }

        attacker_type = UnitType(attacker_type)
        defender_type = UnitType(defender_type)

        if advantage[attacker_type] == defender_type:
            return 1.5

        if advantage[defender_type] == attacker_type:
            return 0.75

        return 1.0

    def _generate_boss(self, phase_idx):
        config = self.boss_phase_configs[phase_idx]

        boss_type = UnitType(self.rng.integers(0, self.num_unit_types))
        max_hp = self.rng.uniform(*config["hp_range"])
        base_atk = self.rng.uniform(*config["atk_range"])

        min_attacks, max_attacks = config["num_attacks_range"]
        num_attacks = int(self.rng.integers(min_attacks, max_attacks + 1))

        return {
            "name": config["name"],
            "type": boss_type,
            "max_hp": float(max_hp),
            "base_atk": float(base_atk),
            "num_attacks": num_attacks,
            "attack_reduction": 0.0,
        }

    def _count_orb_types(self, collected_orbs):
        counts = {orb_type: 0 for orb_type in OrbType}

        for orb in collected_orbs:
            counts[OrbType(orb)] += 1

        return counts

    def _compute_orb_effects(self, orb_counts):
        """
            Convert collected orb counts into gameplay effects.

            The goal is to make orb choice more strategic:
            - STR orbs increase player damage.
            - AGL orbs increase dodge chance for the current turn.
            - TEQ orbs reduce incoming damage for the current turn.
            - INT orbs reduce boss attack for the current turn.
            - PHY orbs heal the player.
            - RAINBOW orbs provide a small general bonus.

            Args:
                orb_counts: dict produced by _count_orb_types().

            Returns:
                dict containing numerical effects used by damage, defense and healing logic.
        """
        rainbow = orb_counts[OrbType.RAINBOW]

        effects = {
            "damage_bonus": 0.12 * orb_counts[OrbType.STR],
            "dodge_bonus": 0.04 * orb_counts[OrbType.AGL],
            "defense_bonus": 0.08 * orb_counts[OrbType.TEQ],
            "boss_attack_reduction": 0.04 * orb_counts[OrbType.INT],
            "heal_amount": 0.025 * self.player_max_hp * orb_counts[OrbType.PHY],
            "rainbow_bonus": 0.03 * rainbow,
        }

        return effects

    def _compute_reward(
        self,
        damage_dealt,
        damage_taken,
        boss_max_hp,
        phase_cleared,
        all_phases_cleared,
        player_dead,
    ):
        reward = 0.0

        reward += damage_dealt / boss_max_hp
        reward -= damage_taken / self.player_max_hp

        # Small time penalty.
        reward -= 0.01

        if phase_cleared:
            reward += 0.50

        if all_phases_cleared:
            reward += 2.00

        if player_dead:
            reward -= 2.00

        return float(reward)

    def _get_obs(self):
        """
        Build the observation vector returned to the agent.

        Observation contains:
        - board one-hot encoding;
        - player HP;
        - current boss HP;
        - current boss type;
        - total next boss attack;
        - current boss attack reduction debuff;
        - current phase;
        - current turn;
        - team unit features.
        """
        obs_parts = []

        # ------------------------------------------------------------
        # Board one-hot encoding.
        # Shape: board_size * board_size * num_orb_types
        # ------------------------------------------------------------
        board_flat = self.board.flatten()
        board_one_hot = np.zeros(
            (self.num_cells, self.num_orb_types),
            dtype=np.float32,
        )
        board_one_hot[np.arange(self.num_cells), board_flat] = 1.0
        obs_parts.append(board_one_hot.flatten())

        # ------------------------------------------------------------
        # Player HP normalized.
        # ------------------------------------------------------------
        player_hp_norm = self.player_hp / self.player_max_hp
        obs_parts.append(np.array([player_hp_norm], dtype=np.float32))

        # ------------------------------------------------------------
        # Boss-related features.
        # ------------------------------------------------------------
        if self.current_phase < len(self.bosses):
            boss_hp_norm = self.current_boss_hp / self.current_boss["max_hp"]
            boss_type = self.current_boss["type"]
            boss_attack_norm = sum(self.next_boss_attacks) / 300.0
            boss_attack_reduction = self.current_boss["attack_reduction"]
        else:
            boss_hp_norm = 0.0
            boss_type = UnitType.STR
            boss_attack_norm = 0.0
            boss_attack_reduction = 0.0

        obs_parts.append(np.array([boss_hp_norm], dtype=np.float32))

        boss_type_one_hot = np.zeros(self.num_unit_types, dtype=np.float32)
        boss_type_one_hot[int(boss_type)] = 1.0
        obs_parts.append(boss_type_one_hot)

        obs_parts.append(np.array([boss_attack_norm], dtype=np.float32))
        obs_parts.append(np.array([boss_attack_reduction], dtype=np.float32))

        # ------------------------------------------------------------
        # Phase and turn normalized.
        # ------------------------------------------------------------
        safe_phase = min(self.current_phase, len(self.bosses) - 1)
        phase_norm = safe_phase / max(1, len(self.bosses) - 1)
        obs_parts.append(np.array([phase_norm], dtype=np.float32))

        turn_norm = min(self.turn / self.max_turns, 1.0)
        obs_parts.append(np.array([turn_norm], dtype=np.float32))

        # ------------------------------------------------------------
        # Unit features.
        # For each unit:
        # - type one-hot;
        # - normalized ATK;
        # - normalized DEF;
        # - dodge chance;
        # - role one-hot.
        # ------------------------------------------------------------
        for unit in self.team:
            unit_type_one_hot = np.zeros(self.num_unit_types, dtype=np.float32)
            unit_type_one_hot[int(unit["type"])] = 1.0
            obs_parts.append(unit_type_one_hot)

            obs_parts.append(np.array([unit["atk"] / 100.0], dtype=np.float32))
            obs_parts.append(np.array([unit["def"] / 100.0], dtype=np.float32))
            obs_parts.append(np.array([unit["dodge"]], dtype=np.float32))

            role_one_hot = np.zeros(self.num_roles, dtype=np.float32)
            role_one_hot[int(unit["role"])] = 1.0
            obs_parts.append(role_one_hot)

        obs = np.concatenate(obs_parts).astype(np.float32)

        return obs

    def _get_info(self):
        info = {
            "turn": self.turn,
            "phase": self.current_phase,
            "player_hp": self.player_hp,
            "boss_hp": self.current_boss_hp if self.current_phase < len(self.bosses) else 0.0,
            "next_boss_attack": self.next_boss_attacks if self.current_phase < len(self.bosses) else 0.0,

        }

        info.update(self.last_info)

        return info
