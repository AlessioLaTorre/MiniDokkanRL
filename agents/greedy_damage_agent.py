from env.mini_dokkan_env import OrbType


class GreedyDamageAgent:
    """
    Greedy baseline agent based on immediate damage.

    It evaluates all possible actions and selects the one that would deal
    the highest immediate damage to the current boss.

    This agent does not consider future survival, healing, boss debuffs,
    or damage taken. It is useful to test whether the environment can be
    solved by pure short-term aggression.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs, info=None, env=None, training=False):
        if env is None:
            raise ValueError("GreedyDamageAgent requires env to evaluate actions.")

        best_action = None
        best_damage = -1.0

        for action in range(env.action_space.n):
            unit_idx, orb_idx = env._decode_action(action)
            unit = env.team[unit_idx]

            row = orb_idx // env.board_size
            col = orb_idx % env.board_size

            collected_positions = env._get_connected_orbs(row, col)
            collected_orbs = [env.board[r, c] for r, c in collected_positions]

            total_orbs = len(collected_orbs)
            matching_orbs = env._count_matching_orbs(collected_orbs, unit["type"])
            rainbow_orbs = sum(1 for orb in collected_orbs if OrbType(orb) == OrbType.RAINBOW)

            orb_counts = env._count_orb_types(collected_orbs)
            orb_effects = env._compute_orb_effects(orb_counts)

            damage = env._compute_player_damage(
                unit=unit,
                total_orbs=total_orbs,
                matching_orbs=matching_orbs,
                rainbow_orbs=rainbow_orbs,
                orb_effects=orb_effects,
            )

            if damage > best_damage:
                best_damage = damage
                best_action = action

        return best_action