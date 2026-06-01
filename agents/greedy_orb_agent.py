class GreedyOrbAgent:
    """
    Greedy baseline agent based on orb count.

    It evaluates all possible actions and selects the one that collects
    the highest number of connected orbs.

    This agent does not learn. It is useful to test whether the environment
    rewards simple local strategies too much.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs, info=None, env=None, training=False):
        """
        Select the action that collects the largest number of orbs.

        Args:
            obs: current observation, unused.
            info: current info dictionary, unused.
            env: MiniDokkanEnv instance, required.

        Returns:
            int action.
        """
        if env is None:
            raise ValueError("GreedyOrbAgent requires env to evaluate actions.")

        best_action = None
        best_orb_count = -1

        for action in range(env.action_space.n):
            unit_idx, orb_idx = env._decode_action(action)

            row = orb_idx // env.board_size
            col = orb_idx % env.board_size

            collected_positions = env._get_connected_orbs(row, col)
            orb_count = len(collected_positions)

            if orb_count > best_orb_count:
                best_orb_count = orb_count
                best_action = action

        return best_action