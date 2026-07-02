import os

from agents.random_agent import RandomAgent
from agents.greedy_orb_agent import GreedyOrbAgent
from agents.greedy_damage_agent import GreedyDamageAgent
from agents.neural_mc_agent import NeuralMCAgent
from agents.dqn_agent import DQNAgent
from agents.dqn_agent_v2 import DQNAgentV2
from agents.reinforce_agent import ReinforceAgent
from agents.reinforce_baseline_agent import ReinforceBaselineAgent
from agents.dqn_agent_v3 import DQNAgentV3
from agents.dqn_qmap_agent import DQNQMapAgent
from agents.reinforce_map_agent import ReinforceMapAgent


def _find_existing_model(possible_paths):
    """
    Return the first model path that exists.

    This is useful because different training scripts may save the same agent
    with slightly different filenames.
    """
    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "No model file found. Tried:\n" + "\n".join(possible_paths)
    )


def build_agent(agent_name, env):
    """
    Build and, when needed, load an agent by name.

    Flat agents use the original flattened observation.
    Q-map agents use structured dict observations.
    """

    if agent_name == "random":
        return RandomAgent(env.action_space)

    if agent_name == "greedy_orb":
        return GreedyOrbAgent(env.action_space)

    if agent_name == "greedy_damage":
        return GreedyDamageAgent(env.action_space)

    # ------------------------------------------------------------
    # DQN Q-map agent.
    # Uses structured observations:
    # obs["board"], obs["global"], obs["units"].
    # ------------------------------------------------------------
    if agent_name == "dqn_qmap":
        obs, _ = env.reset()

        global_dim = int(obs["global"].shape[0])
        unit_dim = int(obs["units"].shape[1])
        num_orb_types = int(obs["board"].shape[-1])
        num_units = int(obs["units"].shape[0])
        board_size = int(obs["board"].shape[0])

        agent = DQNQMapAgent(
            global_dim=global_dim,
            unit_dim=unit_dim,
            board_size=board_size,
            num_orb_types=num_orb_types,
            num_units=num_units,
        )

        agent.load("models/dqn_qmap_10k.keras")
        return agent

    # ------------------------------------------------------------
    # REINFORCE Q-map agent.
    # Uses the same structured observation as DQN Q-map.
    # ------------------------------------------------------------
    if agent_name == "reinforce_map":
        obs, _ = env.reset()

        global_dim = int(obs["global"].shape[0])
        unit_dim = int(obs["units"].shape[1])
        num_orb_types = int(obs["board"].shape[-1])
        num_units = int(obs["units"].shape[0])

        agent = ReinforceMapAgent(
            global_dim=global_dim,
            unit_dim=unit_dim,
            num_orb_types=num_orb_types,
            num_units=num_units,
        )

        agent.load("models/reinforce_map_10k")
        return agent


    # ------------------------------------------------------------
    # All remaining agents use the original flat observation vector.
    # ------------------------------------------------------------
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    if agent_name == "neural_mc":
        agent = NeuralMCAgent(obs_dim=obs_dim, action_dim=action_dim)
        agent.load("models/neural_mc.keras")
        return agent

    if agent_name == "dqn":
        agent = DQNAgent(obs_dim=obs_dim, action_dim=action_dim)
        agent.load("models/minimal_dqn_tf.keras")
        return agent

    if agent_name == "dqn_v2":
        agent = DQNAgentV2(obs_dim=obs_dim, action_dim=action_dim)
        agent.load("models/dqn_v2.keras")
        return agent

    if agent_name == "dqn_v3":
        agent = DQNAgentV3(obs_dim=obs_dim, action_dim=action_dim)
        agent.load("models/dqn_v3_best.keras")
        return agent

    if agent_name == "reinforce":
        agent = ReinforceAgent(obs_dim=obs_dim, action_dim=action_dim)
        agent.load("models/reinforce_10k.keras")
        return agent

    if agent_name == "reinforce_baseline":
        agent = ReinforceBaselineAgent(obs_dim=obs_dim, action_dim=action_dim)
        agent.load("models/reinforce_baseline")
        return agent

    raise ValueError(f"Unknown agent: {agent_name}")


def available_agents():
    return [
        "random",
        "greedy_orb",
        "greedy_damage",
        "neural_mc",
        "dqn",
        "dqn_v2",
        "dqn_v3",
        "reinforce",
        "reinforce_baseline",
        "dqn_qmap",
        "reinforce_map"
    ]