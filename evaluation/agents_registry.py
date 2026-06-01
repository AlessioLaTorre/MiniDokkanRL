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


def build_agent(agent_name, env):
    """
    Build and, when needed, load an agent by name.
    """
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    if agent_name == "random":
        return RandomAgent(env.action_space)

    if agent_name == "greedy_orb":
        return GreedyOrbAgent(env.action_space)

    if agent_name == "greedy_damage":
        return GreedyDamageAgent(env.action_space)

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
        "reinforce",
        "reinforce_baseline",
    ]