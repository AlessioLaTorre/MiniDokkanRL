# MiniDokkanRL

MiniDokkanRL is a custom Gymnasium-style reinforcement learning environment for tactical orb-based turn-based battles.

The environment is inspired by high-level mechanics from Dokkan Battle, such as orb selection, type matchups, unit roles, and multi-phase boss fights. It is a simplified custom implementation created from scratch for educational purposes. No original assets, code, or proprietary content from the original game are used.

This project was developed for the Autonomous and Adaptive Systems course.

---

## Project Overview

The agent controls a team of three units and fights a sequence of boss phases.

At each turn, the agent chooses:

1. which unit to use;
2. which orb cell to select from the board.

The selected cell determines a connected group of compatible orbs. Collected orbs affect damage, healing, dodge chance, damage reduction, and boss debuffs.

The environment includes:

- default `4x4` orb board;
- support for larger square boards such as `6x6`;
- 6 orb types: STR, AGL, TEQ, INT, PHY, RAINBOW;
- 3 unit roles: Tank, Damage Dealer, Healer;
- cyclic type-advantage system;
- 3 stochastic boss phases;
- Gymnasium-like `reset()` and `step()` API;
- handcrafted baselines and manually implemented RL agents;
- visual rendering and interactive human GUI.

The final version includes a structured observation mode and a CNN-based DQN Q-map agent. Instead of predicting a fixed vector of 48 Q-values, the Q-map agent predicts:

```text
Q(row, col, unit)
```

as a spatial map of shape:

```text
N x N x 3
```

This makes the architecture compatible with different square board sizes.

---

## Repository Structure

```text
MiniDokkanRL/
│
├── agents/                  # RL agents and heuristic baselines
│   ├── random_agent.py
│   ├── greedy_orb_agent.py
│   ├── greedy_damage_agent.py
│   ├── dqn_agent.py
│   ├── dqn_agent_v2.py
│   ├── dqn_qmap_agent.py
│   ├── reinforce_baseline_agent.py
│   ├── reinforce_map_agent.py
│   └── ...
│
├── env/                     # Custom Gymnasium-style environment
│   ├── mini_dokkan_env.py
│   ├── observation.py
│   └── action_mapping.py
│
├── training/                # Training scripts
│   ├── train_dqn.py
│   ├── train_dqn_v2.py
│   ├── train_dqn_qmap.py
│   ├── train_reinforce_baseline.py
│   ├── train_reinforce_map.py
│   └── ...
│
├── evaluation/              # Evaluation utilities
│   └── agents_registry.py
│
├── analysis/                # Tactical analysis and plotting scripts
│   └── analyze_tactics.py
│
├── demos/                   # Visual demos and human-play GUI
│   ├── play_human_gui.py
│   ├── play_human_episode.py
│   └── run_agent_episode_visual.py
│
├── utils/                   # Rendering and utility functions
├── models/                  # Saved trained models, optional/local
├── results/                 # Generated results, plots, GIFs, optional/local
├── report/                  # LaTeX report, optional
├── requirements.txt
└── README.md
```

---

## Installation

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

or manually install the main dependencies:

```bash
pip install numpy gymnasium tensorflow matplotlib pillow pandas
```

---

## Minimal Environment Check

The custom environment follows a Gymnasium-like API and can be tested with a simple random interaction.

```python
from env.mini_dokkan_env import MiniDokkanEnv

env = MiniDokkanEnv(seed=42)

obs, info = env.reset()
next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

print(reward)
```

A more informative check for the original flat observation mode:

```python
from env.mini_dokkan_env import MiniDokkanEnv

env = MiniDokkanEnv(seed=42, board_size=4, obs_mode="flat")

obs, info = env.reset()
action = env.action_space.sample()

next_obs, reward, terminated, truncated, info = env.step(action)

print("Environment check completed.")
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
print(f"Sampled action: {action}")
print(f"Reward: {reward}")
print(f"Terminated: {terminated}")
print(f"Truncated: {truncated}")
```

Expected output should include:

```text
Observation shape: (140,)
Action space: Discrete(48)
Observation space: Box(...)
Reward: ...
```

---

## Structured Observation Check

CNN-based agents use a structured observation:

```python
from env.mini_dokkan_env import MiniDokkanEnv

env = MiniDokkanEnv(
    seed=42,
    board_size=4,
    obs_mode="dict",
)

obs, info = env.reset()

print(obs["board"].shape)
print(obs["global"].shape)
print(obs["units"].shape)
print(env.action_space)
```

For a `4x4` board, the board tensor has shape:

```text
4 x 4 x 6
```

For a `6x6` board:

```python
env = MiniDokkanEnv(
    seed=42,
    board_size=6,
    obs_mode="dict",
)
```

the board tensor becomes:

```text
6 x 6 x 6
```

and the action space becomes:

```text
3 x 6 x 6 = 108 actions
```

---

## Human Play Demo

The project includes an interactive GUI that allows a human user to play the environment manually.

Run:

```bash
python -m demos.play_human_gui
```

The GUI displays:

- orb board;
- unit selection;
- player and boss HP;
- current phase;
- orb effect legend;
- action preview;
- last action result.

This demo is useful for understanding and presenting the environment mechanics.

---

## Visual Agent Demo

To generate a visual episode played by a trained agent:

```bash
python -m demos.run_agent_episode_visual --agent dqn_v2 --seed 42
```

For the CNN Q-map agent:

```bash
python -m demos.run_agent_episode_visual --agent dqn_qmap --seed 42
```

For the REINFORCE Policy Map agent:

```bash
python -m demos.run_agent_episode_visual --agent reinforce_map --seed 42
```

If the chosen seed does not produce an interesting episode, try a different seed:

```bash
python -m demos.run_agent_episode_visual --agent dqn_qmap --seed 100
python -m demos.run_agent_episode_visual --agent dqn_qmap --seed 250
python -m demos.run_agent_episode_visual --agent dqn_qmap --seed 999
```

The script saves episode frames and a GIF in the `results/` directory.

If no trained model is available, run a random visual episode instead:

```bash
python -m demos.run_random_episode_visual
```

---

## Tactical Analysis

To analyze the behavior of a specific agent:

```bash
python -m analysis.analyze_tactics --agent dqn_qmap --episodes 1000 --seed 42 --board-size 4
```

Examples:

```bash
python -m analysis.analyze_tactics --agent dqn_v2 --episodes 1000 --seed 42
python -m analysis.analyze_tactics --agent greedy_damage --episodes 1000 --seed 42 --board-size 4
python -m analysis.analyze_tactics --agent greedy_damage --episodes 1000 --seed 42 --board-size 6
python -m analysis.analyze_tactics --agent reinforce_map --episodes 1000 --seed 42 --board-size 4
```

The tactical analysis records:

- unit usage;
- selected orb types;
- collected orb types;
- type advantage / neutral / disadvantage;
- aggressive vs defensive actions;
- phase clears.

Results are saved to:

```text
results/tactical_analysis.json
```

If the same agent is analyzed again, its entry in the JSON file is updated.

---

## Training

### Minimal DQN

```bash
python -m training.train_dqn
```

### DQN with Replay Buffer and Target Network

```bash
python -m training.train_dqn_v2
```

### CNN DQN Q-map

```bash
python -m training.train_dqn_qmap
```

### REINFORCE with Baseline

```bash
python -m training.train_reinforce_baseline
```

### REINFORCE Policy Map

```bash
python -m training.train_reinforce_map
```

Training scripts save models in `models/` and logs/metrics in `results/`, depending on the script configuration.

Saved models are not necessarily included in the repository. If a model file is missing, it can be regenerated by running the corresponding training script.

---

## DQN Q-map Architecture

The DQN Q-map agent uses a structured observation:

```text
obs = {
    "board":  N x N x 6,
    "global": global feature vector,
    "units":  3 x F unit feature matrix,
}
```

The board is processed with a CNN. Global and unit features are processed with MLP layers, then broadcast over the board and concatenated with the spatial board features.

The final network output is:

```text
N x N x 3
```

where each value represents:

```text
Q(row, col, unit)
```

For a `4x4` board:

```text
4 x 4 x 3 = 48 Q-values
```

For a `6x6` board:

```text
6 x 6 x 3 = 108 Q-values
```

This avoids using a fixed `Dense(48)` output and allows the same architecture to operate on different square board sizes.

---

## REINFORCE Policy Map

The REINFORCE Policy Map agent uses the same structured observation idea as DQN Q-map, but its output has a different meaning.

Instead of predicting Q-values, it predicts policy logits:

```text
logits(row, col, unit)
```

These logits are flattened and converted into a probability distribution over actions. During training, the agent samples actions from this distribution. During evaluation, it selects the action with the highest probability.

The agent uses a learned value baseline:

```text
A_t = G_t - V(s_t)
```

where `G_t` is the Monte Carlo return and `V(s_t)` is predicted by a value network.

---

## Main Experimental Results

Final evaluation was performed over 1000 episodes with fixed seeds and exploration disabled.

| Agent / Setting | Avg. Return | Win Rate | Phase 3 Clears |
|---|---:|---:|---:|
| Random 4x4 | 0.39 | 3.5% | 35 |
| GreedyDamage 4x4 | 2.19 | 26.1% | 261 |
| DQN Replay 4x4 | 2.66 | 37.2% | 372 |
| REINFORCE Policy Map 4x4 | 1.03 | 10.2% | 102 |
| DQN Q-map 4x4 | 3.63 | 52.0% | 520 |
| GreedyDamage 6x6 | 4.63 | 56.6% | 566 |
| DQN Q-map 4x4 -> 6x6 | 5.20 | 71.6% | 716 |

The strongest result is obtained by the DQN Q-map agent. It improves over the previous replay-based DQN on the default `4x4` task and can also be evaluated on a larger `6x6` board despite being trained only on `4x4`.

This suggests that the CNN Q-map architecture learns useful spatial action patterns instead of being tied to a fixed 48-action output.

---

## Reproducibility Notes

Most scripts use fixed seeds where possible. Results may vary slightly due to:

- stochastic boss generation;
- random orb boards;
- stochastic boss attacks;
- dodge events;
- neural network initialization.

The environment is intentionally stochastic to avoid deterministic memorization and to make policies more robust.

---

## Report and Demo

The report is written in NeurIPS LaTeX style as required by the project guidelines.

The project also includes:

- a GUI for human play;
- GIF rendering of agent episodes;
- tactical analysis scripts;
- evaluation scripts for comparing agents;
- structured observation support;
- CNN Q-map training and evaluation.

---

## Disclaimer

MiniDokkanRL is a simplified custom environment inspired by high-level mechanics of Dokkan Battle. It is not affiliated with, endorsed by, or derived from the original game. No original game assets, code, or proprietary content are used.