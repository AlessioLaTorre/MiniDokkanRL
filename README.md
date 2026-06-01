# MiniDokkanRL

MiniDokkanRL is a custom Gymnasium-style reinforcement learning environment for tactical orb-based turn-based battles.

The environment is inspired by high-level mechanics from Dokkan Battle, such as orb selection, type matchups, unit roles, and multi-phase boss fights. It is a simplified custom implementation created from scratch for educational purposes. No original assets, code, or proprietary content from the original game are used.

This project was developed for the Autonomous and Adaptive Systems course.

---

## Project Overview

The agent controls a team of three units and fights a sequence of boss phases.

At each turn, the agent chooses:

1. which unit to use;
2. which cell to select from a 4x4 orb board.

The selected cell determines a connected group of compatible orbs. Collected orbs affect damage, healing, dodge chance, damage reduction, and boss debuffs.

The environment includes:

- 4x4 orb board;
- 6 orb types: STR, AGL, TEQ, INT, PHY, RAINBOW;
- 3 unit roles: Tank, Damage Dealer, Healer;
- cyclic type-advantage system;
- 3 stochastic boss phases;
- Gymnasium-like `reset()` and `step()` API;
- handcrafted baselines and manually implemented RL agents;
- visual rendering and interactive human GUI.

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
│   ├── reinforce_baseline_agent.py
│   └── ...
│
├── env/                     # Custom Gymnasium-style environment
│   └── mini_dokkan_env.py
│
├── training/                # Training scripts
│   ├── train_dqn.py
│   ├── train_dqn_v2.py
│   ├── train_reinforce.py
│   └── ...
│
├── evaluation/              # Evaluation scripts
│   └── evaluate_agent.py
│
├── analysis/                # Tactical analysis and plotting
│   └── analyze_tactics.py
│
├── demos/                   # Visual and human-play demos
│   ├── play_human_gui.py
│   ├── play_human_episode.py
│   └── run_agent_episode_visual.py
│
├── utils/                   # Rendering and utility functions
│
├── models/                  # Saved trained models, optional
├── results/                 # Generated results, plots, GIFs, optional
├── report/                  # LaTeX report, optional
├── requirements.txt
└── README.md
# MiniDokkanRL

MiniDokkanRL is a custom Gymnasium-style reinforcement learning environment for tactical orb-based turn-based battles.

The environment is inspired by high-level mechanics from Dokkan Battle, such as orb selection, type matchups, unit roles, and multi-phase boss fights. It is a simplified custom implementation created from scratch for educational purposes. No original assets, code, or proprietary content from the original game are used.

This project was developed for the Autonomous and Adaptive Systems course.

---

## Project Overview

The agent controls a team of three units and fights a sequence of boss phases.

At each turn, the agent chooses:

1. which unit to use;
2. which cell to select from a 4x4 orb board.

The selected cell determines a connected group of compatible orbs. Collected orbs affect damage, healing, dodge chance, damage reduction, and boss debuffs.

The environment includes:

- 4x4 orb board;
- 6 orb types: STR, AGL, TEQ, INT, PHY, RAINBOW;
- 3 unit roles: Tank, Damage Dealer, Healer;
- cyclic type-advantage system;
- 3 stochastic boss phases;
- Gymnasium-like `reset()` and `step()` API;
- handcrafted baselines and manually implemented RL agents;
- visual rendering and interactive human GUI.

---

## Repository Structure

```text
MiniDokkanRL/
│
├── agents/                  # RL agents and heuristic baselines
├── env/                     # Custom Gymnasium-style environment
├── training/                # Training scripts
├── evaluation/              # Evaluation scripts
├── analysis/                # Tactical analysis and plotting scripts
├── demos/                   # Visual demos and human-play GUI
├── utils/                   # Rendering and utility functions
├── models/                  # Saved models, optional/local
├── results/                 # Generated results, optional/local
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

A more informative check:

```python
from env.mini_dokkan_env import MiniDokkanEnv

env = MiniDokkanEnv(seed=42)

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
Observation space: Box(0.0, 1.0, (140,), float32)
Reward: ...
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

If the chosen seed does not produce an interesting episode, try a different seed:

```bash
python -m demos.run_agent_episode_visual --agent dqn_v2 --seed 100
python -m demos.run_agent_episode_visual --agent dqn_v2 --seed 250
python -m demos.run_agent_episode_visual --agent dqn_v2 --seed 999
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
python -m analysis.analyze_tactics --agent dqn_v2 --episodes 1000 --seed 42
```

Other examples:

```bash
python -m analysis.analyze_tactics --agent greedy_damage --episodes 1000 --seed 42
python -m analysis.analyze_tactics --agent reinforce_baseline --episodes 1000 --seed 42
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

### REINFORCE with Baseline

```bash
python -m training.train_reinforce_baseline
```

Training scripts save models in `models/` and logs/metrics in `results/`, depending on the script configuration.

Saved models are not necessarily included in the repository. If a model file is missing, it can be regenerated by running the corresponding training script.

---

## Main Experimental Result

The strongest learned agent is DQN with replay buffer and target network, referred to as `DQN Replay` in the report and `dqn_v2` in the code.

In the final evaluation over 1000 episodes with fixed seeds and exploration disabled, DQN Replay outperformed the strongest greedy baseline.

| Agent | Avg. Return | Win Rate | Phase 3 Clears |
|---|---:|---:|---:|
| Random | 0.39 | 3.5% | 35 |
| DQN | 1.10 | 14.6% | 73 |
| REINFORCE Baseline | 1.50 | 16.7% | 167 |
| GreedyOrb | 1.18 | 17.8% | 178 |
| GreedyDamage | 2.19 | 26.1% | 261 |
| DQN Replay | 2.66 | 37.2% | 372 |

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
- evaluation scripts for comparing agents.

---

## Disclaimer

MiniDokkanRL is a simplified custom environment inspired by high-level mechanics of Dokkan Battle. It is not affiliated with, endorsed by, or derived from the original game. No original game assets, code, or proprietary content are used.
