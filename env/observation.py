"""
observation.py

Observation utilities for MiniDokkanRL.

This file centralizes the construction of observations so that the environment
does not need to directly build flattened vectors internally.

Supported modes:
- "flat": returns the same kind of flattened observation used by the original agents.
- "dict": returns a structured observation suitable for CNN / multi-input models.

Structured observation:
{
    "board":  np.ndarray, shape (board_size, board_size, num_orb_types),
    "global": np.ndarray, shape (global_dim,),
    "units":  np.ndarray, shape (num_units, unit_dim),
}
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------
# Normalization constants.
# These match the scale used in the original flattened observation.
# ---------------------------------------------------------------------

BOSS_ATTACK_SCALE = 300.0
UNIT_ATK_SCALE = 100.0
UNIT_DEF_SCALE = 100.0


# ---------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------

def get_board_size(env) -> int:
    """
    Return the board size.

    MiniDokkanRL uses square boards only:
        board_size x board_size

    This helper also tolerates a tuple/list for future compatibility,
    but it expects height == width.
    """
    board_size = env.board_size

    if isinstance(board_size, (tuple, list)):
        height, width = board_size
        if height != width:
            raise ValueError(
                f"MiniDokkanRL expects a square board, got {height}x{width}."
            )
        return int(height)

    return int(board_size)


def get_num_cells(env) -> int:
    """
    Return the total number of board cells.
    """
    board_size = get_board_size(env)
    return board_size * board_size


def get_global_dim(env) -> int:
    """
    Global features:
    - player HP normalized: 1
    - boss HP normalized: 1
    - boss type one-hot: num_unit_types
    - total incoming boss attack normalized: 1
    - boss attack reduction: 1
    - current phase normalized: 1
    - current turn normalized: 1
    """
    return (
        1
        + 1
        + env.num_unit_types
        + 1
        + 1
        + 1
        + 1
    )


def get_unit_dim(env) -> int:
    """
    Unit features for each unit:
    - unit type one-hot: num_unit_types
    - attack normalized: 1
    - defense normalized: 1
    - dodge probability: 1
    - role one-hot: num_roles
    """
    return (
        env.num_unit_types
        + 1
        + 1
        + 1
        + env.num_roles
    )


def get_flat_obs_dim(env) -> int:
    """
    Return the flattened observation dimension.

    For the original 4x4 environment:
        board one-hot = 16 * 6 = 96
        global       = 11
        units        = 3 * 11 = 33
        total        = 140
    """
    num_cells = get_num_cells(env)

    return (
        num_cells * env.num_orb_types
        + get_global_dim(env)
        + env.num_units * get_unit_dim(env)
    )


# ---------------------------------------------------------------------
# Basic encoding helpers
# ---------------------------------------------------------------------

def one_hot(index: int, size: int) -> np.ndarray:
    """
    Create a one-hot vector.

    Invalid indices are clipped defensively, although the environment should
    normally always provide valid type and role values.
    """
    vec = np.zeros(size, dtype=np.float32)

    index = int(index)
    if 0 <= index < size:
        vec[index] = 1.0

    return vec


# ---------------------------------------------------------------------
# Observation-space builders
# ---------------------------------------------------------------------

def build_observation_space(env, obs_mode: str):
    """
    Build the Gymnasium observation space for the selected observation mode.

    Args:
        env: MiniDokkanEnv instance.
        obs_mode: "flat" or "dict".

    Returns:
        gymnasium.spaces.Space
    """
    obs_mode = obs_mode.lower()
    board_size = get_board_size(env)

    if obs_mode == "flat":
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(get_flat_obs_dim(env),),
            dtype=np.float32,
        )

    if obs_mode == "dict":
        return spaces.Dict(
            {
                "board": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(board_size, board_size, env.num_orb_types),
                    dtype=np.float32,
                ),
                "global": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(get_global_dim(env),),
                    dtype=np.float32,
                ),
                "units": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(env.num_units, get_unit_dim(env)),
                    dtype=np.float32,
                ),
            }
        )

    raise ValueError(f"Unknown obs_mode: {obs_mode}. Use 'flat' or 'dict'.")


# ---------------------------------------------------------------------
# Observation builders
# ---------------------------------------------------------------------

def build_board_observation(env) -> np.ndarray:
    """
    Build a one-hot board tensor.

    Input:
        env.board with shape (board_size, board_size), containing orb ids.

    Output:
        board_obs with shape (board_size, board_size, num_orb_types).
    """
    board_size = get_board_size(env)

    board = np.asarray(env.board, dtype=np.int64)

    if board.shape != (board_size, board_size):
        raise ValueError(
            f"Expected board shape {(board_size, board_size)}, got {board.shape}."
        )

    board_obs = np.zeros(
        (board_size, board_size, env.num_orb_types),
        dtype=np.float32,
    )

    for r in range(board_size):
        for c in range(board_size):
            orb_id = int(board[r, c])
            if 0 <= orb_id < env.num_orb_types:
                board_obs[r, c, orb_id] = 1.0
            else:
                raise ValueError(f"Invalid orb id {orb_id} at position {(r, c)}.")

    return board_obs


def build_global_observation(env) -> np.ndarray:
    """
    Build global battle-state features.

    This contains all non-board and non-unit information needed by the agent.
    The feature order is intentionally kept consistent with the original
    flattened observation.
    """
    features = []

    # ------------------------------------------------------------
    # Player HP normalized.
    # ------------------------------------------------------------
    player_hp_norm = float(env.player_hp) / float(env.player_max_hp)
    features.append(np.array([player_hp_norm], dtype=np.float32))

    # ------------------------------------------------------------
    # Boss-related features.
    # If all phases are cleared, use neutral zero values.
    # ------------------------------------------------------------
    if env.current_phase < len(env.bosses):
        boss_hp_norm = float(env.current_boss_hp) / float(env.current_boss["max_hp"])
        boss_type = int(env.current_boss["type"])
        boss_attack_norm = float(sum(env.next_boss_attacks)) / BOSS_ATTACK_SCALE
        boss_attack_reduction = float(env.current_boss["attack_reduction"])
    else:
        boss_hp_norm = 0.0
        boss_type = 0
        boss_attack_norm = 0.0
        boss_attack_reduction = 0.0

    features.append(np.array([boss_hp_norm], dtype=np.float32))
    features.append(one_hot(boss_type, env.num_unit_types))
    features.append(np.array([boss_attack_norm], dtype=np.float32))
    features.append(np.array([boss_attack_reduction], dtype=np.float32))

    # ------------------------------------------------------------
    # Phase and turn normalized.
    # ------------------------------------------------------------
    safe_phase = min(int(env.current_phase), len(env.bosses) - 1)
    phase_norm = safe_phase / max(1, len(env.bosses) - 1)
    features.append(np.array([phase_norm], dtype=np.float32))

    turn_norm = min(float(env.turn) / float(env.max_turns), 1.0)
    features.append(np.array([turn_norm], dtype=np.float32))

    global_obs = np.concatenate(features).astype(np.float32)

    expected_dim = get_global_dim(env)
    if global_obs.shape != (expected_dim,):
        raise ValueError(
            f"Invalid global observation shape {global_obs.shape}, "
            f"expected {(expected_dim,)}."
        )

    return global_obs


def build_units_observation(env) -> np.ndarray:
    """
    Build unit feature matrix.

    Output shape:
        (num_units, unit_dim)

    Feature order for each unit:
    - type one-hot
    - normalized attack
    - normalized defense
    - dodge probability
    - role one-hot
    """
    unit_features = []

    for unit in env.team:
        parts = []

        # Unit type one-hot.
        parts.append(one_hot(int(unit["type"]), env.num_unit_types))

        # Normalized numerical stats.
        parts.append(np.array([float(unit["atk"]) / UNIT_ATK_SCALE], dtype=np.float32))
        parts.append(np.array([float(unit["def"]) / UNIT_DEF_SCALE], dtype=np.float32))
        parts.append(np.array([float(unit["dodge"])], dtype=np.float32))

        # Unit role one-hot.
        parts.append(one_hot(int(unit["role"]), env.num_roles))

        unit_vec = np.concatenate(parts).astype(np.float32)
        unit_features.append(unit_vec)

    units_obs = np.stack(unit_features, axis=0).astype(np.float32)

    expected_shape = (env.num_units, get_unit_dim(env))
    if units_obs.shape != expected_shape:
        raise ValueError(
            f"Invalid units observation shape {units_obs.shape}, "
            f"expected {expected_shape}."
        )

    return units_obs


def build_dict_observation(env) -> Dict[str, np.ndarray]:
    """
    Build structured observation for multi-input models.

    This is the recommended observation format for CNN-based agents.
    """
    return {
        "board": build_board_observation(env),
        "global": build_global_observation(env),
        "units": build_units_observation(env),
    }


def build_flat_observation(env) -> np.ndarray:
    """
    Build the original flattened observation.

    The output has the same feature ordering as the previous _get_obs()
    implementation inside mini_dokkan_env.py:

    board one-hot flattened
    + global features
    + unit features flattened
    """
    board_obs = build_board_observation(env).reshape(-1)
    global_obs = build_global_observation(env)
    units_obs = build_units_observation(env).reshape(-1)

    flat_obs = np.concatenate(
        [
            board_obs,
            global_obs,
            units_obs,
        ]
    ).astype(np.float32)

    expected_shape = (get_flat_obs_dim(env),)
    if flat_obs.shape != expected_shape:
        raise ValueError(
            f"Invalid flat observation shape {flat_obs.shape}, "
            f"expected {expected_shape}."
        )

    return flat_obs


def build_observation(env, obs_mode: str):
    """
    Build an observation according to obs_mode.

    Args:
        env: MiniDokkanEnv instance.
        obs_mode: "flat" or "dict".

    Returns:
        np.ndarray if obs_mode == "flat"
        dict[str, np.ndarray] if obs_mode == "dict"
    """
    obs_mode = obs_mode.lower()

    if obs_mode == "flat":
        return build_flat_observation(env)

    if obs_mode == "dict":
        return build_dict_observation(env)

    raise ValueError(f"Unknown obs_mode: {obs_mode}. Use 'flat' or 'dict'.")