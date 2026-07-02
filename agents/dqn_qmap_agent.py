import random
from collections import deque

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="MiniDokkanRL")
class BroadcastContext(tf.keras.layers.Layer):
    """
    Broadcast a global context vector over the spatial board feature map.

    Input:
        feature_map: (batch, board_size, board_size, channels)
        context:     (batch, context_dim)

    Output:
        context_map: (batch, board_size, board_size, context_dim)

    This allows the network to combine local board features with global battle
    information at every board cell.
    """

    def call(self, inputs):
        feature_map, context = inputs

        shape = tf.shape(feature_map)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        context_dim = tf.shape(context)[-1]

        context = tf.reshape(
            context,
            [batch_size, 1, 1, context_dim],
        )

        context = tf.tile(
            context,
            [1, height, width, 1],
        )

        return context


class QMapReplayBuffer:
    """
    Replay buffer for DQN Q-map agents.

    The buffer stores transitions:

        (state, action, reward, next_state, done)

    Each state is a dictionary:

        {
            "board":  np.ndarray, shape (N, N, num_orb_types),
            "global": np.ndarray, shape (global_dim,),
            "units":  np.ndarray, shape (num_units, unit_dim),
        }

    Since different board sizes have different tensor shapes, this buffer keeps
    one internal deque for each board size. During sampling, it selects one
    board-size bucket and returns a batch with consistent shapes.
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffers = {}

    def _get_board_size_key(self, state):
        board = state["board"]
        return int(board.shape[0])

    def add(self, state, action, reward, next_state, done):
        board_size = self._get_board_size_key(state)

        if board_size not in self.buffers:
            self.buffers[board_size] = deque(maxlen=self.capacity)

        self.buffers[board_size].append(
            (
                state,
                int(action),
                float(reward),
                next_state,
                float(done),
            )
        )

    def _stack_states(self, states):
        boards = np.stack(
            [state["board"] for state in states],
            axis=0,
        ).astype(np.float32)

        globals_ = np.stack(
            [state["global"] for state in states],
            axis=0,
        ).astype(np.float32)

        units = np.stack(
            [state["units"] for state in states],
            axis=0,
        ).astype(np.float32)

        return {
            "board": boards,
            "global": globals_,
            "units": units,
        }

    def sample(self, batch_size):
        valid_board_sizes = [
            board_size
            for board_size, buffer in self.buffers.items()
            if len(buffer) >= batch_size
        ]

        if not valid_board_sizes:
            raise ValueError("Not enough samples in any board-size bucket.")

        board_size = random.choice(valid_board_sizes)
        batch = random.sample(self.buffers[board_size], batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = self._stack_states(states)
        next_states = self._stack_states(next_states)

        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones, board_size

    def __len__(self):
        return sum(len(buffer) for buffer in self.buffers.values())

    def can_sample(self, batch_size):
        for buffer in self.buffers.values():
            if len(buffer) >= batch_size:
                return True

        return False


class DQNQMapAgent:
    """
    DQN agent with replay buffer, target network, and CNN Q-map output.

    Difference from the original DQNAgentV2:
    - the observation is structured instead of flat;
    - the board is processed by a CNN;
    - the network outputs a Q-map instead of a fixed Dense(action_dim).

    The Q-map has shape:

        (batch, board_size, board_size, num_units)

    For example:
        4x4 board -> 4 * 4 * 3 = 48 actions
        5x5 board -> 5 * 5 * 3 = 75 actions
        6x6 board -> 6 * 6 * 3 = 108 actions

    The network can therefore be applied to different square board sizes.
    """

    def __init__(
        self,
        global_dim,
        unit_dim,
        num_orb_types=6,
        num_units=3,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_size=50_000,
        batch_size=64,
        target_update_freq=250,
    ):
        self.global_dim = int(global_dim)
        self.unit_dim = int(unit_dim)
        self.num_orb_types = int(num_orb_types)
        self.num_units = int(num_units)

        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.update_counter = 0

        self.replay_buffer = QMapReplayBuffer(capacity=buffer_size)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.q_network = self._build_model()
        self.target_network = self._build_model()

        self.update_target_network()

    def _build_model(self):
        """
        Build a lightweight CNN Q-map network.

        Inputs:
            board:  (batch, N, N, num_orb_types)
            global: (batch, global_dim)
            units:  (batch, num_units, unit_dim)

        Output:
            q_map: (batch, N, N, num_units)

        The goal is to introduce a spatial inductive bias without making the model
        unnecessarily large.
        """

        board_input = tf.keras.layers.Input(
            shape=(None, None, self.num_orb_types),
            name="board",
        )

        global_input = tf.keras.layers.Input(
            shape=(self.global_dim,),
            name="global",
        )

        units_input = tf.keras.layers.Input(
            shape=(self.num_units, self.unit_dim),
            name="units",
        )

        # ------------------------------------------------------------
        # Lightweight board encoder.
        # The CNN learns local spatial patterns in the orb board.
        # ------------------------------------------------------------
        x = tf.keras.layers.Conv2D(
            16,
            kernel_size=3,
            padding="same",
            activation="relu",
        )(board_input)

        x = tf.keras.layers.Conv2D(
            32,
            kernel_size=3,
            padding="same",
            activation="relu",
        )(x)

        # ------------------------------------------------------------
        # Lightweight global battle-state encoder.
        # ------------------------------------------------------------
        g = tf.keras.layers.Dense(
            32,
            activation="relu",
        )(global_input)

        # ------------------------------------------------------------
        # Lightweight unit encoder.
        # The team size is fixed, so flattening the unit matrix is fine.
        # ------------------------------------------------------------
        u = tf.keras.layers.Flatten()(units_input)

        u = tf.keras.layers.Dense(
            32,
            activation="relu",
        )(u)

        # ------------------------------------------------------------
        # Combine global battle context and unit context.
        # ------------------------------------------------------------
        context = tf.keras.layers.Concatenate()([g, u])

        context = tf.keras.layers.Dense(
            32,
            activation="relu",
        )(context)

        context_map = BroadcastContext()([x, context])

        # ------------------------------------------------------------
        # Combine local board features with global context at each cell.
        # ------------------------------------------------------------
        h = tf.keras.layers.Concatenate(axis=-1)([x, context_map])

        h = tf.keras.layers.Conv2D(
            32,
            kernel_size=1,
            activation="relu",
        )(h)

        # ------------------------------------------------------------
        # Q-map output.
        # For each cell, output one Q-value for each unit.
        # ------------------------------------------------------------
        q_map = tf.keras.layers.Conv2D(
            self.num_units,
            kernel_size=1,
            activation="linear",
            name="q_map",
        )(h)

        model = tf.keras.Model(
            inputs={
                "board": board_input,
                "global": global_input,
                "units": units_input,
            },
            outputs=q_map,
        )

        return model

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def _obs_to_batch(self, obs):
        """
        Convert one structured observation into a batch of size 1.
        """
        return {
            "board": np.expand_dims(obs["board"], axis=0).astype(np.float32),
            "global": np.expand_dims(obs["global"], axis=0).astype(np.float32),
            "units": np.expand_dims(obs["units"], axis=0).astype(np.float32),
        }

    
    def _q_map_to_flat(self, q_map):
        """
        Convert a Q-map to a flat action vector.

        Input:
            q_map shape = (batch, N, N, num_units)

        Environment action order:
            action = unit_idx * num_cells + cell_idx

        Therefore, we first move the unit dimension before the board dimensions:

            (batch, N, N, U) -> (batch, U, N, N)

        and then flatten.
        """
        q_unit_first = tf.transpose(q_map, perm=[0, 3, 1, 2])
        q_flat = tf.reshape(q_unit_first, [tf.shape(q_map)[0], -1])

        return q_flat

    def _get_action_dim_from_obs(self, obs):
        board_size = int(obs["board"].shape[0])
        return self.num_units * board_size * board_size

    def act(self, obs, info=None, env=None, training=False):
        """
        Select an action using epsilon-greedy.

        During training:
            random action with probability epsilon.

        During evaluation:
            greedy action according to the Q-map network.
        """

        if env is not None:
            action_dim = env.action_space.n
        else:
            action_dim = self._get_action_dim_from_obs(obs)

        if training and random.random() < self.epsilon:
            return random.randrange(action_dim)

        obs_batch = self._obs_to_batch(obs)

        q_map = self.q_network(
            obs_batch,
            training=False,
        )

        q_values = self._q_map_to_flat(q_map).numpy()[0]

        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def update(self):
        if not self.replay_buffer.can_sample(self.batch_size):
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            board_size,
        ) = self.replay_buffer.sample(self.batch_size)

        states = {
            "board": tf.convert_to_tensor(states["board"], dtype=tf.float32),
            "global": tf.convert_to_tensor(states["global"], dtype=tf.float32),
            "units": tf.convert_to_tensor(states["units"], dtype=tf.float32),
        }

        next_states = {
            "board": tf.convert_to_tensor(next_states["board"], dtype=tf.float32),
            "global": tf.convert_to_tensor(next_states["global"], dtype=tf.float32),
            "units": tf.convert_to_tensor(next_states["units"], dtype=tf.float32),
        }

        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # ------------------------------------------------------------
        # Compute DQN target using the target network.
        # ------------------------------------------------------------
        next_q_map = self.target_network(
            next_states,
            training=False,
        )

        next_q_values = self._q_map_to_flat(next_q_map)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)

        targets = rewards + self.gamma * max_next_q_values * (1.0 - dones)

        # ------------------------------------------------------------
        # Update Q-network.
        # ------------------------------------------------------------
        with tf.GradientTape() as tape:
            q_map = self.q_network(
                states,
                training=True,
            )

            q_values = self._q_map_to_flat(q_map)

            action_masks = tf.one_hot(
                actions,
                depth=tf.shape(q_values)[1],
            )

            selected_q_values = tf.reduce_sum(
                q_values * action_masks,
                axis=1,
            )

            loss = self.loss_fn(
                targets,
                selected_q_values,
            )

        gradients = tape.gradient(
            loss,
            self.q_network.trainable_variables,
        )

        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )

        self.update_counter += 1

        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()

        return {
            "loss": float(loss.numpy()),
            "avg_target": float(tf.reduce_mean(targets).numpy()),
            "avg_selected_q": float(tf.reduce_mean(selected_q_values).numpy()),
            "epsilon": float(self.epsilon),
            "board_size": int(board_size),
        }

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path):
        self.q_network.save(path)

    def load(self, path):
        self.q_network = tf.keras.models.load_model(path)
        self.target_network = tf.keras.models.clone_model(self.q_network)
        self.target_network.set_weights(self.q_network.get_weights())