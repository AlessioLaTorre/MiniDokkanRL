import random
from collections import deque

import numpy as np
import tensorflow as tf


class ReplayBuffer:
    """
    Replay buffer for DQN.

    It stores past transitions:
        (state, action, reward, next_state, done)

    During training, the agent samples random mini-batches from this buffer.
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgentV2:
    """
    DQN agent with replay buffer and target network.

    Compared to the minimal DQN:
    - it stores past transitions in a replay buffer;
    - it trains on random batches of transitions;
    - it uses a target network to compute more stable Bellman targets.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_size=50_000,
        batch_size=64,
        target_update_freq=250,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.update_counter = 0

        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.q_network = self._build_model(learning_rate)
        self.target_network = self._build_model(learning_rate)

        self.update_target_network()

    def _build_model(self, learning_rate):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.obs_dim,)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(self.action_dim, activation="linear"),
            ]
        )
        return model

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, obs, info=None, env=None, training=False):
        """
        Select an action using epsilon-greedy.

        During training:
            random action with probability epsilon.

        During evaluation:
            greedy action according to Q-network.
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        obs_batch = np.expand_dims(obs, axis=0).astype(np.float32)
        q_values = self.q_network(obs_batch, training=False).numpy()[0]

        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        next_q_values = self.target_network(next_states, training=False)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)

        targets = rewards + self.gamma * max_next_q_values * (1.0 - dones)

        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)

            action_masks = tf.one_hot(actions, self.action_dim)
            selected_q_values = tf.reduce_sum(q_values * action_masks, axis=1)

            loss = self.loss_fn(targets, selected_q_values)

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )

        self.update_counter += 1

        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss.numpy())

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path):
        self.q_network.save(path)

    def load(self, path):
        self.q_network = tf.keras.models.load_model(
            path,
        )
        self.target_network = tf.keras.models.clone_model(self.q_network)
        self.target_network.set_weights(self.q_network.get_weights())

