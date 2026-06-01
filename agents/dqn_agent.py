import random
import numpy as np
import tensorflow as tf


class DQNAgent:
    """
    DQN agent.

    Simple DQN-like agent:
    - no replay buffer;
    - no target network;
    - one neural network approximating Q(s, a);
    - epsilon-greedy exploration;
    - online update after each environment step.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.model = self._build_model(learning_rate)

    def _build_model(self, learning_rate):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.obs_dim,)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(self.action_dim, activation="linear"),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
        )

        return model

    def act(self, obs, info=None, env=None, training=False):
        """
        Select an action using epsilon-greedy.

        During training:
        - with probability epsilon, choose a random action;
        - otherwise, choose the action with the highest predicted Q-value.

        During evaluation:
        - always choose the action with the highest predicted Q-value.
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        #obs_batch = np.expand_dims(obs, axis=0).astype(np.float32)
        #q_values = self.model.predict(obs_batch, verbose=0)[0]

        obs_batch = np.expand_dims(obs, axis=0).astype(np.float32)
        q_values = self.model(obs_batch, training=False).numpy()[0]

        return int(np.argmax(q_values))

    def update(self, obs, action, reward, next_obs, done):
        obs_batch = np.expand_dims(obs, axis=0).astype(np.float32)
        next_obs_batch = np.expand_dims(next_obs, axis=0).astype(np.float32)

        current_q_values = self.model.predict(obs_batch, verbose=0)
        next_q_values = self.model.predict(next_obs_batch, verbose=0)

        if done:
            target_value = reward
        else:
            target_value = reward + self.gamma * np.max(next_q_values[0])

        target_q_values = current_q_values.copy()
        target_q_values[0, action] = target_value

        history = self.model.fit(
            obs_batch,
            target_q_values,
            verbose=0,
            epochs=1,
        )

        loss = history.history["loss"][0]
        return float(loss)

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)