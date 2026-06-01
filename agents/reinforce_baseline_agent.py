import numpy as np
import tensorflow as tf


class ReinforceBaselineAgent:
    """
    REINFORCE with learned baseline.

    The agent has:
    - a policy network pi(a | s), which outputs action probabilities;
    - a value network V(s), which estimates the expected return from a state.

    The policy is updated using the advantage:

        A_t = G_t - V(s_t)

    where:
    - G_t is the Monte Carlo return from time t;
    - V(s_t) is the baseline prediction.

    This reduces the variance of standard REINFORCE.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        gamma=0.99,
        policy_learning_rate=1e-4,
        value_learning_rate=5e-4,
        entropy_coef=0.001,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.policy_network = self._build_policy_network()
        self.value_network = self._build_value_network()

        self.policy_optimizer = tf.keras.optimizers.Adam(
            learning_rate=policy_learning_rate
        )
        self.value_optimizer = tf.keras.optimizers.Adam(
            learning_rate=value_learning_rate
        )

    def _build_policy_network(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.obs_dim,)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(self.action_dim, activation="softmax"),
            ]
        )

    def _build_value_network(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.obs_dim,)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(1, activation="linear"),
            ]
        )

    def act(self, obs, info=None, env=None, training=False):
        """
        Select an action.

        During training:
            sample from the policy distribution.

        During evaluation:
            take the most probable action.
        """
        obs_batch = np.expand_dims(obs, axis=0).astype(np.float32)
        action_probs = self.policy_network(
            obs_batch,
            training=False,
        ).numpy()[0]

        if training:
            action = np.random.choice(self.action_dim, p=action_probs)
        else:
            action = int(np.argmax(action_probs))

        return int(action)

    def compute_returns(self, rewards):
        """
        Compute discounted Monte Carlo returns.

        G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
        """
        returns = []
        G = 0.0

        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.append(G)

        returns.reverse()

        return np.array(returns, dtype=np.float32)

    def update_from_episode(self, states, actions, returns):
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        returns = np.array(returns, dtype=np.float32)

        # ------------------------------------------------------------
        # Compute current value predictions and advantages.
        # ------------------------------------------------------------
        values = self.value_network(states, training=False)
        values = tf.squeeze(values, axis=1).numpy()

        advantages = returns - values

        advantages = np.clip(advantages, -5.0, 5.0).astype(np.float32)

        # ------------------------------------------------------------
        # Policy network update.
        # ------------------------------------------------------------
        with tf.GradientTape() as tape:
            action_probs = self.policy_network(states, training=True)

            action_masks = tf.one_hot(actions, self.action_dim)
            selected_action_probs = tf.reduce_sum(
                action_probs * action_masks,
                axis=1,
            )

            log_probs = tf.math.log(selected_action_probs + 1e-8)

            policy_loss = -tf.reduce_mean(log_probs * advantages)

            entropy = -tf.reduce_mean(
                tf.reduce_sum(
                    action_probs * tf.math.log(action_probs + 1e-8),
                    axis=1,
                )
            )

            total_policy_loss = policy_loss - self.entropy_coef * entropy

        policy_gradients = tape.gradient(
            total_policy_loss,
            self.policy_network.trainable_variables,
        )
        self.policy_optimizer.apply_gradients(
            zip(policy_gradients, self.policy_network.trainable_variables)
        )

        # ------------------------------------------------------------
        # Value network update.
        # ------------------------------------------------------------
        with tf.GradientTape() as tape:
            predicted_values = self.value_network(states, training=True)
            predicted_values = tf.squeeze(predicted_values, axis=1)

            value_loss = tf.reduce_mean(
                tf.square(returns - predicted_values)
            )

        value_gradients = tape.gradient(
            value_loss,
            self.value_network.trainable_variables,
        )
        self.value_optimizer.apply_gradients(
            zip(value_gradients, self.value_network.trainable_variables)
        )

        return {
            "policy_loss": float(policy_loss.numpy()),
            "value_loss": float(value_loss.numpy()),
            "entropy": float(entropy.numpy()),
            "avg_advantage": float(np.mean(advantages)),
            "avg_value": float(np.mean(values)),
            "avg_return_target": float(np.mean(returns)),
            "avg_raw_advantage": float(np.mean(returns - values)),
            "std_raw_advantage": float(np.std(returns - values)),
            "max_action_prob": float(np.mean(np.max(action_probs.numpy(), axis=1))),
        }

    def save(self, path_prefix):
        self.policy_network.save(path_prefix + "_policy.keras")
        self.value_network.save(path_prefix + "_value.keras")

    def load(self, path_prefix):
        self.policy_network = tf.keras.models.load_model(
            path_prefix + "_policy.keras"
        )
        self.value_network = tf.keras.models.load_model(
            path_prefix + "_value.keras"
        )