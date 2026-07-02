import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="MiniDokkanRL")
class BroadcastContext(tf.keras.layers.Layer):
    """
    Broadcast a global context vector over a spatial feature map.

    Input:
        feature_map: (batch, board_size, board_size, channels)
        context:     (batch, context_dim)

    Output:
        context_map: (batch, board_size, board_size, context_dim)
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


class ReinforceMapAgent:
    """
    REINFORCE with learned baseline and dynamic policy map.

    This agent uses structured observations:

        obs = {
            "board":  np.ndarray, shape (N, N, num_orb_types),
            "global": np.ndarray, shape (global_dim,),
            "units":  np.ndarray, shape (num_units, unit_dim),
        }

    The policy network outputs a spatial action-logit map:

        logits_map shape = (batch, N, N, num_units)

    This makes the policy compatible with different square board sizes:

        4x4 -> 4 * 4 * 3 = 48 actions
        5x5 -> 5 * 5 * 3 = 75 actions
        6x6 -> 6 * 6 * 3 = 108 actions

    The value network estimates:

        V(s) -> scalar

    This is still REINFORCE, so updates are based on Monte Carlo returns.
    """

    def __init__(
        self,
        global_dim,
        unit_dim,
        num_orb_types=6,
        num_units=3,
        gamma=0.99,
        policy_learning_rate=1e-4,
        value_learning_rate=5e-4,
        entropy_coef=0.001,
        batch_size=64,
        normalize_advantages=True,
    ):
        self.global_dim = int(global_dim)
        self.unit_dim = int(unit_dim)
        self.num_orb_types = int(num_orb_types)
        self.num_units = int(num_units)

        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.batch_size = int(batch_size)
        self.normalize_advantages = bool(normalize_advantages)

        self.policy_network = self._build_policy_network()
        self.value_network = self._build_value_network()

        self.policy_optimizer = tf.keras.optimizers.Adam(
            learning_rate=policy_learning_rate
        )
        self.value_optimizer = tf.keras.optimizers.Adam(
            learning_rate=value_learning_rate
        )

    def _build_board_encoder(self, board_input):
        """
        Lightweight CNN encoder for the orb board.

        The input board size is dynamic:
            (None, None, num_orb_types)

        This means the same model can process 4x4, 5x5, 6x6, etc.
        """
        x = tf.keras.layers.Conv2D(
            8,
            kernel_size=3,
            padding="same",
            activation="relu",
        )(board_input)

        x = tf.keras.layers.Conv2D(
            16,
            kernel_size=3,
            padding="same",
            activation="relu",
        )(x)

        return x

    def _build_context_encoder(self, global_input, units_input):
        """
        Encode non-spatial information:
        - global battle state;
        - fixed team/unit features.
        """
        g = tf.keras.layers.Dense(
            16,
            activation="relu",
        )(global_input)

        u = tf.keras.layers.Flatten()(units_input)

        u = tf.keras.layers.Dense(
            16,
            activation="relu",
        )(u)

        context = tf.keras.layers.Concatenate()([g, u])

        context = tf.keras.layers.Dense(
            16,
            activation="relu",
        )(context)

        return context

    def _build_policy_network(self):
        """
        Build the policy network.

        Output:
            logits_map with shape (batch, N, N, num_units)

        Important:
            The output is logits, not softmax probabilities.
            Softmax is applied later after flattening the map into the
            current action space.
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

        x = self._build_board_encoder(board_input)
        context = self._build_context_encoder(global_input, units_input)

        context_map = BroadcastContext()([x, context])

        h = tf.keras.layers.Concatenate(axis=-1)([x, context_map])

        h = tf.keras.layers.Conv2D(
            16,
            kernel_size=1,
            activation="relu",
        )(h)

        logits_map = tf.keras.layers.Conv2D(
            self.num_units,
            kernel_size=1,
            activation="linear",
            name="policy_logits_map",
        )(h)

        model = tf.keras.Model(
            inputs={
                "board": board_input,
                "global": global_input,
                "units": units_input,
            },
            outputs=logits_map,
        )

        return model

    def _build_value_network(self):
        """
        Build the value baseline network.

        Since V(s) is a scalar, the board branch uses global pooling to make
        the representation independent from board size.
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

        x = self._build_board_encoder(board_input)

        # Convert variable-size board features into a fixed-size vector.
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        context = self._build_context_encoder(global_input, units_input)

        h = tf.keras.layers.Concatenate()([x, context])

        h = tf.keras.layers.Dense(
            32,
            activation="relu",
        )(h)

        h = tf.keras.layers.Dense(
            32,
            activation="relu",
        )(h)

        value = tf.keras.layers.Dense(
            1,
            activation="linear",
            name="value",
        )(h)

        model = tf.keras.Model(
            inputs={
                "board": board_input,
                "global": global_input,
                "units": units_input,
            },
            outputs=value,
        )

        return model

    def _obs_to_batch(self, obs):
        """
        Convert one structured observation into a batch of size 1.
        """
        return {
            "board": np.expand_dims(obs["board"], axis=0).astype(np.float32),
            "global": np.expand_dims(obs["global"], axis=0).astype(np.float32),
            "units": np.expand_dims(obs["units"], axis=0).astype(np.float32),
        }

    def _states_to_batch(self, states):
        """
        Convert a list of structured observations into a batch.

        All states in the batch must have the same board size.
        """
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

    def _logits_map_to_flat(self, logits_map):
        """
        Convert a policy-logits map into flat action logits.

        Input:
            logits_map shape = (batch, N, N, num_units)

        Environment action order:
            action = unit_idx * num_cells + cell_idx

        Therefore:
            (batch, N, N, U) -> (batch, U, N, N) -> flat
        """
        logits_unit_first = tf.transpose(
            logits_map,
            perm=[0, 3, 1, 2],
        )

        flat_logits = tf.reshape(
            logits_unit_first,
            [tf.shape(logits_map)[0], -1],
        )

        return flat_logits

    def act(self, obs, info=None, env=None, training=False):
        """
        Select an action.

        During training:
            sample from the policy distribution.

        During evaluation:
            take the most probable action.
        """
        obs_batch = self._obs_to_batch(obs)

        logits_map = self.policy_network(
            obs_batch,
            training=False,
        )

        flat_logits = self._logits_map_to_flat(logits_map)

        if training:
            action = tf.random.categorical(
                flat_logits,
                num_samples=1,
            )[0, 0]
        else:
            action = tf.argmax(
                flat_logits[0],
                axis=-1,
            )

        return int(action.numpy())

    def compute_returns(self, rewards):
        """
        Compute discounted Monte Carlo returns.

        G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
        """
        returns = []
        G = 0.0

        for reward in reversed(rewards):
            G = float(reward) + self.gamma * G
            returns.append(G)

        returns.reverse()

        return np.array(returns, dtype=np.float32)

    def update_from_batch(self, states, actions, returns):
        """
        Update policy and value networks from a batch of on-policy transitions.

        This method is designed to be comparable to DQN's batch update:
            batch_size = 64 transitions by default

        Difference:
            DQN samples transitions from replay memory.
            REINFORCE uses on-policy transitions collected from recent episodes.
        """
        states_batch = self._states_to_batch(states)

        actions = np.array(actions, dtype=np.int32)
        returns = np.array(returns, dtype=np.float32)

        actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)
        returns_tf = tf.convert_to_tensor(returns, dtype=tf.float32)

        # ------------------------------------------------------------
        # Compute current value predictions and advantages.
        # ------------------------------------------------------------
        values = self.value_network(
            states_batch,
            training=False,
        )

        values = tf.squeeze(values, axis=1).numpy()

        raw_advantages = returns - values

        if self.normalize_advantages:
            advantages = (
                raw_advantages - np.mean(raw_advantages)
            ) / (np.std(raw_advantages) + 1e-8)
        else:
            advantages = raw_advantages

        advantages = np.clip(
            advantages,
            -5.0,
            5.0,
        ).astype(np.float32)

        advantages_tf = tf.convert_to_tensor(
            advantages,
            dtype=tf.float32,
        )

        # ------------------------------------------------------------
        # Policy network update.
        # ------------------------------------------------------------
        with tf.GradientTape() as tape:
            logits_map = self.policy_network(
                states_batch,
                training=True,
            )

            flat_logits = self._logits_map_to_flat(logits_map)

            action_probs = tf.nn.softmax(
                flat_logits,
                axis=1,
            )

            log_action_probs = tf.nn.log_softmax(
                flat_logits,
                axis=1,
            )

            action_masks = tf.one_hot(
                actions_tf,
                depth=tf.shape(flat_logits)[1],
            )

            selected_log_probs = tf.reduce_sum(
                log_action_probs * action_masks,
                axis=1,
            )

            policy_loss = -tf.reduce_mean(
                selected_log_probs * advantages_tf
            )

            entropy = -tf.reduce_mean(
                tf.reduce_sum(
                    action_probs * log_action_probs,
                    axis=1,
                )
            )

            total_policy_loss = policy_loss - self.entropy_coef * entropy

        policy_gradients = tape.gradient(
            total_policy_loss,
            self.policy_network.trainable_variables,
        )

        self.policy_optimizer.apply_gradients(
            zip(
                policy_gradients,
                self.policy_network.trainable_variables,
            )
        )

        # ------------------------------------------------------------
        # Value network update.
        # ------------------------------------------------------------
        with tf.GradientTape() as tape:
            predicted_values = self.value_network(
                states_batch,
                training=True,
            )

            predicted_values = tf.squeeze(
                predicted_values,
                axis=1,
            )

            value_loss = tf.reduce_mean(
                tf.square(returns_tf - predicted_values)
            )

        value_gradients = tape.gradient(
            value_loss,
            self.value_network.trainable_variables,
        )

        self.value_optimizer.apply_gradients(
            zip(
                value_gradients,
                self.value_network.trainable_variables,
            )
        )

        return {
            "policy_loss": float(policy_loss.numpy()),
            "value_loss": float(value_loss.numpy()),
            "entropy": float(entropy.numpy()),
            "avg_advantage": float(np.mean(advantages)),
            "avg_value": float(np.mean(values)),
            "avg_return_target": float(np.mean(returns)),
            "avg_raw_advantage": float(np.mean(raw_advantages)),
            "std_raw_advantage": float(np.std(raw_advantages)),
            "max_action_prob": float(
                np.mean(np.max(action_probs.numpy(), axis=1))
            ),
        }

    def save(self, path_prefix):
        """
        Save policy and value networks.

        Example:
            path_prefix = "models/reinforce_map_10k"
            creates:
                models/reinforce_map_10k_policy.keras
                models/reinforce_map_10k_value.keras
        """
        self.policy_network.save(path_prefix + "_policy.keras")
        self.value_network.save(path_prefix + "_value.keras")

    def load(self, path_prefix):
        """
        Load policy and value networks.
        """
        self.policy_network = tf.keras.models.load_model(
            path_prefix + "_policy.keras",
            custom_objects={"BroadcastContext": BroadcastContext},
        )

        self.value_network = tf.keras.models.load_model(
            path_prefix + "_value.keras",
            custom_objects={"BroadcastContext": BroadcastContext},
        )