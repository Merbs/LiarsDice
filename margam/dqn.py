import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow import keras, one_hot
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from margam.player import Player
from margam.rl import MargamError, GameType
from margam.trainer import RLTrainer


class DeulingLayer(layers.Layer):
    def build(self):
        pass

    def call(self, q_values, sv):
        return q_values - tf.math.reduce_mean(q_values, axis=1, keepdims=True) + sv

class DQNPlayer(Player):
    """
    Agent that uses a neural network to compute
    Q(s,a) values for each action at a given state,
    then chooses the highest value as the action.

    `random_weight` gives a change to perform
    a random action instead for better exploration
    while training
    """

    def __init__(self, *args, model=None, deuling=True, random_weight=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.deuling = deuling
        self.model = load_model(model) if model else self.initialize_model()
        self.random_weight = random_weight

    def get_move(self, state) -> int:
        if np.random.random() < self.random_weight:
            return random.choice(state.legal_actions())

        state_for_cov = self.game_handler.get_eval_vector(state)
        q_values = self.model.predict_on_batch(state_for_cov[np.newaxis, :])[0]
        q_values = q_values.astype(float)
        for i, _ in enumerate(q_values):
            if i not in state.legal_actions():
                q_values[i] = -np.inf
        max_q_ind = np.argmax(q_values)

        return max_q_ind

    def initialize_model(self, show_model=True):
        """
        Construct neural network used by agent
        """
        eg_state = self.game_handler.game.new_initial_state()
        eg_input = self.game_handler.get_eval_vector(eg_state)
        nn_input = keras.Input(shape=eg_input.shape)

        if self.game_handler.game_type == GameType.TIC_TAC_TOE:
            q_values = self.initialize_tic_tac_toe_model(nn_input)
        elif self.game_handler.game_type == GameType.CONNECT_FOUR:
            q_values = self.initialize_connect_four_model(nn_input)
        else:
            raise MargamError(f"{self.game_handler.game_type.value} not implemented for DQN")

        model = keras.Model(inputs=nn_input, outputs=q_values, name="DQN-model")
        if show_model:
            model.summary()
        return model

    def initialize_tic_tac_toe_model(self, nn_input):
        input_flat = layers.Flatten()(nn_input)
        x = layers.Dense(32, activation="relu")(input_flat)
        q_values = layers.Dense(self.game_handler.game.num_distinct_actions(), activation="linear")(x)

        # Deuling DQN adds a second column to the neural net that
        # computes state value V(s) and interprets the Q
        # values as advantage of that action in that state
        # Q(s,a) = A(s,a) + V(s)
        # Final output is the same so it is interoperable with vanilla DQN
        if self.deuling:
            x_sv = layers.Dense(32, activation="relu")(input_flat)
            sv = layers.Dense(1, activation="linear")(x_sv)
            q_values = DeulingLayer()(q_values,sv)
        return q_values

    def initialize_connect_four_model(self, nn_input):
        x = layers.Conv2D(64, 4)(nn_input)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        q_values = layers.Dense(self.game_handler.game.num_distinct_actions(), activation="linear")(x)
        if self.deuling:
            x_sv = layers.Dense(32, activation="relu")(x)
            sv = layers.Dense(1, activation="linear")(x_sv)
            q_values = DeulingLayer()(q_values,sv)
        return q_values


class DQNTrainer(RLTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_network = keras.models.clone_model(self.agent.model)
        self.target_network.set_weights(self.agent.model.get_weights())

    def get_unique_name(self) -> str:
        return f"DQN-{self.game_handler.game_type.value}-{self.get_now_str()}"

    def sample_experience_buffer(self, batch_size):
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        return [self.experience_buffer[idx] for idx in indices]

    def initialize_agent(self):
        self.agent = DQNPlayer(
            self.game_handler,
            model=None,
            deuling=self.DEULING_DQN,
            name=f"{self.name}-agent",
        )
        return self.agent

    def initialize_training_stats(self):
        super().initialize_training_stats()
        self.mse_loss = MeanSquaredError()
        self.optimizer = Adam(learning_rate=self.LEARNING_RATE)
        self.experience_buffer = deque(maxlen=self.REPLAY_SIZE)

        if self.RECORD_HISTOGRAMS % self.SYNC_TARGET_NETWORK != 0:
            raise MargamError("SYNC_TARGET_NETWORK must divide RECORD_HISTOGRAMS")

    def _train(self):
        """
        Perform the training loop that runs episodes
        and uses the data to train a DQN player
        """

        while self.episode_ind <= self.MAX_EPISODES:
            self.execute_training_step()

    def execute_training_step(self):

        self.agent.random_weight = max(
            self.EPSILON_FINAL,
            self.EPSILON_START - self.step / self.EPSILON_DECAY_LAST_FRAME,
        )

        agent_transitions, opponent = self.generate_episode_transitions()
        if self.USE_SYMMETRY:
            agent_transitions = self.add_symmetries(agent_transitions)

        self.experience_buffer += agent_transitions
        self.reward_buffer.append(agent_transitions[-1].reward)
        self.reward_buffer_vs[opponent.name].append(agent_transitions[-1].reward)
        if self.writer and self.episode_ind % self.RECORD_EPISODES == 0:
            self.record_episode_statistics()

        if self.writer and self.agent.random_weight > self.EPSILON_FINAL:
            self.writer.add_scalar("epsilon", self.agent.random_weight, self.step)

        self.save_checkpoint_model()

        for transition in agent_transitions:
            self.step += 1
            self.experience_buffer.append(transition)

            # Don't start training the network until we have enough data
            if len(self.experience_buffer) < self.REPLAY_START_SIZE:
                continue

            # Get training data
            training_data = self.sample_experience_buffer(self.BATCH_SIZE)
            self.update_neural_network(training_data)

    def update_neural_network(self, training_data):
        """
        Perform one step of weight update
        """

        x_train = np.array([trsn.state for trsn in training_data])

        # Bellman equation part
        # Take maximum Q(s',a') of board states we end up in
        non_terminal_states = np.array(
            [trsn.next_state is not None for trsn in training_data]
        )
        resulting_boards = np.array(
            [
                (
                    trsn.next_state
                    if trsn.next_state is not None
                    else np.zeros(trsn.state.shape)
                )
                for trsn in training_data
            ]
        )

        # Double DQN - Use on policy network to choose best move
        #   and target network to evaluate the Q-value
        resulting_board_q_target = self.target_network.predict_on_batch(
            resulting_boards
        )
        if self.DOUBLE_DQN:
            resulting_board_q_on_policy = self.agent.model.predict_on_batch(resulting_boards)
            max_move_inds_on_policy = resulting_board_q_on_policy.argmax(axis=1)
            on_policy_move_mask = tf.one_hot(
                max_move_inds_on_policy, depth=resulting_board_q_on_policy.shape[1]
            )
            target_qs_for_on_policy_moves = tf.multiply(
                resulting_board_q_target, on_policy_move_mask
            )
            max_qs = tf.reduce_sum(target_qs_for_on_policy_moves, 1)
        # Single DQN - Take max target network Q to be max Q
        else:
            max_qs = np.max(resulting_board_q_target, axis=1)

        rewards = np.array([trsn.reward for trsn in training_data])
        q_to_train_single_values = rewards + (
            self.DISCOUNT_RATE**self.N_TD
        ) * np.multiply(non_terminal_states, max_qs)

        # Needed for our mask
        selected_actions = [trsn.action for trsn in training_data]
        selected_action_mask = one_hot(
            selected_actions, resulting_board_q_target.shape[1]
        )

        # Compute MSE loss based on chosen move values only
        with tf.GradientTape() as tape:
            predicted_q_values = self.agent.model(x_train)
            predicted_q_values = tf.multiply(predicted_q_values, selected_action_mask)
            predicted_q_values = tf.reduce_sum(predicted_q_values, 1)
            q_prediction_errors = predicted_q_values - q_to_train_single_values
            loss_value = self.mse_loss(q_to_train_single_values, predicted_q_values)

        weights_and_biases_flat_before_update = np.concatenate(
            [v.numpy().flatten() for v in self.agent.model.variables]
        )

        grads = tape.gradient(loss_value, self.agent.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.agent.model.trainable_variables))

        # Record prediction error
        if self.writer:
            self.writer.add_scalar("loss", loss_value.numpy(), self.step)
        if self.writer and self.step % self.RECORD_HISTOGRAMS == 0:
            self.writer.add_histogram("q-predicted", predicted_q_values.numpy(), self.step)
            self.writer.add_histogram("q-train", q_to_train_single_values, self.step)
            self.writer.add_histogram("q-error", q_prediction_errors.numpy(), self.step)

            weights_and_biases_flat = np.concatenate(
                [v.numpy().flatten() for v in self.agent.model.variables]
            )
            self.writer.add_histogram("weights and biases", weights_and_biases_flat, self.step)

            # Track gradient variance
            grads_flat = np.concatenate([v.numpy().flatten() for v in grads])
            self.writer.add_histogram("gradients", grads_flat, self.step)
            grad_rmse = np.sqrt(np.mean(grads_flat**2))
            self.writer.add_scalar("grad_rmse", grad_rmse, self.step)
            grad_max = np.abs(grads_flat).max()
            self.writer.add_scalar("grad_max", grad_max, self.step)

            weights_and_biases_delta = (
                weights_and_biases_flat - weights_and_biases_flat_before_update
            )
            self.writer.add_histogram("weight-bias-updates", weights_and_biases_delta, self.step)

        # Update policy
        if self.step % self.SYNC_TARGET_NETWORK == 0:
            if self.step % self.RECORD_HISTOGRAMS == 0:
                weights_and_biases_flat = np.concatenate(
                    [v.numpy().flatten() for v in self.agent.model.variables]
                )
                weights_and_biases_target = np.concatenate(
                    [v.numpy().flatten() for v in self.target_network.variables]
                )
                target_parameter_updates = (
                    weights_and_biases_flat - weights_and_biases_target
                )
                if self.writer:
                    self.writer.add_histogram(
                        "target parameter updates", target_parameter_updates, self.step
                    )

            self.target_network.set_weights(self.agent.model.get_weights())
