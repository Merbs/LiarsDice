import random

import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow import keras, one_hot
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.nn import softmax

from margam.player import Player
from margam.rl import MargamError, GameType
from margam.trainer import RLTrainer


class PolicyPlayer(Player):

    def __init__(self, *args, model=None, actor_critic=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor_critic = actor_critic
        self.model = load_model(model) if model else self.initialize_model()
        self.actor_critic = len(self.model.outputs) == 2  # may have loaded an AC model

    def get_move(self, state) -> int:
        eval_vector = self.game_handler.get_eval_vector(state)
        logits = self.model.predict_on_batch(eval_vector[np.newaxis, :])
        if self.actor_critic:
            logits, _ = logits
        move_probabilities = softmax(logits[0])
        action_options = state.legal_actions()
        move_probabilities = [move_probabilities[i] for i in action_options]
        selected_move = random.choices(action_options, weights=move_probabilities, k=1)[
            0
        ]
        return selected_move

    def initialize_model(self, show_model=True):
        eg_state = self.game_handler.game.new_initial_state()
        eg_input = self.game_handler.get_eval_vector(eg_state)
        nn_input = keras.Input(shape=eg_input.shape)

        if self.game_handler.game_type == GameType.TIC_TAC_TOE:
            nn_outputs = self.initialize_tic_tac_toe_model(nn_input)
        elif self.game_handler.game_type == GameType.CONNECT_FOUR:
            nn_outputs = self.initialize_connect_four_model(nn_input)
        elif self.game_handler.game_type == GameType.LIARS_DICE:
            nn_outputs = self.initialize_liars_dice_model(nn_input)
        else:
            raise MargamError(f"{self.game_handler.game_type} not implemented")

        model = keras.Model(
            inputs=nn_input, outputs=nn_outputs, name=f"PG_{self.game_handler.game_type.value}_model"
        )

        if show_model:
            model.summary()

        return model

    def initialize_tic_tac_toe_model(self, nn_input):

        input_flat = layers.Flatten()(nn_input)
        model_trunk_f = input_flat

        x = layers.Dense(32, activation="relu")(model_trunk_f)
        logits_output = layers.Dense(
            self.game_handler.game.num_distinct_actions(), activation="linear"
        )(x)
        nn_outputs = logits_output

        if self.actor_critic:
            x = layers.Dense(64, activation="relu")(model_trunk_f)
            state_value_output = layers.Dense(1, activation="linear")(x)
            nn_outputs = [logits_output, state_value_output]

        return nn_outputs

    def initialize_connect_four_model(self, nn_input):
        x = layers.Conv2D(64, 4)(nn_input)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        model_trunk_f = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(model_trunk_f)
        logits_output = layers.Dense(
            self.game_handler.game.num_distinct_actions(), activation="linear"
        )(x)
        nn_outputs = logits_output

        if self.actor_critic:
            x = layers.Dense(64, activation="relu")(model_trunk_f)
            state_value_output = layers.Dense(1, activation="linear")(x)
            nn_outputs = [logits_output, state_value_output]

        return nn_outputs

    def initialize_liars_dice_model(self, nn_input):
        model_trunk_f = layers.Flatten()(nn_input)
        x = layers.Dense(64, activation="relu")(model_trunk_f)
        logits_output = layers.Dense(
            self.game_handler.game.num_distinct_actions(), activation="linear"
        )(x)
        nn_outputs = logits_output

        if self.actor_critic:
            x = layers.Dense(64, activation="relu")(model_trunk_f)
            state_value_output = layers.Dense(1, activation="linear")(x)
            nn_outputs = [logits_output, state_value_output]

        return nn_outputs


class PolicyGradientTrainer(RLTrainer):

    algorithm = "Policy Gradient"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_transitions = []
        self.optimizer = Adam(learning_rate=self.LEARNING_RATE)
        self.mse_loss = MeanSquaredError()

    def initialize_agent(self):
        self.agent = PolicyPlayer(
            self.game_handler,
            model=None,
            actor_critic=self.ACTOR_CRITIC,
            name=f"{self.name}-agent",
        )
        return self.agent

    def get_unique_name(self) -> str:
        return f"PG-{self.game_handler.game_type.value}-{self.get_now_str()}"

    def initialize_training_stats(self):
        super().initialize_training_stats()
        self.episode_transitions = []

    def _train(self):

        # Cannot do tempral differencing without critic
        if not self.ACTOR_CRITIC:
            self.N_TD = -1

        if self.agent is None:
            self.agent = PolicyPlayer(
                name=f"PG-{self.get_now_str()}"
            )  # auto-build player?

        while self.episode_ind <= self.MAX_EPISODES:
            self.execute_episodic_training_step()

    def generate_episode_transitions(self):
        agent_transitions, opponent = super().generate_episode_transitions()
        agent_transitions = self.apply_temporal_difference(
            agent_transitions,
            self.DISCOUNT_RATE,
            n_td=self.N_TD,
        )
        if self.USE_SYMMETRY:
            agent_transitions = self.add_symmetries(agent_transitions)
        return agent_transitions, opponent

    def execute_episodic_training_step(self):
        agent_transitions, opponent = self.generate_episode_transitions()
        self.step += len(agent_transitions)
        self.episode_transitions.append(agent_transitions)

        self.reward_buffer.append(agent_transitions[-1].reward)
        self.reward_buffer_vs[opponent.name].append(agent_transitions[-1].reward)
        self.experience_buffer += agent_transitions
        if self.episode_ind % self.RECORD_EPISODES == 0:
            self.record_episode_statistics()

        self.save_checkpoint_model()

        # Don't start training the network until we have enough data
        if len(self.episode_transitions) >= self.BATCH_N_EPISODES:
            self.update_model()
            self.episode_transitions.clear()

    def update_model(self):
        """
        Execute a backpropagation optimization step on the agent's
        neural network and record statistics
        """

        training_data = [t for et in self.episode_transitions for t in et]
        prev_record_hist_ind = (
            self.step - len(training_data)
        ) // self.RECORD_HISTOGRAMS
        record_histograms = self.step // self.RECORD_HISTOGRAMS > prev_record_hist_ind
        prev_record_scalar_ind = (self.step - len(training_data)) // self.RECORD_SCALARS
        record_scalars = self.step // self.RECORD_SCALARS > prev_record_scalar_ind

        # Unpack training data
        selected_actions = np.array([trsn.action for trsn in training_data])
        selected_move_mask = one_hot(
            selected_actions, self.game_handler.game.num_distinct_actions()
        )
        x_train = np.array([trsn.state for trsn in training_data])
        rewards = np.array([trsn.reward for trsn in training_data]).astype("float32")
        action_legality = np.array(
            [trsn.legal_actions for trsn in training_data]
        ).astype("float32")

        with tf.GradientTape() as tape:

            # Generate logits
            logits = self.agent.model(x_train)

            # Mask logits for illegal moves
            # Illegal moves have large negative logits
            # With no dependence on model parameters
            large_neg_logits = -10 * np.ones(logits.shape)
            logits = tf.multiply(action_legality, logits) + tf.multiply(
                (1 - action_legality), large_neg_logits
            )

            if self.ACTOR_CRITIC:

                # Update rewards with value of future state
                if self.N_TD != -1:

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
                    _, resulting_state_values = self.agent.model(resulting_boards)
                    resulting_state_values = resulting_state_values[:, 0]
                    rewards = rewards + (self.DISCOUNT_RATE**self.N_TD) * np.multiply(
                        non_terminal_states, resulting_state_values
                    )

                logits, state_values = logits
                state_values = state_values[:, 0]
                state_loss = self.mse_loss(rewards, state_values)

            # Compute logits
            move_log_probs = tf.nn.log_softmax(logits)
            masked_log_probs = tf.multiply(move_log_probs, selected_move_mask)
            selected_log_probs = tf.reduce_sum(masked_log_probs, 1)
            obs_advantage = rewards
            if self.ACTOR_CRITIC:
                obs_advantage = rewards - tf.stop_gradient(state_values)
            expectation_loss = -tf.tensordot(
                obs_advantage, selected_log_probs, axes=1
            ) / len(selected_log_probs)

            # Entropy component of loss
            move_probs = tf.nn.softmax(logits)
            entropy_components = tf.multiply(move_probs, move_log_probs)
            entropy_each_state = -tf.reduce_sum(entropy_components, 1)
            entropy = tf.reduce_mean(entropy_each_state)
            entropy_loss = -self.ENTROPY_BETA * entropy

            # Sum the loss contributions
            loss = expectation_loss + entropy_loss
            if self.ACTOR_CRITIC:
                loss += state_loss * self.STATE_VALUE_BETA
                if self.writer and record_scalars:
                    self.writer.add_scalar("state-loss", state_loss.numpy(), self.step)

                if self.writer and record_histograms:
                    self.writer.add_histogram("state_value_train", rewards, self.step)
                    self.writer.add_histogram(
                        "state_value_pred", state_values.numpy(), self.step
                    )
                    state_value_error = state_values - rewards
                    self.writer.add_histogram(
                        "state_value_error", state_value_error.numpy(), self.step
                    )

        if self.writer and record_scalars:
            self.writer.add_scalar(
                "log-expect-loss", expectation_loss.numpy(), self.step
            )
            self.writer.add_scalar("entropy-loss", entropy_loss.numpy(), self.step)
            self.writer.add_scalar("loss", loss.numpy(), self.step)

        grads = tape.gradient(loss, self.agent.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.agent.model.trainable_variables))
        # grads = tape.gradient(loss,tape.watched_variables())
        # optimizer.apply_gradients(zip(grads, tape.watched_variables()))

        # calc KL-div
        if self.writer and record_scalars:
            new_logits_v = self.agent.model.predict_on_batch(x_train)
            if self.ACTOR_CRITIC:
                new_logits_v, _ = new_logits_v
            new_prob_v = tf.nn.softmax(new_logits_v)
            KL_EPSILON = 1e-7
            new_prob_v_kl = new_prob_v + KL_EPSILON
            move_probs_kl = move_probs + KL_EPSILON
            kl_div_v = -np.sum(
                np.log(new_prob_v_kl / move_probs_kl) * move_probs_kl, axis=1
            ).mean()
            self.writer.add_scalar(
                "Kullback-Leibler divergence", kl_div_v.item(), self.step
            )

        # Track gradient variance
        if self.writer and record_histograms:
            weights_and_biases_flat = np.concatenate(
                [v.numpy().flatten() for v in self.agent.model.variables]
            )
            self.writer.add_histogram(
                "weights and biases", weights_and_biases_flat, self.step
            )
            grads_flat = np.concatenate([v.numpy().flatten() for v in grads])
            self.writer.add_histogram("logits", logits.numpy().flatten(), self.step)
            self.writer.add_histogram("gradients", grads_flat, self.step)
            grad_rmse = np.sqrt(np.mean(grads_flat**2))
            self.writer.add_scalar("grad_rmse", grad_rmse, self.step)
            grad_max = np.abs(grads_flat).max()
            self.writer.add_scalar("grad_max", grad_max, self.step)
