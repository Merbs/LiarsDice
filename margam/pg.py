import itertools
import random
from collections import deque
from enum import Enum
import yaml

import click
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorboardX import SummaryWriter
from tensorflow import keras, one_hot
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.nn import softmax
from tqdm import tqdm

import pyspiel

from c4lib.player import ColumnSpammer, MiniMax, Player, RandomPlayer
from c4lib.utils import Connect4Error
from c4lib.utils import (
    get_training_and_viewing_state,
    record_episode_statistics,
    generate_episode_transitions,
    get_now_str,
)


class PolicyPlayer(Player):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None

    def get_move(self, game, state) -> int:
        state_for_cov, _ = get_training_and_viewing_state(game, state)
        logits = self.model.predict_on_batch(state_for_cov[np.newaxis, :])
        if len(self.model.outputs) == 2:  # for actor-critic
            logits, _ = logits
        move_probabilities = softmax(logits[0])
        action_options = state.legal_actions()
        move_probabilities = [move_probabilities[i] for i in action_options]
        selected_move = random.choices(action_options, weights=move_probabilities, k=1)[
            0
        ]
        return selected_move

class ConservativePlayer(Player):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_move(self, game, state) -> int:
        """
        Call with 33% probability. Otherwise take
        most conservative bet
        """
        action_options = state.legal_actions()
        if np.random.uniform(low=0.0, high=1.0) < 0.33:
            return action_options[-1]
        return action_options[0]


def initialize_model(game_type, hp, show_model=True):
    game = pyspiel.load_game(game_type,{"numdice":5})
    state = game.new_initial_state()
    #state_np_for_cov, human_view_state = get_training_and_viewing_state(game, state)
    nn_input = keras.Input(shape=(20,))

    if game_type == "tic_tac_toe":
        input_flat = layers.Flatten()(nn_input)
        model_trunk_f = input_flat

        x = layers.Dense(32, activation="relu")(model_trunk_f)
        logits_output = layers.Dense(game.num_distinct_actions(), activation="linear")(
            x
        )
        nn_outputs = logits_output

        if hp["ACTOR_CRITIC"]:
            x = layers.Dense(64, activation="relu")(model_trunk_f)
            state_value_output = layers.Dense(1, activation="linear")(x)
            nn_outputs = [logits_output, state_value_output]

    elif game_type == "connect_four":
        x = layers.Conv2D(64,4)(nn_input)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)
        model_trunk_f = layers.Flatten()(x)
        x = layers.Dense(64,activation="relu")(model_trunk_f)
        logits_output = layers.Dense(game.num_distinct_actions(), activation="linear")(x)
        nn_outputs = logits_output

        if hp["ACTOR_CRITIC"]:
            x = layers.Dense(64, activation="relu")(model_trunk_f)
            state_value_output = layers.Dense(1, activation="linear")(x)
            nn_outputs = [logits_output, state_value_output]

    elif game_type == "liars_dice":
        model_trunk_f = layers.Flatten()(nn_input)
        x = layers.Dense(64,activation="relu")(model_trunk_f)
        logits_output = layers.Dense(
            61, activation="linear"
            )(x)
        nn_outputs = logits_output

        if hp["ACTOR_CRITIC"]:
            x = layers.Dense(64, activation="relu")(model_trunk_f)
            state_value_output = layers.Dense(1, activation="linear")(x)
            nn_outputs = [logits_output, state_value_output]

    else:
        raise Connect4Error(f"{game_type} not implemented")

    model = keras.Model(inputs=nn_input, outputs=nn_outputs, name="policy-model")

    if show_model:
        model.summary()

    return model


def train_pg(game_type, hp):

    # Cannot do tempral differencing without critic
    if not hp["ACTOR_CRITIC"]:
        hp["N_TD"] = -1

    # Intialize players

    agent = PolicyPlayer(name=f"PG-{get_now_str()}")
    agent.model = initialize_model(game_type, hp)
    with open(f"saved-models/{agent.name}.yaml", "w") as f:
        yaml.dump(hp, f)

    opponents = [ConservativePlayer(name="Conservative")]
    reward_buffer = deque(maxlen=hp["REWARD_BUFFER_SIZE"])
    reward_buffer_vs = {}
    for opp in opponents:
        reward_buffer_vs[opp.name] = deque(
            maxlen=hp["REWARD_BUFFER_SIZE"] // len(opponents)
        )

    optimizer = Adam(learning_rate=hp["LEARNING_RATE"])
    mse_loss = MeanSquaredError()

    epsisode_transitions = []
    experience_buffer = deque(maxlen=hp["REPLAY_SIZE"])

    writer = SummaryWriter(f"runs/{agent.name}")
    best_reward = hp["SAVE_MODEL_ABS_THRESHOLD"]
    episode_ind = 0  # Number of full episodes completed
    step = 0  # Number of agent actions taken
    while True:

        if episode_ind > hp["MAX_EPISODES"]:
            break

        opponent = opponents[(episode_ind // 2) % len(opponents)]
        player_pos = episode_ind % 2
        agent_transitions = generate_episode_transitions(
            game_type, hp, agent, opponent, player_pos
        )
        episode_ind += 1
        step += len(agent_transitions)
        epsisode_transitions.append(agent_transitions)

        reward_buffer.append(agent_transitions[-1].reward)
        reward_buffer_vs[opponent.name].append(agent_transitions[-1].reward)
        for t in agent_transitions:
            experience_buffer.append(t)
        if episode_ind % hp["RECORD_EPISODES"] == 0:
            record_episode_statistics(
                writer, game, step, experience_buffer, reward_buffer, reward_buffer_vs
            )

        # Save model if we have a historically best result
        smoothed_reward = sum(reward_buffer) / len(reward_buffer)
        if (
            len(reward_buffer) == hp["REWARD_BUFFER_SIZE"]
            and smoothed_reward > best_reward + hp["SAVE_MODEL_REL_THRESHOLD"]
        ):
            agent.model.save(f"saved-models/{agent.name}.keras")
            best_reward = smoothed_reward

        # Don't start training the network until we have enough data
        if len(epsisode_transitions) < hp["BATCH_N_EPISODES"]:
            continue

        training_data = [t for et in epsisode_transitions for t in et]
        record_histograms = (
            step // hp["RECORD_HISTOGRAMS"]
            != (step - len(training_data)) // hp["RECORD_HISTOGRAMS"]
        )
        record_scalars = (
            step // hp["RECORD_SCALARS"]
            != (step - len(training_data)) // hp["RECORD_SCALARS"]
        )

        # Unpack training data
        game = pyspiel.load_game(game_type,{"numdice":5})
        selected_actions = np.array([trsn.action for trsn in training_data])
        #selected_move_mask = one_hot(selected_actions, game.num_distinct_actions())
        selected_move_mask = one_hot(selected_actions, 61)
        x_train = np.array([trsn.state for trsn in training_data])
        rewards = np.array([trsn.reward for trsn in training_data]).astype("float32")
        action_legality = np.array([trsn.legal_actions for trsn in training_data]).astype("float32")

        with tf.GradientTape() as tape:

            # Generate logits
            logits = agent.model(x_train)

            # Mask logits for illegal moves
            # Illegal moves have large negative logits
            # With no dependence on model parameters
            large_neg_logits = -10 * np.ones(logits.shape)
            logits = tf.multiply(action_legality,logits) + tf.multiply((1-action_legality),large_neg_logits)


            if hp["ACTOR_CRITIC"]:

                # Update rewards with value of future state
                if hp["N_TD"] != -1:

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
                    _, resulting_state_values = agent.model(resulting_boards)
                    resulting_state_values = resulting_state_values[:, 0]
                    rewards = rewards + (
                        hp["DISCOUNT_RATE"] ** hp["N_TD"]
                    ) * np.multiply(non_terminal_states, resulting_state_values)

                logits, state_values = logits
                state_values = state_values[:, 0]
                state_loss = mse_loss(rewards, state_values)

            # Compute logits
            move_log_probs = tf.nn.log_softmax(logits)
            masked_log_probs = tf.multiply(move_log_probs, selected_move_mask)
            selected_log_probs = tf.reduce_sum(masked_log_probs, 1)
            obs_advantage = rewards
            if hp["ACTOR_CRITIC"]:
                obs_advantage = rewards - tf.stop_gradient(state_values)
            expectation_loss = -tf.tensordot(
                obs_advantage, selected_log_probs, axes=1
            ) / len(selected_log_probs)

            # Entropy component of loss
            move_probs = tf.nn.softmax(logits)
            entropy_components = tf.multiply(move_probs, move_log_probs)
            entropy_each_state = -tf.reduce_sum(entropy_components, 1)
            entropy = tf.reduce_mean(entropy_each_state)
            entropy_loss = -hp["ENTROPY_BETA"] * entropy

            # Sum the loss contributions
            loss = expectation_loss + entropy_loss
            if hp["ACTOR_CRITIC"]:
                loss += state_loss * hp["STATE_VALUE_BETA"]
                if record_scalars:
                    writer.add_scalar("state-loss", state_loss.numpy(), step)

                if record_histograms:
                    writer.add_histogram("state_value_train", rewards, step)
                    writer.add_histogram("state_value_pred", state_values.numpy(), step)
                    state_value_error = state_values - rewards
                    writer.add_histogram(
                        "state_value_error", state_value_error.numpy(), step
                    )

        if record_scalars:
            writer.add_scalar("log-expect-loss", expectation_loss.numpy(), step)
            writer.add_scalar("entropy-loss", entropy_loss.numpy(), step)
            writer.add_scalar("loss", loss.numpy(), step)

        grads = tape.gradient(loss, agent.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, agent.model.trainable_variables))
        # grads = tape.gradient(loss,tape.watched_variables())
        # optimizer.apply_gradients(zip(grads, tape.watched_variables()))

        # calc KL-div
        if record_scalars:
            new_logits_v = agent.model.predict_on_batch(x_train)
            if hp["ACTOR_CRITIC"]:
                new_logits_v, _ = new_logits_v
            new_prob_v = tf.nn.softmax(new_logits_v)
            KL_EPSILON = 1e-7
            new_prob_v_kl = new_prob_v + KL_EPSILON
            move_probs_kl = move_probs + KL_EPSILON
            kl_div_v = -np.sum(
                np.log(new_prob_v_kl / move_probs_kl) * move_probs_kl, axis=1
            ).mean()
            writer.add_scalar("Kullback-Leibler divergence", kl_div_v.item(), step)

        # Track gradient variance
        if record_histograms:
            weights_and_biases_flat = np.concatenate(
                [v.numpy().flatten() for v in agent.model.variables]
            )
            writer.add_histogram("weights and biases", weights_and_biases_flat, step)
            grads_flat = np.concatenate([v.numpy().flatten() for v in grads])
            writer.add_histogram("logits",logits.numpy().flatten(), step)
            writer.add_histogram("gradients", grads_flat, step)
            grad_rmse = np.sqrt(np.mean(grads_flat**2))
            writer.add_scalar("grad_rmse", grad_rmse, step)
            grad_max = np.abs(grads_flat).max()
            writer.add_scalar("grad_max", grad_max, step)

        # Reset sampling
        epsisode_transitions.clear()

    writer.close()


if __name__ == "__main__":
    main()
