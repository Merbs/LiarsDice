import itertools
import random
from collections import deque
from enum import Enum
from functools import partial
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
from tqdm import tqdm

import pyspiel

from c4lib.player import MiniMax, Player
from c4lib.utils import (
    Connect4Error,
    get_training_and_viewing_state,
    record_episode_statistics,
    generate_episode_transitions,
    get_now_str,
)


class DQNPlayer(Player):
    """
    Agent that uses a neural network to compute
    Q(s,a) values for each action at a given state,
    then chooses the highest value as the action.

    `random_weight` gives a change to perform
    a random action instead for better exploration
    while training
    """

    def __init__(self, *args, random_weight=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.random_weight = random_weight

    def get_move(self, game, state) -> int:
        if np.random.random() < self.random_weight:
            return random.choice(state.legal_actions())

        state_for_cov, human_view_state = get_training_and_viewing_state(game, state)
        q_values = self.model.predict_on_batch(state_for_cov[np.newaxis, :])[0]
        q_values = q_values.astype(float)
        for i, _ in enumerate(q_values):
            if i not in state.legal_actions():
                q_values[i] = -np.inf
        max_q_ind = np.argmax(q_values)

        return max_q_ind


def sample_experience_buffer(buffer, batch_size):
    indices = np.random.choice(len(buffer), batch_size, replace=False)
    return [buffer[idx] for idx in indices]


def initialize_model(game_type, hp, show_model=True):
    """
    Construct neural network used by agent
    """
    game = pyspiel.load_game(game_type)
    state = game.new_initial_state()
    state_np_for_cov, human_view_state = get_training_and_viewing_state(game, state)
    nn_input = keras.Input(shape=state_np_for_cov.shape)

    if game_type == "tic_tac_toe":
        input_flat = layers.Flatten()(nn_input)
        x = layers.Dense(32, activation="relu")(input_flat)
        q_values = layers.Dense(game.num_distinct_actions(), activation="linear")(x)

        # Deuling DQN adds a second column to the neural net that
        # computes state value V(s) and interprets the Q
        # values as advantage of that action in that state
        # Q(s,a) = A(s,a) + V(s)
        # Final output is the same so it is interoperable with vanilla DQN
        if hp["DEULING_DQN"]:
            x_sv = layers.Dense(32, activation="relu")(input_flat)
            sv = layers.Dense(1, activation="linear")(x_sv)
            q_values = (
                q_values - tf.math.reduce_mean(q_values, axis=1, keepdims=True) + sv
            )
    elif game_type == "connect_four":
        x = layers.Conv2D(64,4)(nn_input)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64,activation="relu")(x)
        q_values = layers.Dense(game.num_distinct_actions(), activation="linear")(x)
        if hp["DEULING_DQN"]:
            x_sv = layers.Dense(32, activation="relu")(x)
            sv = layers.Dense(1, activation="linear")(x_sv)
            q_values = (
                q_values - tf.math.reduce_mean(q_values, axis=1, keepdims=True) + sv
            )
    else:
        raise Connect4Error(f"Unrecognized game type: {game_type}")

    model = keras.Model(inputs=nn_input, outputs=q_values, name="DQN-model")
    if show_model:
        model.summary()
    return model


def update_neural_network(
    step, training_data, agent, target_network, hp, optimizer, mse_loss, writer
):
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
    resulting_board_q_target = target_network.predict_on_batch(resulting_boards)
    if hp["DOUBLE_DQN"]:
        resulting_board_q_on_policy = agent.model.predict_on_batch(resulting_boards)
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
        hp["DISCOUNT_RATE"] ** hp["N_TD"]
    ) * np.multiply(non_terminal_states, max_qs)

    # Needed for our mask
    selected_actions = [trsn.action for trsn in training_data]
    selected_action_mask = one_hot(selected_actions, resulting_board_q_target.shape[1])

    # Compute MSE loss based on chosen move values only
    with tf.GradientTape() as tape:
        predicted_q_values = agent.model(x_train)
        predicted_q_values = tf.multiply(predicted_q_values, selected_action_mask)
        predicted_q_values = tf.reduce_sum(predicted_q_values, 1)
        q_prediction_errors = predicted_q_values - q_to_train_single_values
        loss_value = mse_loss(q_to_train_single_values, predicted_q_values)

    weights_and_biases_flat_before_update = np.concatenate(
        [v.numpy().flatten() for v in agent.model.variables]
    )

    grads = tape.gradient(loss_value, agent.model.trainable_variables)
    optimizer.apply_gradients(zip(grads, agent.model.trainable_variables))

    # Record prediction error
    writer.add_scalar("loss", loss_value.numpy(), step)
    if step % hp["RECORD_HISTOGRAMS"] == 0:
        writer.add_histogram("q-predicted", predicted_q_values.numpy(), step)
        writer.add_histogram("q-train", q_to_train_single_values, step)
        writer.add_histogram("q-error", q_prediction_errors.numpy(), step)

        weights_and_biases_flat = np.concatenate(
            [v.numpy().flatten() for v in agent.model.variables]
        )
        writer.add_histogram("weights and biases", weights_and_biases_flat, step)

        # Track gradient variance
        grads_flat = np.concatenate([v.numpy().flatten() for v in grads])
        writer.add_histogram("gradients", grads_flat, step)
        grad_rmse = np.sqrt(np.mean(grads_flat**2))
        writer.add_scalar("grad_rmse", grad_rmse, step)
        grad_max = np.abs(grads_flat).max()
        writer.add_scalar("grad_max", grad_max, step)

        weights_and_biases_delta = (
            weights_and_biases_flat - weights_and_biases_flat_before_update
        )
        writer.add_histogram("weight-bias-updates", weights_and_biases_delta, step)

    # Update policy
    if hp["RECORD_HISTOGRAMS"] % hp["SYNC_TARGET_NETWORK"] != 0:
        raise Connect4Error("SYNC_TARGET_NETWORK must divide RECORD_HISTOGRAMS")
    if step % hp["SYNC_TARGET_NETWORK"] == 0:
        if step % hp["RECORD_HISTOGRAMS"] == 0:
            weights_and_biases_flat = np.concatenate(
                [v.numpy().flatten() for v in agent.model.variables]
            )
            weights_and_biases_target = np.concatenate(
                [v.numpy().flatten() for v in target_network.variables]
            )
            target_parameter_updates = (
                weights_and_biases_flat - weights_and_biases_target
            )
            writer.add_histogram(
                "target parameter updates", target_parameter_updates, step
            )

        target_network.set_weights(agent.model.get_weights())


def train_dqn(game_type, hp):
    """
    Perform the training loop that runs episodes
    and uses the data to train a DQN player
    """

    # Intialize agent and model
    agent = DQNPlayer(name=f"DQN-{get_now_str()}")
    agent.model = initialize_model(game_type, hp)
    target_network = keras.models.clone_model(agent.model)
    target_network.set_weights(agent.model.get_weights())
    with open(f"saved-models/{agent.name}.yaml", "w") as f:
        yaml.dump(hp, f)

    game = pyspiel.load_game(game_type)
    opponents = [MiniMax(name="Minnie", max_depth=1)]

    experience_buffer = deque(maxlen=hp["REPLAY_SIZE"])
    reward_buffer = deque(maxlen=hp["REWARD_BUFFER_SIZE"])
    reward_buffer_vs = {}
    for opp in opponents:
        reward_buffer_vs[opp.name] = deque(
            maxlen=hp["REWARD_BUFFER_SIZE"] // len(opponents)
        )

    mse_loss = MeanSquaredError()
    optimizer = Adam(learning_rate=hp["LEARNING_RATE"])

    writer = SummaryWriter(f"runs/{agent.name}")
    best_reward = hp["SAVE_MODEL_ABS_THRESHOLD"]
    episode_ind = 0  # Number of full episodes completed
    step = 0  # Number of agent actions taken
    while True:

        if episode_ind > hp["MAX_EPISODES"]:
            break

        agent.random_weight = max(
            hp["EPSILON_FINAL"],
            hp["EPSILON_START"] - step / hp["EPSILON_DECAY_LAST_FRAME"],
        )

        opponent = opponents[(episode_ind // 2) % len(opponents)]
        player_pos = episode_ind % 2
        agent_transitions = generate_episode_transitions(
            game_type, hp, agent, opponent, player_pos
        )
        episode_ind += 1

        for t in agent_transitions:
            experience_buffer.append(t)
        reward_buffer.append(agent_transitions[-1].reward)
        reward_buffer_vs[opponent.name].append(agent_transitions[-1].reward)
        if episode_ind % hp["RECORD_EPISODES"] == 0:
            record_episode_statistics(
                writer, game, step, experience_buffer, reward_buffer, reward_buffer_vs
            )

        if agent.random_weight > hp["EPSILON_FINAL"]:
            writer.add_scalar("epsilon", agent.random_weight, step)

        # Save model if we have a historically best result
        smoothed_reward = sum(reward_buffer) / len(reward_buffer)
        if (
            len(reward_buffer) == hp["REWARD_BUFFER_SIZE"]
            and smoothed_reward > best_reward + hp["SAVE_MODEL_REL_THRESHOLD"]
        ):
            agent.model.save(f"saved-models/{agent.name}.keras")
            best_reward = smoothed_reward

        for transition in agent_transitions:
            step += 1
            experience_buffer.append(transition)

            # Don't start training the network until we have enough data
            if len(experience_buffer) < hp["REPLAY_START_SIZE"]:
                continue

            # Get training data
            training_data = sample_experience_buffer(
                experience_buffer, hp["BATCH_SIZE"]
            )

            update_neural_network(
                step,
                training_data,
                agent,
                target_network,
                hp,
                optimizer,
                mse_loss,
                writer,
            )

    writer.close()
