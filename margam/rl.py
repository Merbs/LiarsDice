import numpy as np
from datetime import date, datetime
import re

from typing import List
from dataclasses import dataclass
import random

import pyspiel


def get_now_str():
    now_str = str(datetime.now())
    now_str = re.sub(" ", "-", now_str)
    return now_str


class Connect4Error(Exception):
    pass


def get_training_and_viewing_state(game, state):
    """
    For tic-tac-toe and Connect4, convert the
    open_spiel tensor to something trainable
    and viewable from current player POV

    state_np_for_conv: state as numpy array that can be
        fed to a convolutional neural network
        Dimensions 0 and 1 are rows and columns
        Dimension 2 are different players
    human_view_state: 2D numpy array with vacancies
        and different player tokens as different integers
    """
    
    state_as_tensor = state.observation_tensor()
    tensor_shape = game.observation_tensor_shape()
    state_np = np.reshape(np.asarray(state_as_tensor), tensor_shape)

    """
    // One-hot encoding for player number.
    // One-hot encoding for each die (max_dice_per_player_ * sides).
    // One slot(bit) for each legal bid.
    // One slot(bit) for calling liar. (Necessary because observations and
    // information states need to be defined at terminals)
    Only the previous bid of each player are reported
    """

    die_counts = np.reshape(state_np[2:2+5*6],(5,6)).sum(axis=0)
    bets_placed = np.reshape(state_np[32:-1],(10,6))

    view_matrix = np.concatenate(
        [die_counts[np.newaxis, :],
        bets_placed,]
    )

    train_tensor = np.zeros(14)
    bets_placed_flat = bets_placed.flatten()
    if bets_placed_flat.sum() >= 1:    # only has previous bet
        ind = bets_placed_flat.argmax()
        quantity = ind // 6 + 1
        value = ind % 6 + 1
        train_tensor[0] = quantity
        train_tensor[value] = 1

        if bets_placed_flat.sum() >= 2:    # current player has previous bet
            bets_placed_flat[ind] = 0
            ind2 = bets_placed_flat.argmax()
            quantity = ind2 // 6 + 1
            value = ind2 % 6 + 1
            train_tensor[7] = quantity
            train_tensor[7+value] = 1
            bets_placed_flat[ind] = 1

    train_tensor = np.concatenate([die_counts,train_tensor])

    return train_tensor, view_matrix

    # For tic-tac-toe and Connect4...

    # Remove 1st element of 1st dimension showing empty spaces
    state_np = state_np[-1:0:-1, :, :]

    # Move players axis last to be the channels for conv net
    state_np_for_cov = np.moveaxis(state_np, 0, -1)

    # view as 1 2D matrix with the last row being first
    human_view_state = state_np[0, ::-1, :] + 2 * state_np[1, ::-1, :]

    return state_np_for_cov, human_view_state


@dataclass
class Transition:
    """
    Data from the game used to train the agent
    """

    state: np.array = None
    legal_actions: List = None
    action: int = 0             # Action taken in game
    reward: float = 0
    next_state: np.array = None


def apply_temporal_difference(transitions, reward_discount, n_td=1):
    """
    Assign the next_state of each transition n_td steps ahead
    Add discounted rewards of next n_td-1 steps to each transition

    Use n_td=-1 to discount to the end of the episode
    """
    if n_td == -1:
        n_td = len(transitions)
    if n_td < 1:
        raise Connect4Error(f"n_td must be >=1. Got {n_td}")
    transitions_td = []
    for i, tr in enumerate(transitions):
        td_tsn = Transition(
            state=tr.state,
            action=tr.action,
            legal_actions=tr.legal_actions,
            reward=tr.reward,
        )

        for j in range(i + 1, min(len(transitions), i + n_td)):
            td_tsn.reward += transitions[j].reward * reward_discount ** (j - i)

        if i + n_td < len(transitions):
            td_tsn.next_state = transitions[i + n_td].state

        transitions_td.append(td_tsn)
    return transitions_td


def generate_episode_transitions(game_type, hp, agent, opponent, player_pos) -> List:
    if game_type == "liars_dice":
        game = pyspiel.load_game(game_type,{"numdice":5})
    else:
        game = pyspiel.load_game(game_type)
    state = game.new_initial_state()

    agent_transitions = []
    while not state.is_terminal():
        if state.is_chance_node():
            # Sample a chance event outcome.
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
            continue

        current_player_ind = state.current_player()
        current_player = agent if current_player_ind == player_pos else opponent
        # If the player action is legal, do it. Otherwise, do random
        desired_action = current_player.get_move(game, state)
        if desired_action in state.legal_actions():
            action = desired_action
        else:
            action = random.choice(state.legal_actions())                

        legal_actions = [int(i in state.legal_actions()) for i in range(game.num_distinct_actions())]
        state_for_cov, _ = get_training_and_viewing_state(game, state)
        new_transition = Transition(
            state=state_for_cov,
            action=action,
            legal_actions=legal_actions,
        )
        if current_player_ind == player_pos:
            agent_transitions.append(new_transition)

        state.apply_action(action)

        if agent_transitions:
            agent_transitions[-1].reward = state.rewards()[player_pos]

    agent_transitions = apply_temporal_difference(
        agent_transitions,
        hp["DISCOUNT_RATE"],
        n_td=hp["N_TD"],
    )

    if hp["USE_SYMMETRY"]:
        agent_transitions = add_symmetries(game_type, agent_transitions)

    return agent_transitions


def record_episode_statistics(
    writer, game, step, experience_buffer, reward_buffer, reward_buffer_vs
):

    # Record move distribution
    move_distribution = [mr.action for mr in experience_buffer]
    move_distribution = np.array(
        [move_distribution.count(i) for i in range(game.num_distinct_actions())]
    )
    for i in range(game.num_distinct_actions()):
        f = move_distribution[i] / sum(move_distribution)
        writer.add_scalar(f"Action frequency: {i}", f, step)

    # Record reward
    smoothed_reward = sum(reward_buffer) / len(reward_buffer)
    writer.add_scalar("Average reward", smoothed_reward, step)

    # Record win rate overall
    wins = sum(r == game.max_utility() for r in reward_buffer)
    ties = sum(r == 0 for r in reward_buffer)
    losses = sum(r == game.min_utility() for r in reward_buffer)
    assert wins + ties + losses == len(reward_buffer)
    writer.add_scalar("Win rate", wins / len(reward_buffer), step)
    writer.add_scalar("Tie rate", ties / len(reward_buffer), step)
    writer.add_scalar("Loss rate", losses / len(reward_buffer), step)

    # Record reward vs. each opponent
    for opp_name, opp_buffer in reward_buffer_vs.items():
        if len(opp_buffer) == 0:
            continue
        reward_vs = sum(opp_buffer) / len(opp_buffer)
        writer.add_scalar(f"reward-vs-{opp_name}", reward_vs, step)


def add_symmetries(game_type, training_data):
    """
    Not implemented yet
    """
    return training_data
