import numpy as np
from datetime import date, datetime
import re

from typing import List
from dataclasses import dataclass
import random
from enum import Enum
from abc import ABC, abstractmethod

import pyspiel


class MargamError(Exception):
    pass

class GameType(Enum):
    # get pyspeil game names from `pyspiel.registered_names()`
    CONNECT_FOUR = "connect_four"
    TIC_TAC_TOE = "tic_tac_toe"
    LIARS_DICE = "liars_dice"

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



def build_game_handler(game_type: str):
    gt = GameType(game_type)
    if gt == GameType.CONNECT_FOUR:
        return ConnectFourHandler()
    elif gt == GameType.TIC_TAC_TOE:
        return TicTacToeHandler()
    elif gt == GameType.LIARS_DICE:
        return LiarsDiceHandler()
    else:
        raise MargamError(f"Unsupported game type: {game_type}")

class GameHandler(ABC):

    def __init__(self,game_type: str):
        self.game_type = GameType(game_type)
        self.game = self.get_open_spiel_game()

    def get_eval_vector(self,state):
        """
        Get a vector representing the observation
        of the current player
        """
        state_as_tensor = state.observation_tensor()
        tensor_shape = self.game.observation_tensor_shape()
        return np.reshape(np.asarray(state_as_tensor), tensor_shape)
         

    @abstractmethod
    def show_state_on_terminal(self,eval_vector):
        """
        Show to a human
        """
        pass

    def get_open_spiel_game(self):
        return pyspiel.load_game(self.game_type)

    def generate_episode_transitions(self, players) -> List[List[Transition]]:
        """
        Generate state transition histories for all players.
        Result is indexed in the same order as the supplied players
        """
        state = self.game.new_initial_state()

        agent_transitions = [[]] *len(players)
        while not state.is_terminal():
            if state.is_chance_node():
                # Sample a chance event outcome.
                outcomes_with_probs = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes_with_probs)
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
                continue

            current_player_ind = state.current_player()
            current_player = players[current_player_ind]
            # If the player action is legal, do it. Otherwise, do random
            desired_action = current_player.get_move(state)
            if desired_action in state.legal_actions():
                action = desired_action
            else:
                action = random.choice(state.legal_actions())                

            legal_actions = [int(i in state.legal_actions()) for i in range(self.game.num_distinct_actions())]
            state_for_cov = self.get_eval_vector(state)
            new_transition = Transition(
                state=state_for_cov,
                action=action,
                legal_actions=legal_actions,
            )
            agent_transitions[current_player_ind].append(new_transition)

            state.apply_action(action)

            # Update rewards for last action taken by each player
            # since these may be updated after another player
            # takes their turn
            for player_ind, _ in enumerate(players):
                if len(agent_transitions[player_ind]) == 0:
                    continue
                agent_transitions[player_ind][-1].reward = state.rewards()[player_ind]

        return agent_transitions

    def add_symmetries(self,training_data):
        """
        Not implemented yet
        """
        return training_data

class ConnectFourHandler(GameHandler):

    def __init__(self):
        super().__init__(GameType.CONNECT_FOUR)

    def get_eval_vector(self,state):
        """
        Get a vector representing the observation
        of the current player
        """
        
        state_np = super().get_eval_vector(state)

        # Remove 1st element of 1st dimension showing empty spaces
        state_np = state_np[-1:0:-1, :, :]

        # Move players axis last to be the channels for conv net
        state_np_for_cov = np.moveaxis(state_np, 0, -1)

        return state_np_for_cov

    def show_state_on_terminal(self,eval_vector):
        """
        Show to a human
        """

        # view as 1 2D matrix with the last row being first
        human_view_state = eval_vector[0, ::-1, :] + 2 * eval_vector[1, ::-1, :]
        print(human_view_state)

class TicTacToeHandler(GameHandler):

    def __init__(self):
        super().__init__(GameType.TIC_TAC_TOE)

    def get_eval_vector(self,state):
        """
        Get a vector representing the observation
        of the current player
        """
        
        state_np = super().get_eval_vector(state)

        # Remove 1st element of 1st dimension showing empty spaces
        state_np = state_np[-1:0:-1, :, :]

        # Move players axis last to be the channels for conv net
        state_np_for_cov = np.moveaxis(state_np, 0, -1)

        return state_np_for_cov

    def show_state_on_terminal(self,eval_vector):
        """
        Show to a human
        """

        # view as 1 2D matrix with the last row being first
        human_view_state = eval_vector[0, ::-1, :] + 2 * eval_vector[1, ::-1, :]
        print(human_view_state)

class LiarsDiceHandler(GameHandler):

    def __init__(self, n_dice=5):
        super().__init__(GameType.LIARS_DICE)
        self.n_dice = n_dice

    def get_open_spiel_game(self):
        return pyspiel.load_game(self.game_type.value,{"numdice":self.n_dice})

    def get_eval_vector(self,state):
        """
        Get a vector representing the observation
        of the current player

        // One-hot encoding for player number.
        // One-hot encoding for each die (max_dice_per_player_ * sides).
        // One slot(bit) for each legal bid.
        // One slot(bit) for calling liar. (Necessary because observations and
        // information states need to be defined at terminals)
        Only the previous bid of each player are reported
        """
        
        state_np = super().get_eval_vector(state)

        die_counts = np.reshape(state_np[2:2+5*6],(5,6)).sum(axis=0)
        bets_placed = np.reshape(state_np[32:-1],(10,6))

        train_tensor = np.zeros(14)
        bets_placed_flat = bets_placed.flatten()
        if bets_placed_flat.sum() >= 1:    # only opponent has previous bet
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
                train_tensor = np.concat([train_tensor[7:14],train_tensor[0:7]])

        train_tensor = np.concatenate([die_counts,train_tensor])
        return train_tensor

    def show_state_on_terminal(self,eval_vector):
        """
        Show to a human
        """

        print("Your die counts:")
        for i, count in enumerate(eval_vector[:6]):
            if count == 0:
                continue
            print(f"{int(count)} {i+1}s")

        your_previous_bet_q = int(eval_vector[13])
        your_previous_bet = eval_vector[14:20]
        if your_previous_bet_q > 0:
            value = int(np.where(your_previous_bet==1)[0]+1)
            print(f"Your previous bet: {your_previous_bet_q} {value}'s")

        opp_bet_q = int(eval_vector[6])
        opp_bet = eval_vector[7:13]
        if opp_bet_q > 0:
            value = int(np.where(opp_bet==1)[0]+1)
            print(f"Opponent's bet: {opp_bet_q} {value}'s")
