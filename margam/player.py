import copy
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np

from margam.rl import GameHandler


class Player(ABC):

    def __init__(self, game_handler, name=None):
        self.game_handler = game_handler
        self.name = name or "nameless"

    @abstractmethod
    def get_move(self, state) -> int:
        pass


class HumanPlayer(Player):

    def get_move(self, state) -> int:

        print(f"\nPlayer: {self.name}")
        eval_vector = self.game_handler.get_eval_vector(state)
        self.game_handler.show_state_on_terminal(eval_vector)
        print("Available moves:")
        print(state.legal_actions())
        
        valid_input = False
        while not valid_input:
            new_input = input(f"Select a move:")

            try:
                move_to_play = int(new_input)
            except ValueError:
                continue

            valid_input = move_to_play in state.legal_actions()

        return move_to_play


class RandomPlayer(Player):
    def get_move(self, state) -> int:
        return random.choice(state.legal_actions())


class ColumnSpammer(Player):
    """
    Rank all game actions at the beginning and
    always play the most preferred legal action
    """
    def __init__(self, game_handler, name=None, move_preferences=None):
        super().__init__(game_handler, name)
        self.move_preferences = move_preferences
        if move_preferences is None:
            self.move_preferences = [i for _ in range(game_handler.game.num_distinct_actions())]
            random.shuffle(self.move_preferences)

    def get_move(self, state) -> int:
        for move in self.move_preferences:
            if move in state.legal_actions():
                return self.favorite_move
        raise MargamError("No preferred moves {self.move_preferences} out of legal moves: {state.legal_actions()}")


class MiniMax(Player):
    """
    Only works for 2 player games

    Depth 0: random player
    Depth 1: Always makes winning move if available
    Depth 2: Blocks opponent from winning on next move
    Depth 3: Sets up forced win on next move
    etc.

    TODO: don't depend upon full game
    """

    def __init__(self, *args, max_depth=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_depth = max_depth

    def eval_state(
        self, state, game, depth, orig_player
    ) -> Tuple[float, Optional[int]]:
        """
        Returns a tuple with
        - The value of the current state for player 0
        - The best move to be taken for current agent
        """

        if state.is_terminal():
            return (state.returns()[orig_player], None)
        if depth <= 0 or len(state.legal_actions()) == 0:
            tie_reward = (game.max_utility() + game.min_utility()) / 2
            return tie_reward, random.choice(state.legal_actions())

        actions_with_value = defaultdict(list)
        for move in state.legal_actions():
            state_result = copy.copy(state)
            state_result.apply_action(move)
            value, _ = self.eval_state(state_result, game, depth - 1, orig_player)
            actions_with_value[value].append(move)

        if state.current_player() == orig_player:
            move_value = max(actions_with_value.keys())
        else:
            move_value = min(actions_with_value.keys())
        action = random.choice(actions_with_value[move_value])

        return move_value, action

    def get_move(self, state) -> int:
        value, move = self.eval_state(
            state,
            game,
            self.max_depth,
            orig_player=state.current_player(),
        )
        return move
