import copy
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np

from c4lib.utils import get_training_and_viewing_state


class Player(ABC):
    requires_user_input = False

    def __init__(self, name=None):
        self.name = name or "nameless"

    @abstractmethod
    def get_move(self, game, state) -> int:
        pass


class HumanPlayer(Player):
    requires_user_input = True

    def get_move(self, game, state) -> int:

        valid_input = False
        while not valid_input:
            print(f"\nPlayer: {self.name}")
            print("State")
            state_np_for_cov, human_view_state = get_training_and_viewing_state(
                game, state
            )
            print(state_np_for_cov)
            print(human_view_state)
            print("Available moves:")
            print(state.legal_actions())
            new_input = input(f"Select a move:")

            try:
                move_to_play = int(new_input)
            except ValueError:
                continue

            valid_input = move_to_play in state.legal_actions()

        return move_to_play


class RandomPlayer(Player):
    def get_move(self, game, state) -> int:
        return random.choice(state.legal_actions())


class ColumnSpammer(Player):
    def __init__(self, name=None, move_preference=0):
        super().__init__(name)
        self.favorite_move = move_preference

    def get_move(self, game, state) -> int:
        if self.favorite_move in state.legal_actions():
            return self.favorite_move
        return random.choice(state.legal_actions())


class MiniMax(Player):
    """
    Only works for 2 player games

    Depth 0: random player
    Depth 1: Always makes winning move if available
    Depth 2: Blocks opponent from winning on next move
    Depth 3: Sets up forced win on next move
    etc.
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

    def get_move(self, game, state) -> int:
        value, move = self.eval_state(
            state,
            game,
            self.max_depth,
            orig_player=state.current_player(),
        )
        return move
