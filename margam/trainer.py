from abc import ABC, abstractmethod
from datetime import datetime
from collections import deque
from pathlib import Path
import os
import re
import numpy as np
import yaml


from typing import List

from margam.rl import Transition, build_game_handler, MargamError
from margam.player import Player, create_player

from tensorboardX import SummaryWriter


class RLTrainer(ABC):
    """
    Assumes you have an agent with a model and that you
    are training it in a 2-player with a set of
    rotating opponents
    """

    def __init__(
        self,
        hyperparameters: dict,
        save_to_disk: bool = True,
    ):
        self.load_hyperparameters(hyperparameters)
        self.game_handler = build_game_handler(self.GAME)
        self.name = self.get_unique_name()
        self.save_folder = Path(self.name) if save_to_disk else None
        self.agent = self.initialize_agent()
        self.rotating_opponents = [
            create_player(pt, self.game_handler) for pt in self.OPPONENTS
        ]
        self.writer = None
        self.step = 0
        self.episode_ind = 0
        self.experience_buffer = []
        self.reward_buffer = []
        self.reward_buffer_vs = {}
        self.best_reward = 0

        if self.save_folder:
            os.makedirs(self.save_folder, exist_ok=True)

            yaml_save = os.path.join(self.save_folder, f"{self.name}.yaml")
            with open(yaml_save, "w") as f:
                yaml.dump(hyperparameters, f)

            self.writer = SummaryWriter(f"runs/{self.name}")

    @staticmethod
    def get_now_str():
        """
        For naming logs
        """
        now_str = str(datetime.now())
        now_str = re.sub(" ", "-", now_str)
        return now_str

    @abstractmethod
    def get_unique_name(self) -> str:
        pass

    @abstractmethod
    def initialize_agent(self) -> Player:
        """
        Create an untrained trainable agent
        """
        pass

    def initialize_training_stats(self):
        self.reward_buffer = deque(maxlen=self.REWARD_BUFFER_SIZE)
        self.reward_buffer_vs = {}
        for opp in self.rotating_opponents:
            self.reward_buffer_vs[opp.name] = deque(
                maxlen=self.REWARD_BUFFER_SIZE // len(self.rotating_opponents)
            )

        self.best_reward = self.SAVE_MODEL_ABS_THRESHOLD
        self.episode_ind = 0  # Number of full episodes completed
        self.step = 0  # Number of agent actions taken

    def generate_episode_transitions(self) -> List[List[Transition]]:
        """
        make player list with agent and rotating opponent
        """
        opponent = self.rotating_opponents[
            (self.episode_ind // 2) % len(self.rotating_opponents)
        ]
        players = [self.agent, opponent]
        if self.episode_ind % 2:
            players = list(reversed(players))
        agent_transitions = self.game_handler.generate_episode_transitions(players)[
            self.episode_ind % 2
        ]
        self.episode_ind += 1
        return agent_transitions, opponent

    def train(self):
        """
        Train the agent
        """

        if not self.agent:
            raise MargamError("Agent is required for training")
        if not self.rotating_opponents:
            raise MargamError("Opponent list must not be empty")
        
        self.initialize_training_stats()

        try:
            self._train()
        finally:
            if self.writer:
                self.writer.close()
            self.writer = None

    @abstractmethod
    def _train(self):
        pass

    def load_hyperparameters(self, hp):
        for key, value in hp.items():
            setattr(self, key, value)

    @staticmethod
    def apply_temporal_difference(
        transitions: List[Transition], reward_discount: float, n_td: int = 1
    ):
        """
        Assign the next_state of each transition n_td steps ahead
        Add discounted rewards of next n_td-1 steps to each transition

        Use n_td=-1 to discount to the end of the episode
        """
        if n_td == -1:
            n_td = len(transitions)
        if n_td < 1:
            raise MargamError(f"n_td must be >=1. Got {n_td}")
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

    def save_checkpoint_model(self):
        """
        Save model if we have a historically best result
        """
        if not self.save_folder:
            return
        smoothed_reward = sum(self.reward_buffer) / len(self.reward_buffer)
        if (
            len(self.reward_buffer) == self.REWARD_BUFFER_SIZE
            and smoothed_reward > self.best_reward + self.SAVE_MODEL_REL_THRESHOLD
        ):
            self.agent.model.save(f"{self.save_folder}/{self.agent.name}.keras")
            self.best_reward = smoothed_reward

    def record_episode_statistics(self):
        if not self.writer:
            return

        # Record move distribution
        move_distribution = [mr.action for mr in self.experience_buffer]
        move_distribution = np.array(
            [
                move_distribution.count(i)
                for i in range(self.game_handler.game.num_distinct_actions())
            ]
        )
        for i in range(self.game_handler.game.num_distinct_actions()):
            f = move_distribution[i] / sum(move_distribution)
            self.writer.add_scalar(f"Action frequency: {i}", f, self.step)

        # Record reward
        smoothed_reward = sum(self.reward_buffer) / len(self.reward_buffer)
        self.writer.add_scalar("Average reward", smoothed_reward, self.step)

        # Record win rate overall
        wins = sum(
            r == self.game_handler.game.max_utility() for r in self.reward_buffer
        )
        ties = sum(r == 0 for r in self.reward_buffer)
        losses = sum(
            r == self.game_handler.game.min_utility() for r in self.reward_buffer
        )
        if (wins + ties + losses) != len(self.reward_buffer):
            raise MargamError(
                f"Wins: {wins}, losses: {ties}, ties: {losses}, episodes in buffer: {len(self.reward_buffer)}"
            )
        self.writer.add_scalar("Win rate", wins / len(self.reward_buffer), self.step)
        self.writer.add_scalar("Tie rate", ties / len(self.reward_buffer), self.step)
        self.writer.add_scalar("Loss rate", losses / len(self.reward_buffer), self.step)

        # Record reward vs. each opponent
        for opp_name, opp_buffer in self.reward_buffer_vs.items():
            if len(opp_buffer) == 0:
                continue
            reward_vs = sum(opp_buffer) / len(opp_buffer)
            self.writer.add_scalar(f"reward-vs-{opp_name}", reward_vs, self.step)
