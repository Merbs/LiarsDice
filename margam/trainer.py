from abc import ABC, abstractmethod
from enum import Enum

from margam.rl import GameHandler, GameType, build_game_handler

class RLAlgorithm:
    DQN = "dqn"
    PG = "pg"

class RLTrainer(ABC):

    def __init__(self, game_type: str):
        self.hp = {}
        self.game_handler = build_game_handler(game_type)
        self.agent = None
        self.writer = None
        self.step = 0
        self.experience_buffer = []
        self.reward_buffer = []
        self.reward_buffer_vs = {}

    @staticmethod
    def get_now_str():
        """
        For naming logs
        """
        now_str = str(datetime.now())
        now_str = re.sub(" ", "-", now_str)
        return now_str

    @abstractmethod
    def initiaize_agent(self) -> Player:
        """
        Create an untrained trainable agent
        """
        pass

    
    def train(self):
        """
        Train the agent
        """
        # Create agent
        if self.agent is None:
            self.initiaize_agent()

        # Make collateral folder and save hyperparameters
        with open(f"saved-models/{agent.name}.yaml", "w") as f:
            yaml.dump(hp, f)

        # Open writer
        self.writer = SummaryWriter(f"runs/{agent.name}")


        try:
            self._train()
        finally:
            self.writer.close()
            self.writer = None

    @abstractmethod
    def _train(self):
        pass

    @staticmethod
    def apply_temporal_difference(
        transitions: List[Transition],
        reward_discount: float,
        n_td: int = 1
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

    def record_episode_statistics(self):

        # Record move distribution
        move_distribution = [mr.action for mr in self.experience_buffer]
        move_distribution = np.array(
            [move_distribution.count(i) for i in range(self.game_handler.game.num_distinct_actions())]
        )
        for i in range(self.game_handler.game.num_distinct_actions()):
            f = move_distribution[i] / sum(move_distribution)
            self.writer.add_scalar(f"Action frequency: {i}", f, self.step)

        # Record reward
        smoothed_reward = sum(self.reward_buffer) / len(self.reward_buffer)
        self.writer.add_scalar("Average reward", smoothed_reward, self.step)

        # Record win rate overall
        wins = sum(r == self.game_handler.game.max_utility() for r in self.reward_buffer)
        ties = sum(r == 0 for r in self.reward_buffer)
        losses = sum(r == self.game_handler.game.min_utility() for r in self.reward_buffer)
        assert wins + ties + losses == len(self.reward_buffer)
        self.writer.add_scalar("Win rate", wins / len(self.reward_buffer), self.step)
        self.writer.add_scalar("Tie rate", ties / len(self.reward_buffer), self.step)
        self.writer.add_scalar("Loss rate", losses / len(self.reward_buffer), self.step)

        # Record reward vs. each opponent
        for opp_name, opp_buffer in self.reward_buffer_vs.items():
            if len(opp_buffer) == 0:
                continue
            reward_vs = sum(opp_buffer) / len(opp_buffer)
            self.writer.add_scalar(f"reward-vs-{opp_name}", reward_vs, self.step)
