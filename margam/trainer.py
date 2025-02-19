from abc import ABC, abstractmethod

class Trainer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def initiaize_agent(self):
        """
        Create an untrained trainable agent
        """
        pass

    @abstractmethod
    def train_agent(self, agent, game, state) -> int:
        """
        Train the agent
        """
        pass
