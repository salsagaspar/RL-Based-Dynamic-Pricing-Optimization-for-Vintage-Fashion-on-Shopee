"""Base Abstract Class for Dynamic Pricing Agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class BasePricingAgent(ABC):
    """Abstract interface for all pricing decision agents."""

    def __init__(self, num_actions: int, name: str = "BaseAgent"):
        self.num_actions = num_actions
        self.name = name

    @abstractmethod
    def select_action(self, observation: np.ndarray, evaluate: bool = False) -> int:
        """Chooses an action index given current environment observation."""
        raise NotImplementedError

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Performs policy or value update. Default is pass for non-learning baseline agents."""
        pass

    def save(self, filepath: str) -> None:
        """Saves agent parameters/table to disk."""
        pass

    def load(self, filepath: str) -> None:
        """Loads agent parameters from disk."""
        pass
