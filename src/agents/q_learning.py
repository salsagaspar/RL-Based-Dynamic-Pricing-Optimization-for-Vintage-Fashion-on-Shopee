"""Tabular Q-Learning Dynamic Pricing Agent with Discretized State Space.
Provides guaranteed convergence, transparency, and high sample efficiency.
"""

import pickle
from typing import Tuple, Dict, Any, Optional
import numpy as np
from src.agents.base import BasePricingAgent


class QLearningPricingAgent(BasePricingAgent):
    """Q-Learning Agent using discretized state binning for inventory & market states."""

    def __init__(
        self,
        num_actions: int = 7,
        learning_rate: float = 0.05,
        gamma: float = 0.98,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ):
        super().__init__(num_actions=num_actions, name="QLearningPricing")
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # State discretization bin edges
        self.stock_bins = np.array([0.2, 0.4, 0.6, 0.8])
        self.time_bins = np.array([0.25, 0.5, 0.75])
        self.velocity_bins = np.array([0.5, 1.5])

        # Q-table: key is state tuple (stock_bin, time_bin, campaign_bin, velocity_bin)
        self.q_table: Dict[Tuple[int, int, int, int], np.ndarray] = {}

    def _discretize_state(self, observation: np.ndarray) -> Tuple[int, int, int, int]:
        """Maps continuous state vector to discrete tuple."""
        stock_ratio = observation[0]
        time_rem = observation[1]
        is_campaign = int(observation[4] > 0.5)
        velocity = observation[5]

        stock_idx = int(np.digitize(stock_ratio, self.stock_bins))
        time_idx = int(np.digitize(time_rem, self.time_bins))
        velocity_idx = int(np.digitize(velocity, self.velocity_bins))

        return (stock_idx, time_idx, is_campaign, velocity_idx)

    def _get_q_values(self, state_key: Tuple[int, int, int, int]) -> np.ndarray:
        """Retrieves or initializes Q-values for a state."""
        if state_key not in self.q_table:
            # Initialize with small positive values or zeros
            self.q_table[state_key] = np.zeros(self.num_actions, dtype=np.float32)
        return self.q_table[state_key]

    def select_action(self, observation: np.ndarray, evaluate: bool = False) -> int:
        """Epsilon-greedy action selection."""
        state_key = self._discretize_state(observation)
        q_vals = self._get_q_values(state_key)

        if not evaluate and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        return int(np.argmax(q_vals))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Performs Bellman optimality Q-value update."""
        s_key = self._discretize_state(state)
        next_s_key = self._discretize_state(next_state)

        current_q = self._get_q_values(s_key)[action]
        next_max_q = np.max(self._get_q_values(next_s_key)) if not done else 0.0

        target = reward + self.gamma * next_max_q
        self.q_table[s_key][action] += self.lr * (target - current_q)

    def decay_epsilon(self) -> None:
        """Decays exploration parameter after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def save(self, filepath: str) -> None:
        """Saves Q-table and agent state."""
        with open(filepath, "wb") as f:
            pickle.dump({
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "lr": self.lr,
                "gamma": self.gamma,
            }, f)

    def load(self, filepath: str) -> None:
        """Loads Q-table from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.q_table = data["q_table"]
            self.epsilon = data.get("epsilon", self.epsilon_min)
            self.lr = data.get("lr", self.lr)
            self.gamma = data.get("gamma", self.gamma)
