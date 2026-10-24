"""Contextual Bandit (LinUCB) and Multi-Armed Bandit Dynamic Pricing Agents.
Suited for real-time online pricing exploration with immediate reward feedback.
"""

from typing import Optional
import numpy as np
from src.agents.base import BasePricingAgent


class EpsilonGreedyBanditAgent(BasePricingAgent):
    """Classic Multi-Armed Bandit with empirical reward tracking."""

    def __init__(self, num_actions: int = 7, epsilon: float = 0.1, seed: int = 42):
        super().__init__(num_actions=num_actions, name="EpsilonGreedyBandit")
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.action_counts = np.zeros(num_actions)
        self.action_values = np.zeros(num_actions)

    def select_action(self, observation: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        return int(np.argmax(self.action_values))

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.action_counts[action] += 1
        n = self.action_counts[action]
        # Incremental mean update
        self.action_values[action] += (1.0 / n) * (reward - self.action_values[action])


class LinUCBPricingAgent(BasePricingAgent):
    """Contextual Linear Upper Confidence Bound (LinUCB) Pricing Agent.

    Estimates linear relationship between state context and expected pricing reward:
        Reward = theta_a^T * Context + Noise
    """

    def __init__(self, context_dim: int = 6, num_actions: int = 7, alpha: float = 0.5):
        super().__init__(num_actions=num_actions, name="LinUCBPricing")
        self.context_dim = context_dim
        self.alpha = alpha

        # Design matrices A_a and response vectors b_a for each action
        self.A = [np.identity(context_dim) for _ in range(num_actions)]
        self.b = [np.zeros((context_dim, 1)) for _ in range(num_actions)]

    def select_action(self, observation: np.ndarray, evaluate: bool = False) -> int:
        x = observation.reshape(-1, 1)  # Context column vector
        ucb_scores = np.zeros(self.num_actions)

        for a in range(self.num_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.b[a]

            expected_reward = (theta_a.T @ x).item()
            if not evaluate:
                confidence_bound = self.alpha * np.sqrt((x.T @ A_inv @ x).item())
                ucb_scores[a] = expected_reward + confidence_bound
            else:
                ucb_scores[a] = expected_reward

        return int(np.argmax(ucb_scores))

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        x = state.reshape(-1, 1)
        self.A[action] += x @ x.T
        self.b[action] += reward * x
