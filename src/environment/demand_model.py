"""Demand Model Simulation for Vintage Fashion on Shopee.
Models price elasticity of demand, campaign shocks (Double Date, Payday), and stochastic purchase decisions.
"""

from typing import Dict, Any
import numpy as np


class ShopeeDemandSimulator:
    """Stochastic demand simulator based on iso-elastic / log-linear demand curve

    Demand equation:
        Expected_Units(p) = Base_Demand * (p / p_base) ^ elasticity * Campaign_Factor
        Realized_Units ~ Poisson(Expected_Units(p))
    """

    def __init__(
        self,
        base_daily_demand: float,
        base_price: float,
        price_elasticity: float = -1.5,
        campaign_boost: float = 1.6,
        seed: int = 42
    ):
        self.base_daily_demand = max(0.1, base_daily_demand)
        self.base_price = max(1.0, base_price)
        self.elasticity = price_elasticity
        self.campaign_boost = campaign_boost
        self.rng = np.random.default_rng(seed)

    def sample_demand(self, offered_price: float, is_campaign_day: bool = False) -> int:
        """Samples realization of units demanded at offered price.

        Args:
            offered_price: Selling price set by RL agent.
            is_campaign_day: True if Shopee Double Date or Payday Sale.

        Returns:
            Non-negative integer units demanded by Shopee shoppers.
        """
        price_ratio = max(0.01, offered_price / self.base_price)
        # Iso-elastic demand response
        elastic_demand = self.base_daily_demand * (price_ratio ** self.elasticity)

        # Apply campaign boost if active
        campaign_multiplier = self.campaign_boost if is_campaign_day else 1.0
        expected_demand = elastic_demand * campaign_multiplier

        # Add market randomness via Poisson process
        realized_demand = int(self.rng.poisson(lam=max(0.0, expected_demand)))
        return realized_demand
