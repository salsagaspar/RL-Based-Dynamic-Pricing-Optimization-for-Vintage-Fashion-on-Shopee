"""Baseline Heuristic Pricing Strategies for Benchmark Comparisons.
Includes Fixed Catalog Pricing, Time-Based Markdown, and Campaign Exploitation.
"""

from typing import List, Optional
import numpy as np
from src.agents.base import BasePricingAgent


class FixedCatalogPricingAgent(BasePricingAgent):
    """Benchmark: Maintains fixed catalog price throughout product lifecycle."""

    def __init__(self, action_index: int = 3, num_actions: int = 7):
        # Default index 3 corresponds to 0.00% adjustment (base price)
        super().__init__(num_actions=num_actions, name="FixedCatalogPricing")
        self.action_index = action_index

    def select_action(self, observation: np.ndarray, evaluate: bool = False) -> int:
        return self.action_index


class RuleBasedMarkdownAgent(BasePricingAgent):
    """Industry Heuristic: Sells at premium early, aggressively marks down as deadline approaches."""

    def __init__(self, num_actions: int = 7):
        super().__init__(num_actions=num_actions, name="RuleBasedMarkdown")

    def select_action(self, observation: np.ndarray, evaluate: bool = False) -> int:
        stock_ratio = observation[0]
        time_remaining_ratio = observation[1]

        # If plenty of time remaining, keep standard or slight premium
        if time_remaining_ratio > 0.6:
            return 3  # 0%
        elif time_remaining_ratio > 0.3:
            # Halfway through season: if stock is still high (>40%), start 10% markdown
            if stock_ratio > 0.4:
                return 1  # -10% markdown
            return 3      # 0%
        else:
            # Late stage: clearance discount if still stock left
            if stock_ratio > 0.2:
                return 0  # -20% deep discount to avoid deadstock
            return 2      # -5% mild discount


class CampaignAwarePricingAgent(BasePricingAgent):
    """E-Commerce Retailer Heuristic: Exploits high traffic during Shopee campaigns, normal otherwise."""

    def __init__(self, num_actions: int = 7):
        super().__init__(num_actions=num_actions, name="CampaignAwarePricing")

    def select_action(self, observation: np.ndarray, evaluate: bool = False) -> int:
        is_campaign = observation[4] > 0.5
        stock_ratio = observation[0]

        if is_campaign:
            # High intent to buy during campaign: slight markup or optimal catalog price
            return 5 if stock_ratio < 0.5 else 4  # +10% or +5%
        else:
            # Regular days: keep competitive base price
            return 2 if stock_ratio > 0.5 else 3  # -5% or 0%
