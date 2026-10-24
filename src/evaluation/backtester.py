"""Backtesting Engine for Evaluating Pricing Agents on Simulated Shopee Environment."""

from typing import List, Dict, Any, Type
import pandas as pd
import numpy as np
from src.environment.shopee_pricing_env import ShopeeDynamicPricingEnv
from src.agents.base import BasePricingAgent
from src.evaluation.metrics import compute_financial_metrics
from src.utils.helpers import setup_logger

logger = setup_logger("backtester")


class PricingBacktester:
    """Runs Monte Carlo evaluation of pricing policies across multiple market episodes."""

    def __init__(self, env: ShopeeDynamicPricingEnv):
        self.env = env

    def evaluate_agent(
        self,
        agent: BasePricingAgent,
        num_episodes: int = 50,
        base_seed: int = 1000
    ) -> pd.DataFrame:
        """Evaluates an agent over multiple random episodes.

        Args:
            agent: The pricing agent to test.
            num_episodes: Number of simulation test episodes.
            base_seed: Starting seed for repeatable stochastic demand.

        Returns:
            DataFrame of metrics per episode.
        """
        results = []

        for ep in range(num_episodes):
            seed = base_seed + ep
            obs, info = self.env.reset(seed=seed)
            terminated = False
            truncated = False
            history = []

            while not (terminated or truncated):
                action = agent.select_action(obs, evaluate=True)
                next_obs, reward, terminated, truncated, step_info = self.env.step(action)
                history.append(step_info)
                obs = next_obs

            metrics = compute_financial_metrics(
                history=history,
                initial_stock=self.env.initial_stock,
                unit_cost=self.env.unit_cost
            )
            metrics["Episode"] = ep
            metrics["Agent"] = agent.name
            results.append(metrics)

        return pd.DataFrame(results)

    def compare_agents(
        self,
        agents: List[BasePricingAgent],
        num_episodes: int = 50,
        base_seed: int = 1000
    ) -> pd.DataFrame:
        """Evaluates and compares multiple agents head-to-head under identical random seeds."""
        all_dfs = []
        for agent in agents:
            logger.info(f"Evaluating policy: {agent.name} across {num_episodes} episodes...")
            df = self.evaluate_agent(agent, num_episodes=num_episodes, base_seed=base_seed)
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)
        summary = combined.groupby("Agent").agg(
            Mean_Net_Profit=("Total_Net_Profit_Rp", "mean"),
            Std_Net_Profit=("Total_Net_Profit_Rp", "std"),
            Mean_GMV=("Total_GMV_Rp", "mean"),
            Mean_Sell_Through=("Sell_Through_Rate_%", "mean"),
            Mean_ASP=("Avg_Selling_Price_Rp", "mean"),
            Mean_Holding_Cost=("Total_Holding_Cost_Rp", "mean")
        ).reset_index()

        return summary.sort_values(by="Mean_Net_Profit", ascending=False)
