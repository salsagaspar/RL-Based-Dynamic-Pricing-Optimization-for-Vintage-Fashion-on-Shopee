"""Gymnasium-Compatible Dynamic Pricing Environment for Vintage Fashion on Shopee.
Formulates the revenue & inventory management challenge as a Markov Decision Process (MDP).
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from src.environment.demand_model import ShopeeDemandSimulator

try:
    import gymnasium as gym
    from gymnasium import spaces
    BaseEnv = gym.Env
except ImportError:
    # Graceful fallback if gymnasium is not yet installed in local environment
    class SpacesMock:
        class Box:
            def __init__(self, low, high, shape, dtype=np.float32):
                self.low = np.array(low, dtype=dtype)
                self.high = np.array(high, dtype=dtype)
                self.shape = shape
                self.dtype = dtype
        class Discrete:
            def __init__(self, n):
                self.n = n

    spaces = SpacesMock()
    class BaseEnv:
        pass


class ShopeeDynamicPricingEnv(BaseEnv):
    """E-Commerce Dynamic Pricing Environment.

    State Vector (6-dim continuous):
        0: stock_ratio (current_stock / initial_stock)
        1: time_remaining_ratio ((horizon - current_day) / horizon)
        2: cost_to_base_price_ratio (cost / base_price)
        3: current_price_ratio (current_price / base_price)
        4: campaign_flag (1.0 if promo day, 0.0 otherwise)
        5: recent_sales_velocity (units/day in last 3 days normalized)

    Action Space:
        Discrete (7 actions):
        [-20%, -10%, -5%, 0%, +5%, +10%, +20%] price change from base catalog price.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        sku: str = "V90-TEE-001",
        base_price: float = 120000.0,
        unit_cost: float = 56000.0,
        initial_stock: int = 50,
        horizon_days: int = 60,
        base_daily_demand: float = 1.5,
        price_elasticity: float = -1.5,
        platform_fee_pct: float = 0.09,
        daily_holding_cost: float = 350.0,
        stockout_penalty: float = 15000.0,
        salvage_value_ratio: float = 0.40,
        campaign_days: Optional[List[int]] = None,
        discrete_actions: Optional[List[float]] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.sku = sku
        self.base_price = float(base_price)
        self.unit_cost = float(unit_cost)
        self.initial_stock = int(initial_stock)
        self.horizon_days = int(horizon_days)
        self.platform_fee_pct = float(platform_fee_pct)
        self.daily_holding_cost = float(daily_holding_cost)
        self.stockout_penalty = float(stockout_penalty)
        self.salvage_value_ratio = float(salvage_value_ratio)
        self.campaign_days = campaign_days or [9, 10, 25, 26, 27, 40, 55]

        # Action space: percentage adjustment multipliers
        self.discrete_actions = discrete_actions or [-0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20]
        self.num_actions = len(self.discrete_actions)

        # Observation & Action Spaces
        self.action_space = spaces.Discrete(self.num_actions)
        # Continuous bounded state vector [0.0, 1.0] or [-1.0, 2.0]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.5, 1.0, 1.0, 2.0, 1.0, 5.0], dtype=np.float32),
            shape=(6,),
            dtype=np.float32,
        )

        self.demand_simulator = ShopeeDemandSimulator(
            base_daily_demand=base_daily_demand,
            base_price=self.base_price,
            price_elasticity=price_elasticity,
            campaign_boost=1.6,
            seed=seed,
        )

        self.current_day = 0
        self.current_stock = self.initial_stock
        self.current_price = self.base_price
        self.recent_sales = []
        self.cumulative_profit = 0.0
        self.total_units_sold = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Resets the environment to initial state."""
        if seed is not None:
            self.demand_simulator = ShopeeDemandSimulator(
                base_daily_demand=self.demand_simulator.base_daily_demand,
                base_price=self.base_price,
                price_elasticity=self.demand_simulator.elasticity,
                campaign_boost=1.6,
                seed=seed,
            )

        self.current_day = 0
        self.current_stock = self.initial_stock
        self.current_price = self.base_price
        self.recent_sales = [0, 0, 0]
        self.cumulative_profit = 0.0
        self.total_units_sold = 0

        obs = self._get_obs()
        info = {
            "sku": self.sku,
            "day": self.current_day,
            "stock": self.current_stock,
            "price": self.current_price,
        }
        return obs, info

    def _get_obs(self) -> np.ndarray:
        """Constructs normalized state representation."""
        stock_ratio = self.current_stock / max(1, self.initial_stock)
        time_rem_ratio = (self.horizon_days - self.current_day) / self.horizon_days
        cost_ratio = self.unit_cost / self.base_price
        price_ratio = self.current_price / self.base_price
        is_campaign = 1.0 if self.current_day in self.campaign_days else 0.0
        velocity = np.mean(self.recent_sales[-3:]) if len(self.recent_sales) >= 3 else 0.0

        obs = np.array(
            [stock_ratio, time_rem_ratio, cost_ratio, price_ratio, is_campaign, velocity],
            dtype=np.float32,
        )
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Executes one day of dynamic pricing on Shopee.

        Args:
            action: Index of discrete action multiplier.

        Returns:
            obs, reward, terminated, truncated, info
        """
        assert 0 <= action < self.num_actions, f"Invalid action index {action}"
        price_pct = self.discrete_actions[action]
        offered_price = self.base_price * (1.0 + price_pct)
        # Ensure price never falls below bare unit cost
        offered_price = max(offered_price, self.unit_cost * 1.05)
        self.current_price = offered_price

        # Check campaign day
        is_campaign = self.current_day in self.campaign_days

        # Sample customer demand
        customer_demand = self.demand_simulator.sample_demand(
            offered_price=offered_price, is_campaign_day=is_campaign
        )

        # Compute fulfilled units & stockouts
        units_sold = min(self.current_stock, customer_demand)
        stockout_occurred = customer_demand > self.current_stock
        unmet_demand = customer_demand - units_sold if stockout_occurred else 0

        # Update stock & historical metrics
        self.current_stock -= units_sold
        self.total_units_sold += units_sold
        self.recent_sales.append(units_sold)

        # Financial Calculations (in IDR)
        # Net unit revenue after Shopee fees: Price * (1 - PlatformFee)
        net_selling_price = offered_price * (1.0 - self.platform_fee_pct)
        gross_profit_today = units_sold * (net_selling_price - self.unit_cost)

        # Deduct holding costs for unsold items in storage
        daily_holding = self.current_stock * self.daily_holding_cost

        # Deduct penalty for stockout (lost sales & ranking degradation)
        stockout_penalty_cost = unmet_demand * self.stockout_penalty

        # Step reward
        reward = gross_profit_today - daily_holding - stockout_penalty_cost

        # Advance day
        self.current_day += 1

        # Check termination condition
        terminated = (self.current_stock <= 0) or (self.current_day >= self.horizon_days)
        truncated = False

        # Terminal liquidation salvage value if period ends with unsold deadstock
        if terminated and self.current_stock > 0 and self.current_day >= self.horizon_days:
            salvage_revenue = self.current_stock * (self.unit_cost * self.salvage_value_ratio)
            salvage_loss = self.current_stock * self.unit_cost - salvage_revenue
            reward -= salvage_loss

        self.cumulative_profit += reward

        obs = self._get_obs()
        info = {
            "day": self.current_day,
            "offered_price": offered_price,
            "units_sold": units_sold,
            "stock_remaining": self.current_stock,
            "daily_net_profit": gross_profit_today,
            "daily_holding_cost": daily_holding,
            "stockout_penalty": stockout_penalty_cost,
            "cumulative_profit": self.cumulative_profit,
            "total_sold": self.total_units_sold,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Displays simulation day status."""
        print(
            f"Day {self.current_day:02d}/{self.horizon_days} | "
            f"Stock: {self.current_stock:02d} | "
            f"Price: Rp {self.current_price:,.0f} | "
            f"Profit: Rp {self.cumulative_profit:,.0f}"
        )
