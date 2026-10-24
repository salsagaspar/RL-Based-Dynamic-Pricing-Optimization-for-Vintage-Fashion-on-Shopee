"""Unit tests for Gymnasium-compatible Dynamic Pricing Environment."""

import pytest
import numpy as np
from src.environment.shopee_pricing_env import ShopeeDynamicPricingEnv


def test_env_initialization_and_reset():
    env = ShopeeDynamicPricingEnv(
        sku="V90-TEE-001",
        base_price=120000.0,
        unit_cost=56000.0,
        initial_stock=50,
        horizon_days=60
    )
    obs, info = env.reset(seed=42)

    assert obs.shape == (6,)
    assert info["sku"] == "V90-TEE-001"
    assert info["stock"] == 50
    assert info["price"] == 120000.0
    assert 0.0 <= obs[0] <= 1.5  # stock ratio


def test_env_step_execution():
    env = ShopeeDynamicPricingEnv(
        base_price=100000.0,
        unit_cost=50000.0,
        initial_stock=20,
        horizon_days=30
    )
    obs, _ = env.reset(seed=123)

    action = 3  # 0% adjustment (base price)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == (6,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["day"] == 1
    assert "units_sold" in info
    assert "stock_remaining" in info
