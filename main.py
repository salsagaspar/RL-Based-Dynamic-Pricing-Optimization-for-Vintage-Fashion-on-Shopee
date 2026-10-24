"""Main Execution Pipeline for Training and Benchmarking Dynamic Pricing RL Agents.

Usage:
    python main.py --episodes 500 --sku V90-TEE-001
"""

import argparse
import os
import pandas as pd
import numpy as np

from src.utils.helpers import setup_logger, load_config, set_seed, format_currency_idr
from src.data.loader import ShopeeDatasetLoader
from src.features.feature_pipeline import PricingFeatureEngineer
from src.environment.shopee_pricing_env import ShopeeDynamicPricingEnv
from src.agents.baselines import (
    FixedCatalogPricingAgent,
    RuleBasedMarkdownAgent,
    CampaignAwarePricingAgent,
)
from src.agents.q_learning import QLearningPricingAgent
from src.agents.bandit_agents import LinUCBPricingAgent
from src.evaluation.backtester import PricingBacktester

logger = setup_logger("main_pipeline")


def run_training_and_benchmark(
    sku: str = "V90-TEE-001",
    train_episodes: int = 500,
    eval_episodes: int = 50,
    config_path: str = "configs/config.yaml",
):
    config = load_config(config_path)
    set_seed(config["simulation"]["random_seed"])

    # 1. Load Data & Compute Features
    logger.info("Loading supply chain & order dataset...")
    loader = ShopeeDatasetLoader(config["data"]["raw_file"])
    sheets = loader.load_all_sheets()

    engineer = PricingFeatureEngineer(
        platform_fee_pct=config["platform"]["fee_rates"]["total_effective_fee_pct"],
        daily_holding_cost_rp=config["inventory_economics"]["daily_holding_cost_rp"],
    )
    sku_features = engineer.build_sku_features(sheets)

    # 2. Extract Target SKU profile
    sku_row = sku_features[sku_features["SKU"] == sku]
    if sku_row.empty:
        logger.warning(f"SKU {sku} not found, defaulting to first available SKU: {sku_features['SKU'].iloc[0]}")
        sku_row = sku_features.iloc[0:1]
        sku = sku_row["SKU"].values[0]

    sku_data = sku_row.iloc[0]
    logger.info(
        f"Selected SKU: {sku} ({sku_data.get('Nama_Produk', 'Item')}) | "
        f"Cost: {format_currency_idr(sku_data['Modal_Unit_Rp'])} | "
        f"Base Price: {format_currency_idr(sku_data['Harga_Jual_Rp'])} | "
        f"Stock: {sku_data.get('Stok_Akhir', 50)} units"
    )

    # 3. Instantiate Environment
    env = ShopeeDynamicPricingEnv(
        sku=sku,
        base_price=float(sku_data["Harga_Jual_Rp"]),
        unit_cost=float(sku_data["Modal_Unit_Rp"]),
        initial_stock=int(max(10, sku_data.get("Stok_Akhir", 50))),
        horizon_days=config["simulation"]["time_horizon_days"],
        base_daily_demand=float(sku_data.get("Daily_Velocity", 1.5)),
        price_elasticity=float(sku_data.get("Estimated_Price_Elasticity", -1.5)),
        platform_fee_pct=config["platform"]["fee_rates"]["total_effective_fee_pct"],
        daily_holding_cost=config["inventory_economics"]["daily_holding_cost_rp"],
        stockout_penalty=config["inventory_economics"]["stockout_penalty_rp"],
        salvage_value_ratio=config["inventory_economics"]["salvage_value_ratio"],
        discrete_actions=config["rl"]["action_space"]["discrete_actions"],
        seed=config["simulation"]["random_seed"],
    )

    # 4. Train Tabular Q-Learning Agent
    logger.info(f"Training Q-Learning Pricing Agent for {train_episodes} episodes...")
    q_agent = QLearningPricingAgent(
        num_actions=env.num_actions,
        learning_rate=config["rl"]["hyperparameters"]["learning_rate"],
        gamma=config["rl"]["hyperparameters"]["gamma"],
        epsilon=config["rl"]["hyperparameters"]["epsilon_start"],
        epsilon_min=config["rl"]["hyperparameters"]["epsilon_min"],
        epsilon_decay=config["rl"]["hyperparameters"]["epsilon_decay"],
        seed=config["simulation"]["random_seed"],
    )

    for ep in range(train_episodes):
        obs, _ = env.reset(seed=ep)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = q_agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            q_agent.update(obs, action, reward, next_obs, terminated)
            obs = next_obs

        q_agent.decay_epsilon()

    # Save Q-agent policy
    models_dir = "models" if os.path.exists("src") else os.path.join("..", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"q_policy_{sku}.pkl")
    q_agent.save(model_path)
    logger.info(f"Model checkpoint saved to: {model_path}")

    # 5. Benchmark Comparison
    logger.info(f"Running evaluation benchmark over {eval_episodes} test episodes...")
    backtester = PricingBacktester(env)

    agents = [
        FixedCatalogPricingAgent(action_index=3, num_actions=env.num_actions),
        RuleBasedMarkdownAgent(num_actions=env.num_actions),
        CampaignAwarePricingAgent(num_actions=env.num_actions),
        LinUCBPricingAgent(context_dim=6, num_actions=env.num_actions),
        q_agent,
    ]

    summary_df = backtester.compare_agents(agents, num_episodes=eval_episodes, base_seed=5000)

    # Format output currency
    print("\n" + "=" * 90)
    print(f"BENCHMARK RESULTS: DYNAMIC PRICING COMPARISON FOR SKU [{sku}]")
    print("=" * 90)
    display_df = summary_df.copy()
    display_df["Mean_Net_Profit"] = display_df["Mean_Net_Profit"].apply(format_currency_idr)
    display_df["Std_Net_Profit"] = display_df["Std_Net_Profit"].apply(format_currency_idr)
    display_df["Mean_GMV"] = display_df["Mean_GMV"].apply(format_currency_idr)
    display_df["Mean_ASP"] = display_df["Mean_ASP"].apply(format_currency_idr)
    display_df["Mean_Holding_Cost"] = display_df["Mean_Holding_Cost"].apply(format_currency_idr)
    display_df["Mean_Sell_Through"] = display_df["Mean_Sell_Through"].apply(lambda x: f"{x:.1f}%")
    print(display_df.to_string(index=False))
    print("=" * 90 + "\n")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Dynamic Pricing for Vintage Fashion on Shopee")
    parser.add_argument("--sku", type=str, default="V90-TEE-001", help="Target SKU")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    parser.add_argument("--eval", type=int, default=40, help="Number of evaluation episodes")
    args = parser.parse_args()

    run_training_and_benchmark(
        sku=args.sku,
        train_episodes=args.episodes,
        eval_episodes=args.eval,
    )
