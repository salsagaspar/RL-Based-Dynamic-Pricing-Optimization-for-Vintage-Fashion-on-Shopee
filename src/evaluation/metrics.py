"""Evaluation Metrics for Dynamic Pricing Performance and Retail Economics."""

from typing import Dict, List, Any
import numpy as np


def compute_financial_metrics(history: List[Dict[str, Any]], initial_stock: int, unit_cost: float) -> Dict[str, float]:
    """Computes comprehensive business, inventory, and revenue metrics from an episode trajectory.

    Args:
        history: List of step info dictionaries from environment.
        initial_stock: Initial inventory units at day 0.
        unit_cost: Cost per unit (Modal_Unit_Rp).

    Returns:
        Dict of computed financial and operational KPIs.
    """
    total_sold = sum(h["units_sold"] for h in history)
    total_gmv = sum(h["units_sold"] * h["offered_price"] for h in history)
    total_holding_cost = sum(h.get("daily_holding_cost", 0.0) for h in history)
    total_stockout_penalty = sum(h.get("stockout_penalty", 0.0) for h in history)
    total_net_profit = history[-1]["cumulative_profit"] if history else 0.0

    cogs_realized = total_sold * unit_cost
    ending_stock = history[-1]["stock_remaining"] if history else initial_stock
    sell_through_rate = (total_sold / max(1, initial_stock)) * 100.0

    avg_selling_price = (total_gmv / total_sold) if total_sold > 0 else 0.0
    net_margin_pct = (total_net_profit / total_gmv * 100.0) if total_gmv > 0 else 0.0

    return {
        "Total_GMV_Rp": float(total_gmv),
        "Total_Net_Profit_Rp": float(total_net_profit),
        "Total_Units_Sold": int(total_sold),
        "Ending_Stock_Units": int(ending_stock),
        "Sell_Through_Rate_%": float(sell_through_rate),
        "Avg_Selling_Price_Rp": float(avg_selling_price),
        "Total_Holding_Cost_Rp": float(total_holding_cost),
        "Total_Stockout_Penalty_Rp": float(total_stockout_penalty),
        "Net_Margin_%": float(net_margin_pct),
    }
