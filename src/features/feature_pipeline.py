"""Feature Engineering Pipeline for Vintage Fashion Dynamic Pricing.
Computes SKU-level economic metrics, inventory states, and Shopee fee-adjusted margins.
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np


class PricingFeatureEngineer:
    """Computes features required for the dynamic pricing state space and simulation."""

    def __init__(self, platform_fee_pct: float = 0.09, daily_holding_cost_rp: float = 350.0):
        self.platform_fee_pct = platform_fee_pct
        self.daily_holding_cost_rp = daily_holding_cost_rp

    def build_sku_features(self, sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Constructs an integrated SKU-level feature table for RL state representation."""
        products = sheets["products"].copy()
        inventory = sheets["inventory"].copy()
        orders = sheets["orders"].copy()
        returns = sheets["returns"].copy()

        # 1. Aggregate Order Stats per SKU
        order_stats = orders.groupby("SKU").agg(
            total_orders=("Order_ID", "nunique"),
            total_qty_sold=("Qty", "sum"),
            avg_sold_price=("Harga_Jual_Unit_Rp", "mean"),
            avg_discount=("Diskon_Rp", "mean"),
            total_gross_omzet=("Omzet_Kotor_Rp", "sum"),
            total_net_omzet=("Omzet_Neto_Estimasi_Rp", "sum"),
            total_realized_margin=("Margin_Estimasi_Rp", "sum")
        ).reset_index()

        # 2. Aggregate Return Stats per SKU
        return_stats = returns.groupby("SKU").agg(
            total_returns=("Qty_Retur", "sum"),
            total_refund_rp=("Nilai_Refund_Rp", "sum")
        ).reset_index()

        # 3. Merge with Master & Inventory
        df = pd.merge(products, inventory[["SKU", "Stok_Awal_01Sep", "Stok_Akhir", "Demand_Rata2_30d_Unit_per_Hari", "Reorder_Point_Unit", "Status_Stok"]], on="SKU", how="left")
        df = pd.merge(df, order_stats, on="SKU", how="left").fillna(0)
        df = pd.merge(df, return_stats, on="SKU", how="left").fillna(0)

        # 4. Computed Pricing & Inventory Economics Features
        df["Return_Rate_%"] = np.where(
            df["total_qty_sold"] > 0,
            (df["total_returns"] / df["total_qty_sold"]) * 100,
            0.0
        )

        # Estimated Daily Demand Velocity
        df["Daily_Velocity"] = df["Demand_Rata2_30d_Unit_per_Hari"].clip(lower=0.1)

        # Days of Inventory Remaining (DOI)
        df["Days_Of_Inventory"] = df["Stok_Akhir"] / df["Daily_Velocity"]

        # Stock Scarcity Ratio (Stock / Safety_Stock)
        df["Stock_to_Safety_Ratio"] = df["Stok_Akhir"] / df["Safety_Stock_Unit"].clip(lower=1)

        # Unit Economics at Baseline Catalog Price
        # Shopee Platform Fee: Price * Fee%
        df["Shopee_Fee_Unit_Rp"] = df["Harga_Jual_Rp"] * self.platform_fee_pct
        # Net Margin = (Price - Fee - Cost) / Price
        df["Net_Unit_Profit_Rp"] = df["Harga_Jual_Rp"] * (1.0 - self.platform_fee_pct) - df["Modal_Unit_Rp"]
        df["Net_Margin_%"] = (df["Net_Unit_Profit_Rp"] / df["Harga_Jual_Rp"]) * 100

        # Break-Even Minimum Price (where Net Unit Profit = 0)
        df["Break_Even_Price_Rp"] = df["Modal_Unit_Rp"] / (1.0 - self.platform_fee_pct)

        # Category Demand Elasticity Estimation Proxy
        # Grunge & Rare Jackets tend to be more inelastic; standard tees more elastic
        elasticity_map = {
            "Kaos": -1.8,
            "Kemeja": -1.4,
            "Jaket": -1.1,
            "Celana": -1.3,
            "Aksesoris": -1.6
        }
        df["Estimated_Price_Elasticity"] = df["Kategori"].map(elasticity_map).fillna(-1.5)

        return df
