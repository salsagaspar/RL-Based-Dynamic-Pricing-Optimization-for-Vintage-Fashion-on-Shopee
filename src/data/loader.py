"""Data Loader Module for Vintage Fashion Supply Chain & Sales on Shopee.
Handles reading and parsing multi-sheet Excel files.
"""

import os
from typing import Dict, Tuple, Optional
import pandas as pd
from src.utils.helpers import setup_logger

logger = setup_logger("data_loader")


class ShopeeDatasetLoader:
    """Loads and standardizes supply chain and order data from Shopee Excel dataset."""

    def __init__(self, excel_path: str = "data/raw/dataset_supply_chain_pakaian_90an_shopee.xlsx"):
        candidates = [
            excel_path,
            os.path.join("..", excel_path),
            os.path.join("..", "..", excel_path),
            os.path.basename(excel_path),
            os.path.join("..", os.path.basename(excel_path)),
            os.path.join("data", "raw", os.path.basename(excel_path)),
            os.path.join("..", "data", "raw", os.path.basename(excel_path)),
        ]
        resolved_path = None
        for candidate in candidates:
            if os.path.exists(candidate):
                resolved_path = candidate
                break

        if resolved_path is None:
            raise FileNotFoundError(f"Dataset file not found at: {excel_path} (checked paths: {candidates})")
        self.excel_path = resolved_path

    def load_all_sheets(self) -> Dict[str, pd.DataFrame]:
        """Loads all 6 sheets from the Excel workbook.

        Returns:
            Dict containing DataFrames for:
            - 'products' (Produk_Master)
            - 'suppliers' (Pemasok)
            - 'purchases' (Pembelian)
            - 'orders' (Pesanan)
            - 'returns' (Retur)
            - 'inventory' (Persediaan)
        """
        logger.info(f"Loading dataset from: {self.excel_path}")
        xls = pd.ExcelFile(self.excel_path)

        sheets = {
            "products": xls.parse("Produk_Master"),
            "suppliers": xls.parse("Pemasok"),
            "purchases": xls.parse("Pembelian"),
            "orders": xls.parse("Pesanan"),
            "returns": xls.parse("Retur"),
            "inventory": xls.parse("Persediaan"),
        }

        # Date parsing
        date_cols_map = {
            "purchases": ["Tanggal_PO", "Tanggal_Terima"],
            "orders": ["Tanggal_Order", "Tanggal_Dikirim", "Tanggal_Diterima"],
            "returns": ["Tanggal_Retur"],
        }
        for sheet_key, date_cols in date_cols_map.items():
            for col in date_cols:
                if col in sheets[sheet_key].columns:
                    sheets[sheet_key][col] = pd.to_datetime(sheets[sheet_key][col], errors="coerce")

        logger.info(
            f"Successfully loaded {len(sheets)} sheets. "
            f"Products: {len(sheets['products'])}, "
            f"Orders: {len(sheets['orders'])}, "
            f"Inventory: {len(sheets['inventory'])}"
        )
        return sheets

    def get_sku_catalog(self) -> pd.DataFrame:
        """Merges Produk_Master with latest Persediaan (inventory) to produce active catalog."""
        sheets = self.load_all_sheets()
        products = sheets["products"]
        inventory = sheets["inventory"]

        # Merge on SKU
        catalog = pd.merge(
            products,
            inventory[["SKU", "Stok_Akhir", "Demand_Rata2_30d_Unit_per_Hari", "Reorder_Point_Unit", "Status_Stok"]],
            on="SKU",
            how="left"
        )
        return catalog


def load_data(excel_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Helper functional interface to load all sheets."""
    loader = ShopeeDatasetLoader(excel_path) if excel_path else ShopeeDatasetLoader()
    return loader.load_all_sheets()
