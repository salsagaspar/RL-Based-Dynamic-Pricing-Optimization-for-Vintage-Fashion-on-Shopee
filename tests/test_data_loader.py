"""Unit tests for Data Loader."""

import pytest
import os
from src.data.loader import ShopeeDatasetLoader


def test_loader_loads_sheets():
    loader = ShopeeDatasetLoader()
    sheets = loader.load_all_sheets()

    expected_sheets = ["products", "suppliers", "purchases", "orders", "returns", "inventory"]
    for s in expected_sheets:
        assert s in sheets, f"Sheet '{s}' missing from loaded data."
        assert not sheets[s].empty, f"Sheet '{s}' is unexpectedly empty."


def test_sku_catalog_merging():
    loader = ShopeeDatasetLoader()
    catalog = loader.get_sku_catalog()

    assert not catalog.empty
    assert "SKU" in catalog.columns
    assert "Harga_Jual_Rp" in catalog.columns
    assert "Modal_Unit_Rp" in catalog.columns
    assert "Stok_Akhir" in catalog.columns
