"""Utility functions for config loading, logging, and currency formatting."""

import os
import random
import logging
import yaml
import numpy as np


def setup_logger(name: str = "pricing_rl", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standard application logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Loads configuration from YAML file."""
    if not os.path.exists(config_path):
        # Fallback if run from subfolder
        parent_config = os.path.join("..", config_path)
        if os.path.exists(parent_config):
            config_path = parent_config
        else:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def format_currency_idr(amount: float) -> str:
    """Formats numeric value to Indonesian Rupiah string (e.g., Rp 150.000)."""
    return f"Rp {amount:,.0f}".replace(",", ".")
