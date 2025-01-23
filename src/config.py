# src/config.py

import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any
import os


@dataclass
class ModelConfig:
    nested_clustering: Dict[str, Any]

    def __getitem__(self, key: str) -> Dict[str, Any]:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"{key} not found in ModelConfig")


@dataclass
class Config:
    folder: str
    input_files_folder: str
    input_files: List[str]
    models: Dict[str, List[str]]
    download: bool
    plot_daily_returns: bool
    plot_cumulative_returns: bool
    plot_clustering: bool
    plot_anomalies: bool
    plot_mean_reversion: bool
    anomaly_detection_deviation_threshold: float
    correlation_threshold: float
    expand_etfs: bool
    min_weight: float
    max_weight: float
    portfolio_max_size: int
    risk_free_rate: float
    sort_by_weights: bool
    allow_short: bool
    short_long_ratio: float
    test_mode: bool
    test_data_visible_pct: float
    model_config: ModelConfig
    expand_etfs: bool

    @classmethod
    def from_yaml(cls, config_file: str) -> "Config":
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} does not exist.")

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Parse model config
        model_config = ModelConfig(**config_dict.get("model_config", {}))

        return cls(
            folder=config_dict["folder"],
            input_files_folder=config_dict.get("input_files_folder", "watchlists"),
            input_files=config_dict["input_files"],
            models=config_dict["models"],
            download=config_dict.get("download", False),
            plot_daily_returns=config_dict.get("plot_daily_returns", False),
            plot_cumulative_returns=config_dict.get("plot_cumulative_returns", False),
            plot_clustering=config_dict.get("plot_clustering", False),
            plot_anomalies=config_dict.get("plot_anomalies", False),
            plot_mean_reversion=config_dict.get("plot_mean_reversion", False),
            anomaly_detection_deviation_threshold=config_dict.get(
                "anomaly_detection_deviation_threshold", 7.0
            ),
            correlation_threshold=config_dict.get("correlation_threshold", 0.8),
            expand_etfs=config_dict.get("expand_etfs", False),
            min_weight=config_dict["min_weight"],
            max_weight=config_dict["max_weight"],
            portfolio_max_size=config_dict["portfolio_max_size"],
            risk_free_rate=config_dict.get("risk_free_rate", 0.0),
            allow_short=config_dict.get("allow_short", False),
            short_long_ratio=config_dict.get("short_long_ratio", 0.3),
            sort_by_weights=config_dict.get("sort_by_weights", False),
            test_mode=config_dict.get("test_mode", False),
            test_data_visible_pct=config_dict["test_data_visible_pct"],
            model_config=model_config,
        )
