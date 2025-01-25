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
    data_dir: str
    input_files_dir: str
    input_files: List[str]
    models: Dict[str, List[str]]

    download: bool
    expand_etfs: bool
    min_weight: float
    max_weight: float
    portfolio_max_size: int
    risk_free_rate: float

    plot_daily_returns: bool
    plot_cumulative_returns: bool
    plot_clustering: bool
    plot_anomalies: bool
    plot_mean_reversion: bool
    plot_signal_threshold: bool

    buy_signal_threshold: float
    sell_signal_threshold: float
    anomaly_detection_deviation_threshold: float
    correlation_threshold: float

    use_reversion_filter: bool
    use_signal_filter: bool
    use_anomaly_filter: bool

    test_mode: bool
    test_data_visible_pct: float
    model_config: ModelConfig

    @classmethod
    def from_yaml(cls, config_file: str) -> "Config":
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} does not exist.")

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        data_dir = config_dict["data_dir"]
        input_files_dir = config_dict.get("input_files_dir", "watchlists")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(input_files_dir, exist_ok=True)

        # Parse model config
        model_config = ModelConfig(**config_dict.get("model_config", {}))

        return cls(
            data_dir=config_dict["data_dir"],
            input_files_dir=config_dict.get("input_files_dir", "watchlists"),
            input_files=config_dict["input_files"],
            models=config_dict["models"],
            download=config_dict.get("download", False),
            expand_etfs=config_dict.get("expand_etfs", False),
            min_weight=config_dict["min_weight"],
            max_weight=config_dict["max_weight"],
            portfolio_max_size=config_dict["portfolio_max_size"],
            risk_free_rate=config_dict.get("risk_free_rate", 0.0),
            plot_daily_returns=config_dict.get("plot_daily_returns", False),
            plot_cumulative_returns=config_dict.get("plot_cumulative_returns", False),
            plot_clustering=config_dict.get("plot_clustering", False),
            plot_anomalies=config_dict.get("plot_anomalies", False),
            plot_mean_reversion=config_dict.get("plot_mean_reversion", False),
            plot_signal_threshold=config_dict.get("plot_signal_threshold", False),
            buy_signal_threshold=config_dict.get("buy_signal_threshold", 1.0),
            sell_signal_threshold=config_dict.get("sell_signal_threshold", 1.0),
            anomaly_detection_deviation_threshold=config_dict.get(
                "anomaly_detection_deviation_threshold", 7.0
            ),
            correlation_threshold=config_dict.get("correlation_threshold", 0.8),
            use_reversion_filter=config_dict.get("use_reversion_filter", True),
            use_signal_filter=config_dict.get("use_signal_filter", True),
            use_anomaly_filter=config_dict.get("use_anomaly_filter", True),
            test_mode=config_dict.get("test_mode", False),
            test_data_visible_pct=config_dict["test_data_visible_pct"],
            model_config=model_config,
        )
