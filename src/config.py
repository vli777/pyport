# src/config.py

import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any
import os


@dataclass
class ModelConfig:
    nested_clustering: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Dict[str, Any]:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"{key} not found in ModelConfig")


@dataclass
class Config:
    data_dir: str
    input_files_dir: str
    input_files: List[str]

    download: bool
    include_etf_top_holdings: bool
    min_weight: float
    max_weight: float
    portfolio_max_size: int
    risk_free_rate: float

    plot_daily_returns: bool
    plot_cumulative_returns: bool
    plot_clustering: bool
    plot_anomalies: bool
    plot_reversion_threshold: bool

    top_n_candidates: int

    use_reversion_filter: bool
    use_signal_filter: bool
    use_anomaly_filter: bool
    use_correlation_filter: bool

    test_mode: bool
    test_data_visible_pct: float
    model_config: ModelConfig = field(default_factory=ModelConfig)
    models: Dict[str, List[str]] = field(
        default_factory=lambda: {"1.00": ["nested_clustering"]}
    )

    @classmethod
    def from_yaml(cls, config_file: str) -> "Config":
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} does not exist.")

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        # Ensure default values for missing or empty keys
        config_dict["models"] = config_dict.get(
            "models", {"1.00": ["nested_clustering"]}
        )
        config_dict["model_config"] = config_dict.get("model_config", {})

        data_dir = config_dict["data_dir"]
        input_files_dir = config_dict.get("input_files_dir", "watchlists")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(input_files_dir, exist_ok=True)

        # Parse model config
        model_config = ModelConfig(**config_dict["model_config"])

        return cls(
            data_dir=data_dir,
            input_files_dir=input_files_dir,
            input_files=config_dict["input_files"],
            models=config_dict["models"],
            download=config_dict.get("download", False),
            include_etf_top_holdings=config_dict.get("include_etf_top_holdings", False),
            min_weight=config_dict.get("min_weight", 0.01),
            max_weight=config_dict.get("max_weight", 1.0),
            portfolio_max_size=config_dict.get("portfolio_max_size", 20),
            risk_free_rate=config_dict.get("risk_free_rate", 0.0),
            plot_daily_returns=config_dict.get("plot_daily_returns", False),
            plot_cumulative_returns=config_dict.get("plot_cumulative_returns", False),
            plot_clustering=config_dict.get("plot_clustering", False),
            plot_anomalies=config_dict.get("plot_anomalies", False),
            plot_reversion_threshold=config_dict.get("plot_reversion_threshold", False),
            top_n_candidates=config_dict.get("top_n_candidates", None),
            use_reversion_filter=config_dict.get("use_reversion_filter", True),
            use_signal_filter=config_dict.get("use_signal_filter", True),
            use_anomaly_filter=config_dict.get("use_anomaly_filter", True),
            use_correlation_filter=config_dict.get("use_correlation_filter", True),
            test_mode=config_dict.get("test_mode", False),
            test_data_visible_pct=config_dict.get("test_data_visible_pct", 0.1),
            model_config=model_config,
        )
