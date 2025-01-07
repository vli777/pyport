# src/config.py

import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any
import os

@dataclass
class OptimizationConfig:
    hrp: Dict[str, Any]
    herc: Dict[str, Any]
    nco: Dict[str, Any]
    nco2: Dict[str, Any]
    mc: Dict[str, Any]
    mc2: Dict[str, Any]
    olmar: Dict[str, Any]
    rmr: Dict[str, Any]
    scorn: Dict[str, Any]
    fcornk: Dict[str, Any]
    cla: Dict[str, Any]
    cla2: Dict[str, Any]
    efficient_risk: float
    efficient_return: float
    risk_aversion: float

@dataclass
class Config:
    folder: str
    input_files_folder: str
    input_files: List[str]
    models: Dict[str, List[str]]
    download: bool
    plot_daily_returns: bool
    plot_cumulative_returns: bool
    min_weight: float
    portfolio_max_size: int
    sort_by_weights: bool
    verbose: bool
    use_short: bool
    test_mode: bool
    test_data_visible_pct: float
    optimization_config: OptimizationConfig

    @classmethod
    def from_yaml(cls, config_file: str) -> 'Config':
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} does not exist.")

        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Parse nested optimization_config
        optimization_config = OptimizationConfig(**config_dict.get("optimization_config", {}))

        return cls(
            folder=config_dict["folder"],
            input_files_folder=config_dict.get("input_files_folder", "watchlists"),
            input_files=config_dict["input_files"],
            models=config_dict["models"],
            download=config_dict.get("download", False),
            plot_daily_returns=config_dict.get("plot_daily_returns", False),
            plot_cumulative_returns=config_dict.get("plot_cumulative_returns", False),
            min_weight=config_dict["min_weight"],
            portfolio_max_size=config_dict["portfolio_max_size"],
            sort_by_weights=config_dict.get("sort_by_weights", False),
            verbose=config_dict.get("verbose", False),
            use_short=config_dict.get("use_short", False),
            test_mode=config_dict.get("test_mode", False),
            test_data_visible_pct=config_dict["test_data_visible_pct"],
            optimization_config=optimization_config
        )
