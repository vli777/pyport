from pathlib import Path
from typing import Any, Callable, Dict
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from result_output import output_results
from config import Config
from models.nco import nested_clustered_optimization
from utils.caching_utils import (
    load_model_results_from_cache,
    make_cache_key,
    save_model_results_to_cache,
)
from utils.portfolio_utils import (
    normalize_weights,
)

# Dispatch table mapping model names to their corresponding optimization functions
OPTIMIZATION_DISPATCH: Dict[str, Callable[..., Any]] = {
    "nested_clustering": nested_clustered_optimization,
    # Future models can be added here
}


def run_optimization(
    model: str, cov: pd.DataFrame, mu: pd.Series, args: Dict[str, Any]
) -> Any:
    """
    Dispatch to the appropriate optimization function based on `model` and provided arguments.
    Returns weights or optimization results.
    """
    try:
        optimization_func = OPTIMIZATION_DISPATCH[model]
    except KeyError:
        raise ValueError(f"Unsupported optimization method: {model}")

    # Pass configuration parameters directly to the optimization function
    return optimization_func(cov=cov, mu=mu, **args)


def run_optimization_and_save(
    df, config: Config, start_date, end_date, symbols, stack, years
):
    for model in config.models[years]:
        cache_key = make_cache_key(model, years, symbols)

        # 1) Check cache
        cached = load_model_results_from_cache(cache_key)
        if cached is not None:
            print(f"Using cached results for {model} with {years} years.")
            normalized_weights = normalize_weights(cached, config.min_weight)
            stack[model + str(years)] = normalized_weights
        else:
            # 2) Not in cache => run optimization
            asset_returns = np.log(df[symbols]).diff().dropna()
            mu_daily = asset_returns.mean()
            lw = LedoitWolf()
            cov_daily = lw.fit(asset_returns).covariance_
            cov_daily = pd.DataFrame(
                cov_daily, index=asset_returns.columns, columns=asset_returns.columns
            )

            trading_days_per_year = 252
            mu_annual = mu_daily * trading_days_per_year
            cov_annual = cov_daily * trading_days_per_year

            # Run the single optimization model
            weights = run_optimization(
                model=model,
                cov=cov_annual,
                mu=mu_annual,
                args=config.model_config[model],
            )
            normalized_weights = normalize_weights(weights, config.min_weight)

            # 3) Save new result to cache
            save_model_results_to_cache(cache_key, normalized_weights)

            # 4) Update your stack
            stack[model + str(years)] = normalized_weights

        # Output / Print / Plot results
        output_results(
            df, normalized_weights, model, config, start_date, end_date, years
        )
