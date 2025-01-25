from pathlib import Path
import sys
from typing import Any, Callable, Dict
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from result_output import output_results
from config import Config
from models.nco import nested_clustered_optimization
from utils import logger
from utils.caching_utils import (
    load_model_results_from_cache,
    make_cache_key,
    save_model_results_to_cache,
)
from utils.portfolio_utils import (
    limit_portfolio_size,
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
    final_weights = None

    for model in config.models[years]:
        cache_key = make_cache_key(model, years, symbols)

        # 1) Check cache
        cached = load_model_results_from_cache(cache_key)
        if cached is not None:
            print(f"Using cached results for {model} with {years} years.")
            normalized_weights = normalize_weights(cached, config.min_weight)
            final_weights = normalized_weights
            # Convert the Series to a dict before storing in stack
            stack[model + str(years)] = normalized_weights.to_dict()
            save_model_results_to_cache(cache_key, final_weights.to_dict())
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

            max_weight = config.max_weight

            model_args = config.model_config[model]
            model_args.update({"max_weight": max_weight})

            try:
                weights = run_optimization(
                    model=model,
                    cov=cov_annual,
                    mu=mu_annual,
                    args=model_args,
                )

                # Explicitly convert weights to a Pandas Series if it's a dictionary
                if isinstance(weights, dict):
                    weights = pd.Series(weights)

                normalized_weights = normalize_weights(weights, config.min_weight)

                final_weights = limit_portfolio_size(
                    normalized_weights, config.portfolio_max_size, target_sum=1.0
                )

                # 3) Save new result to cache
                save_model_results_to_cache(cache_key, final_weights.to_dict())

                # 4) Update your stack
                stack[model + str(years)] = final_weights.to_dict()

            except Exception as e:
                logger.error(
                    f"Error processing weights for {model} with {years} years: {e}"
                )
                final_weights = pd.Series(dtype=float)

        # Ensure final_weights is valid
        if final_weights is None or final_weights.empty:
            logger.warning(f"No valid weights generated for {model} ({years} years).")
            final_weights = pd.Series(dtype=float)  # Default empty Series

        # Output / Print / Plot results
        output_results(df, final_weights, model, config, start_date, end_date, years)
