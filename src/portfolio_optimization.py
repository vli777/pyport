from pathlib import Path
import sys
from typing import Any, Callable, Dict, List
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
    convert_weights_to_series,
    estimate_optimal_num_assets,
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
    df: pd.DataFrame,
    config: Config,
    start_date: str,
    end_date: str,
    symbols: List[str],
    stack: Dict,
    years: str,
):
    final_weights = None

    for model in config.models[years]:
        cache_key = make_cache_key(
            model=model,
            years=years,
            objective=config.optimization_objective,
            symbols=symbols,
        )

        # Check cache
        # cached = load_model_results_from_cache(cache_key)
        # if cached is not None:
        #     print(f"Using cached results for {model} with {years} years.")
        #     normalized_weights = normalize_weights(cached, config.min_weight)
        #     final_weights = normalized_weights
        #     stack[model + str(years)] = normalized_weights.to_dict()
        #     save_model_results_to_cache(cache_key, final_weights.to_dict())
        # else:

        # Ensure shorter-history stocks are retained
        asset_returns = np.log(df[symbols]).diff().dropna(how="all")

        # Ensure covariance matrix is computed only on valid assets
        valid_assets = asset_returns.dropna(
            thresh=int(len(asset_returns) * 0.5), axis=1
        ).columns
        asset_returns = asset_returns[valid_assets]

        # Compute covariance with aligned data
        try:
            lw = LedoitWolf()
            cov_daily = lw.fit(asset_returns).covariance_
            cov_daily = pd.DataFrame(
                cov_daily, index=valid_assets, columns=valid_assets
            )
        except ValueError as e:
            logger.error(f"Covariance computation failed: {e}")
            return pd.Series(dtype=float)

        trading_days_per_year = 252
        mu_daily = asset_returns.mean()
        mu_annual = mu_daily * trading_days_per_year
        cov_annual = cov_daily * trading_days_per_year

        # Ensure `mu` is aligned with covariance matrix
        mu_annual = mu_annual.loc[valid_assets]
        mu_annual = mu_annual.reindex(valid_assets)

        max_weight = config.max_weight
        optimization_objective = config.optimization_objective
        allow_short = config.allow_short
        risk_free_rate = config.risk_free_rate
        risk_free_rate_log_daily = np.log(1 + risk_free_rate) / trading_days_per_year

        model_args = {
            "returns": asset_returns,
            "max_weight": max_weight,
            "objective": optimization_objective,
            "allow_short": allow_short,
            "risk_free_rate": risk_free_rate_log_daily,
        }

        try:
            weights = run_optimization(
                model=model,
                cov=cov_annual,
                mu=mu_annual,
                args=model_args,
            )

            portfolio_max_size = estimate_optimal_num_assets(
                vol_limit=config.portfolio_max_vol,
                portfolio_max_size=config.portfolio_max_size,
            ) or len(valid_assets)

            weights = convert_weights_to_series(weights, index=mu_annual.index)
            normalized_weights = normalize_weights(weights, config.min_weight)
            final_weights = limit_portfolio_size(
                normalized_weights, portfolio_max_size, target_sum=1.0
            )

            # 3) Save new result to cache
            save_model_results_to_cache(cache_key, final_weights.to_dict())

            # 4) Update your stack
            stack[model + str(years)] = final_weights.to_dict()

        except Exception as e:
            logger.error(
                f"Error processing weights for {model} {optimization_objective} {years} years: {e}"
            )
            final_weights = pd.Series(dtype=float)

        # Ensure final_weights is valid
        if final_weights is None or final_weights.empty:
            logger.warning(
                f"No valid weights generated for {model} {optimization_objective} ({years} years)."
            )
            final_weights = pd.Series(dtype=float)  # Default empty Series

        # Output results for individual optimizations
        # output_results(
        #     df=df,
        #     weights=final_weights,
        #     model_name=f"{model} {config.optimization_objective}",
        #     start_date=start_date,
        #     end_date=end_date,
        #     years=years,
        #     config=config,
        # )
