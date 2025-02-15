from typing import Dict, Tuple
import numpy as np
import pandas as pd

from reversion.mean_reversion import apply_mean_reversion
from boxplot import generate_boxplot_data
from result_output import output
from config import Config
from stat_arb.apply_adaptive_weighting import apply_adaptive_weighting
from stat_arb.multi_asset_plots import plot_multi_asset_signals
from stat_arb.multi_asset_reversion import MultiAssetReversion
from stat_arb.portfolio_allocator import PortfolioAllocator
from stat_arb.single_asset_reversion import OUHeatPotential
from utils.performance_metrics import risk_return_contributions
from utils.portfolio_utils import normalize_weights
from utils.logger import logger


def apply_z_reversion(
    dfs: dict,
    normalized_avg_weights: dict,
    combined_input_files: str,
    combined_models: str,
    sorted_time_periods: list,
    config: Config,
    asset_cluster_map: dict,
    returns_df: pd.DataFrame,
) -> dict:
    """
    Applies Z-score-based mean reversion to update the portfolio weights
    and returns a final result dictionary.
    """
    logger.info("\nApplying Z-score-based mean reversion on normalized weights...")
    # Compute mean-reverted weights
    mean_reverted_weights = apply_mean_reversion(
        asset_cluster_map=asset_cluster_map,
        baseline_allocation=normalized_avg_weights,
        returns_df=returns_df,
        config=config,
        cache_dir="optuna_cache",
    )
    sorted_symbols_post = sorted(mean_reverted_weights.keys())
    # Filter the data to only include the post-reversion symbols
    dfs["data"] = dfs["data"].filter(items=sorted_symbols_post)

    # Compute performance metrics using the common helper
    (
        post_daily_returns,
        post_cumulative_returns,
        post_boxplot_stats,
        return_contributions_pct,
        risk_contributions_pct,
    ) = compute_performance_results(
        data=dfs["data"],
        start_date=str(dfs["start"]),
        end_date=str(dfs["end"]),
        allocation_weights=mean_reverted_weights,
        sorted_symbols=sorted_symbols_post,
        combined_input_files=combined_input_files,
        combined_models=combined_models,
        sorted_time_periods=sorted_time_periods,
        config=config,
    )

    # Build and return the final result dictionary
    return build_final_result_dict(
        start_date=str(dfs["start"]),
        end_date=str(dfs["end"]),
        models=combined_models,
        symbols=sorted_symbols_post,
        normalized_avg=mean_reverted_weights,
        daily_returns=post_daily_returns,
        cumulative_returns=post_cumulative_returns,
        boxplot_stats=post_boxplot_stats,
        return_contributions=return_contributions_pct,
        risk_contributions=risk_contributions_pct,
    )


def apply_ou_reversion(
    dfs: dict,
    normalized_avg_weights: dict,
    combined_input_files: str,
    combined_models: str,
    sorted_time_periods: list,
    config: Config,
    returns_df: pd.DataFrame,
) -> dict:
    """
    Applies OU-based (heat potential) mean reversion and returns the final
    result dictionary.
    """
    logger.info("\nApplying OU-based mean reversion...")
    # --- Prepare individual OU strategies for each ticker
    ou_strategies = {
        ticker: OUHeatPotential(dfs["data"][ticker], returns_df[ticker])
        for ticker in dfs["data"].columns
    }
    ou_signals = {
        ticker: ou.generate_trading_signals() for ticker, ou in ou_strategies.items()
    }
    ou_results = {
        ticker: ou.simulate_strategy(ou_signals[ticker])
        for ticker, ou in ou_strategies.items()
    }

    individual_returns = {
        ticker: pd.Series(result[0]).reset_index(drop=True)
        for ticker, result in ou_results.items()
    }
    max_len = max(s.size for s in individual_returns.values())
    individual_returns = {
        ticker: s.reindex(range(max_len), fill_value=0)
        for ticker, s in individual_returns.items()
    }

    # --- Cross-asset (multi-asset) mean reversion
    multi_asset_strategy = MultiAssetReversion(dfs["data"])
    multi_asset_results = multi_asset_strategy.optimize_and_trade()

    weights_series = pd.Series(multi_asset_results["Hedge Ratios"]).reindex(
        dfs["data"].columns, fill_value=0
    )
    if weights_series.sum() != 0:
        weights_series /= weights_series.abs().sum()

    basket_returns = (
        dfs["data"].pct_change().fillna(0).mul(weights_series, axis=1).sum(axis=1)
    )
    multi_asset_returns = (
        multi_asset_results["Signals"]["Position"].shift(1) * basket_returns
    )
    multi_asset_returns = multi_asset_returns.fillna(0)

    if config.plot_reversion and config.test_mode:
        plot_multi_asset_signals(
            spread_series=multi_asset_strategy.spread_series,
            signals=multi_asset_results["Signals"],
            title="Multi-Asset Mean Reversion Trading Signals",
        )

    portfolio_allocator = PortfolioAllocator()
    reversion_allocations = portfolio_allocator.compute_allocations(
        individual_returns,
        multi_asset_returns,
        hedge_ratios=multi_asset_results["Hedge Ratios"],
    )
    normalized_reversion_allocations = normalize_weights(reversion_allocations)
    stat_arb_adjusted_allocation = apply_adaptive_weighting(
        baseline_allocation=normalized_avg_weights,
        mean_reversion_weights=normalized_reversion_allocations,
        returns_df=returns_df,
        base_alpha=0.2,
    )
    sorted_stat_arb_allocation = stat_arb_adjusted_allocation.sort_values(
        ascending=False
    )

    # Compute performance with the adjusted (OU-based) allocations.
    # (Assuming that the symbols remain the same; otherwise, update as needed.)
    (
        adjusted_daily_returns,
        adjusted_cumulative_returns,
        adjusted_boxplot_stats,
        return_contributions_pct,
        risk_contributions_pct,
    ) = compute_performance_results(
        data=dfs["data"],
        start_date=str(dfs["start"]),
        end_date=str(dfs["end"]),
        allocation_weights=stat_arb_adjusted_allocation,
        sorted_symbols=sorted(normalized_avg_weights.keys()),
        combined_input_files=combined_input_files,
        combined_models=combined_models,
        sorted_time_periods=sorted_time_periods,
        config=config,
    )

    return build_final_result_dict(
        start_date=str(dfs["start"]),
        end_date=str(dfs["end"]),
        models=combined_models,
        symbols=sorted(normalized_avg_weights.keys()),
        normalized_avg=sorted_stat_arb_allocation,
        daily_returns=adjusted_daily_returns,
        cumulative_returns=adjusted_cumulative_returns,
        boxplot_stats=adjusted_boxplot_stats,
        return_contributions=return_contributions_pct,
        risk_contributions=risk_contributions_pct,
    )


def compute_performance_results(
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
    allocation_weights: Dict[str, float],
    sorted_symbols: list,
    combined_input_files: str,
    combined_models: str,
    sorted_time_periods: list,
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Compute daily returns, cumulative returns, boxplot stats, and risk/return contributions.

    Returns:
    daily_returns, cumulative_returns, boxplot_stats, return_contributions_pct, risk_contributions_pct
    """
    # Compute daily & cumulative returns
    daily_returns, cumulative_returns = output(
        data=data,
        start_date=start_date,
        end_date=end_date,
        allocation_weights=allocation_weights,
        inputs=combined_input_files,
        optimization_model=f"{combined_models} ENSEMBLE: {config.optimization_objective}",
        time_period=sorted_time_periods[0],
        config=config,
    )

    boxplot_stats = generate_boxplot_data(daily_returns)

    # Ensure only stocks present in cumulative_returns are used
    valid_stocks = [s for s in sorted_symbols if s in cumulative_returns.columns]
    np_weights = np.array([allocation_weights[s] for s in valid_stocks])
    # Extract final cumulative returns per stock
    contribution_cumulative_returns = cumulative_returns.loc[
        cumulative_returns.index[-1], valid_stocks
    ].values

    # Compute risk and return contributions
    return_contributions_pct, risk_contributions_pct = risk_return_contributions(
        weights=np_weights,
        daily_returns=daily_returns[valid_stocks],
        cumulative_returns=contribution_cumulative_returns,
    )

    return (
        daily_returns,
        cumulative_returns,
        boxplot_stats,
        return_contributions_pct,
        risk_contributions_pct,
    )


def build_final_result_dict(
    start_date: str,
    end_date: str,
    models: str,
    symbols: list,
    normalized_avg: dict,
    daily_returns: pd.DataFrame,
    cumulative_returns: pd.DataFrame,
    boxplot_stats,
    return_contributions: np.ndarray,
    risk_contributions: np.ndarray,
) -> dict:
    """
    Constructs the final results dictionary.
    """
    return {
        "start_date": start_date,
        "end_date": end_date,
        "models": models,
        "symbols": symbols,
        "normalized_avg": normalized_avg,
        "daily_returns": daily_returns,
        "cumulative_returns": cumulative_returns,
        "boxplot_stats": boxplot_stats,
        "return_contributions": return_contributions,
        "risk_contributions": risk_contributions,
    }
