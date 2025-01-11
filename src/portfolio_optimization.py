import json
from pathlib import Path
import numpy as np
from portfoliolab.clustering.herc import HierarchicalEqualRiskContribution
from portfoliolab.clustering.hrp import HierarchicalRiskParity
from portfoliolab.modern_portfolio_theory.mean_variance import MeanVarianceOptimisation
from portfoliolab.modern_portfolio_theory.mean_variance import ReturnsEstimators
from portfoliolab.modern_portfolio_theory import CriticalLineAlgorithm
from portfoliolab.clustering.nco import NestedClusteredOptimisation
from portfoliolab.online_portfolio_selection.rmr import RMR
from portfoliolab.online_portfolio_selection.olmar import OLMAR
from portfoliolab.online_portfolio_selection.fcornk import FCORNK
from portfoliolab.online_portfolio_selection.scorn import SCORN

from result_output import output_results
from config import Config
from utils.portfolio_utils import convert_to_dict, normalize_weights


def run_optimization(method, df, config: Config):
    """
    Dispatch to the appropriate optimization class or function
    based on `method` and `config`.
    Returns an optimizer instance or something containing .weights
    """
    OPTIMIZATION_METHODS = {
        "hrp": HierarchicalRiskParity,
        "herc": HierarchicalEqualRiskContribution,
        "nco": NestedClusteredOptimisation,
        "mc": NestedClusteredOptimisation,
        "mc2": NestedClusteredOptimisation,
        "cla": CriticalLineAlgorithm,
        "cla2": CriticalLineAlgorithm,
        "olmar": OLMAR,
        "rmr": RMR,
        "scorn": SCORN,
        "fcornk": FCORNK,
        "mean_variance": MeanVarianceOptimisation,
    }

    optimizer_class = OPTIMIZATION_METHODS.get(method)
    if not optimizer_class:
        raise ValueError(f"Unknown optimization method: {method}")

    # Retrieve the dictionary for this method from optimization_config
    # e.g., for method="mc", this will be config.optimization_config.mc (a Dict[str, Any])
    method_config = getattr(config.optimization_config, method, {})

    # 1) Initialize the correct optimizer class
    if method == "olmar":
        optimizer = optimizer_class(
            reversion_method=method_config["method"],
            epsilon=method_config["epsilon"],
            window=method_config["window"],
            alpha=method_config["alpha"],
        )
    elif method == "rmr":
        optimizer = optimizer_class(
            epsilon=method_config["epsilon"],
            n_iteration=method_config["n_iteration"],
            tau=method_config["tau"],
            window=method_config["window"],
        )
    elif method == "scorn":
        optimizer = optimizer_class(
            window=method_config["window"],
            rho=method_config["rho"],
        )
    elif method == "fcornk":
        optimizer = optimizer_class(
            window=method_config["window"],
            rho=method_config["rho"],
            lambd=method_config["lambd"],
            k=method_config["k"],
        )
    else:
        # For methods that require no special init arguments
        optimizer = optimizer_class()

    # 2) Handle method-specific allocation logic
    if method == "nco":
        asset_returns = np.log(df) - np.log(df.shift(1))
        asset_returns = asset_returns.iloc[1:, :]

        if method_config["sharpe"]:
            mu_vec = np.array(asset_returns.mean())
        else:
            mu_vec = np.ones(len(df.columns))

        weights = optimizer.allocate_nco(
            asset_names=df.columns,
            cov=np.array(asset_returns.cov()),
            mu_vec=mu_vec.reshape(-1, 1),
        )
        optimizer.weights = weights

    elif method in ["mc", "mc2"]:
        asset_returns = np.log(df) - np.log(df.shift(1))
        asset_returns = asset_returns.iloc[1:, :]
        mu_vec = np.array(asset_returns.mean())

        w_cvo, w_nco = optimizer.allocate_mcos(
            mu_vec=mu_vec.reshape(-1, 1),
            cov=np.array(asset_returns.cov()),
            num_obs=method_config["num_obs"],
            num_sims=method_config["num_sims"],
            kde_bwidth=method_config["kde_bandwidth"],
            min_var_portf=not method_config["sharpe"],
            lw_shrinkage=method_config["lw_shrinkage"],
        )
        w_nco = w_nco.mean(axis=0)
        optimizer.weights = dict(zip(df.columns, w_nco))

    elif method in ["cla", "cla2"]:
        solution = method_config["solution"]
        optimizer.allocate(asset_prices=df, solution=solution)

    elif method == "mean_variance":
        expected_returns = ReturnsEstimators().calculate_mean_historical_returns(
            asset_prices=df
        )
        covariance = ReturnsEstimators().calculate_returns(asset_prices=df).cov()

        # Top-level optimization fields like efficient_risk, etc. are floats
        optimizer.allocate(
            asset_names=df.columns,
            asset_prices=df,
            expected_asset_returns=expected_returns,
            covariance_matrix=covariance,
            solution=method,
            target_return=config.optimization_config.efficient_risk,
            target_risk=config.optimization_config.efficient_return,
            risk_aversion=config.optimization_config.risk_aversion,
        )
        optimizer.get_portfolio_metrics()

    elif method in ["fcornk", "scorn", "rmr", "olmar"]:
        # Some of these have special initialization above, but also allow a `resample` param
        # Safely get the 'resample' parameter, defaulting to None if not present
        resample_by = method_config.get("resample", None)

        if resample_by:
            optimizer.allocate(asset_prices=df, resample_by=resample_by)
        else:
            optimizer.allocate(asset_prices=df)

    # 3) Generic fallback (rare case)
    else:
        # If there's any leftover method that doesn't need special logic,
        # pass all dictionary keys as kwargs:
        optimizer.allocate(asset_prices=df, **method_config)

    return optimizer


def make_cache_key(method, years, symbols, config_hash):
    # config_hash can be an MD5 of the config dictionary or something
    # e.g. str(sorted(config.items()))
    sorted_symbols = "_".join(sorted(symbols))
    return f"{method}_{years}_{sorted_symbols}_{config_hash}.json"


def load_model_results_from_cache(cache_key):
    cache_file = Path("cache") / cache_key
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)  # or pickle.load
    return None


def save_model_results_to_cache(cache_key, weights_dict):
    cache_file = Path("cache") / cache_key
    with open(cache_file, "w") as f:
        json.dump(weights_dict, f, indent=4)


def run_optimization_and_save(
    df, config: Config, start_date, end_date, symbols, stack, years
):
    for optimization in config.models[years]:
        method = optimization.lower()
        cache_key = make_cache_key(method, years, symbols, config_hash="123456")  # etc.

        # 1) Check cache
        cached = load_model_results_from_cache(cache_key)
        if cached is not None:
            print(f"Using cached results for {method.upper()} with {years} years.")
            normalized_weights = normalize_weights(cached, config.min_weight)
            stack[method + str(years)] = normalized_weights
        else:
            # 2) Not in cache => run optimization
            optimizer = run_optimization(method, df, config)
            # Convert optimizer.weights to a dictionary and normalize
            converted_weights = convert_to_dict(
                optimizer.weights, asset_names=df.columns
            )
            normalized_weights = normalize_weights(converted_weights, config.min_weight)

            # 3) Save new result to cache
            save_model_results_to_cache(cache_key, normalized_weights)

            # 4) Update your stack
            stack[method + str(years)] = normalized_weights

        # Output / Print / Plot
        output_results(
            df, normalized_weights, method, config, start_date, end_date, years
        )
