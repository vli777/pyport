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
from utils.portfolio_utils import convert_to_dict, normalize_weights

def run_optimization(method, df, config):
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
    if optimizer_class:
        if method == "olmar":
            optimizer = optimizer_class(
                reversion_method=config[method]["method"],
                epsilon=config[method]["epsilon"],
                window=config[method]["window"],
                alpha=config[method]["alpha"],
            )
        elif method == "rmr":
            optimizer = optimizer_class(
                epsilon=config[method]["epsilon"],
                n_iteration=config[method]["n_iteration"],
                tau=config[method]["tau"],
                window=config[method]["window"],
            )
        elif method == "scorn":
            optimizer = optimizer_class(
                window=config[method]["window"],
                rho=config[method]["rho"],
            )
        elif method == "fcornk":
            optimizer = optimizer_class(
                window=config[method]["window"],
                rho=config[method]["rho"],
                lambd=config[method]["lambd"],
                k=config[method]["k"],
            )
        else:
            optimizer = optimizer_class()

        # Handle NCO with Sharpe check
        if method == "nco":
            asset_returns = np.log(df) - np.log(df.shift(1))
            asset_returns = asset_returns.iloc[1:, :]

            if config[method]["sharpe"]:
                mu_vec = np.array(asset_returns.mean())
            else:
                mu_vec = np.ones(len(df.columns))

            weights = optimizer.allocate_nco(
                asset_names=df.columns,
                cov=np.array(asset_returns.cov()),
                mu_vec=mu_vec.reshape(-1, 1),
            )
            optimizer.weights = weights

        # Handle MC optimization
        elif method in ["mc", "mc2"]:
            asset_returns = np.log(df) - np.log(df.shift(1))
            asset_returns = asset_returns.iloc[1:, :]
            mu_vec = np.array(asset_returns.mean())

            w_cvo, w_nco = optimizer.allocate_mcos(
                mu_vec=mu_vec.reshape(-1, 1),
                cov=np.array(asset_returns.cov()),
                num_obs=config[method]["num_obs"],
                num_sims=config[method]["num_sims"],
                kde_bwidth=config[method]["kde_bandwidth"],
                min_var_portf=not config[method]["sharpe"],
                lw_shrinkage=config[method]["lw_shrinkage"],
            )
            w_nco = w_nco.mean(axis=0)
            optimizer.weights = dict(zip(df.columns, w_nco))

        # Handle Critical Line Algorithm (CLA)
        elif method in ["cla", "cla2"]:
            solution = config[method]["solution"]
            optimizer.allocate(asset_prices=df, solution=solution)

        # Handle Mean-Variance Optimization
        elif method == "mean_variance":
            expected_returns = ReturnsEstimators().calculate_mean_historical_returns(
                asset_prices=df
            )
            covariance = ReturnsEstimators().calculate_returns(asset_prices=df).cov()

            optimizer.allocate(
                asset_names=df.columns,
                asset_prices=df,
                expected_asset_returns=expected_returns,
                covariance_matrix=covariance,
                solution=method,
                target_return=config["efficient_risk"],
                target_risk=config["efficient_return"],
                risk_aversion=config["risk_aversion"],
            )
            optimizer.get_portfolio_metrics()

        elif method in ["fcornk", "scorn", "rmr", "olmar"]:
            # Safely get the 'resample' parameter, defaulting to None if not present
            resample_by = config[method].get("resample", None)

            # Call allocate with or without resample based on the configuration
            if resample_by:
                optimizer.allocate(asset_prices=df, resample_by=resample_by)
            else:
                optimizer.allocate(asset_prices=df)
        # For all other optimizations
        else:
            optimizer.allocate(asset_prices=df, **config[method])

        return optimizer
    else:
        raise ValueError(f"Unknown optimization method: {method}")


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


def run_optimization_and_save(df, config, start_date, end_date, symbols, stack, years):
    for optimization in config.models[years]:
        method = optimization.lower()
        cache_key = make_cache_key(method, years, symbols, config_hash="123456")  # etc.

        # 1) Check cache
        cached = load_model_results_from_cache(cache_key)
        if cached is not None:
            print(f"Using cached results for {method.upper()} with {years} years.")
            normalized_weights = normalize_weights(cached, config["min_weight"])
            stack[method + str(years)] = normalized_weights
        else:
            # 2) Not in cache => run optimization
            optimizer = run_optimization(method, df, config["optimization_config"])
            # Convert optimizer.weights to a dictionary and normalize
            converted_weights = convert_to_dict(optimizer.weights, asset_names=df.columns)
            normalized_weights = normalize_weights(converted_weights, config["min_weight"])

            # 3) Save new result to cache
            save_model_results_to_cache(cache_key, normalized_weights)

            # 4) Update your stack
            stack[method + str(years)] = normalized_weights

        # Output / Print / Plot
        output_results(df, normalized_weights, method, config, start_date, end_date, years)
