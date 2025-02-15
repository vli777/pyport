from typing import Any, Callable, Dict, List
import numpy as np
import pandas as pd
import optuna

from config import Config
from models.optimize_portfolio import (
    optimize_weights_objective,
    estimated_portfolio_volatility,
)
from utils.performance_metrics import conditional_var
from utils.portfolio_utils import convert_weights_to_series
from utils import logger


def adaptive_risk_constraints(
    config: Config, risk_estimates: dict, initial_weights: np.ndarray
):
    def objective(trial):
        """
        Optuna objective for tuning constraint relaxations.

        It adjusts the constraints based on a suggested relaxation factor and then
        evaluates the resulting portfolio's volatility and CVaR. The loss is defined
        as the sum of squared violations of the original risk targets.

        Returns a scalar loss that is 0 if both risk constraints are met.
        """
        # Suggest a relaxation factor (in steps of 0.1)
        relaxation_factor = trial.suggest_float("relax_factor", 0.8, 1.3, step=0.1)

        # Adjust constraints based on risk_priority
        if config.portfolio_risk_priority == "cvar":
            vol_limit_adj = (
                config.portfolio_max_vol * relaxation_factor
            )  # Relax volatility constraint
            cvar_limit_adj = config.portfolio_max_cvar  # Keep CVaR fixed
        elif config.portfolio_risk_priority == "vol":
            vol_limit_adj = config.portfolio_max_vol  # Keep volatility fixed
            cvar_limit_adj = (
                config.portfolio_max_cvar * relaxation_factor
            )  # Relax CVaR constraint
        else:  # "both"
            vol_limit_adj = config.portfolio_max_vol * relaxation_factor
            cvar_limit_adj = config.portfolio_max_cvar * relaxation_factor

        try:
            final_w = optimize_weights_objective(
                cov=risk_estimates["cov"],
                mu=risk_estimates["mu"],
                returns=risk_estimates["returns"],
                objective=config.optimization_objective,
                order=3,
                target=0.0,
                max_weight=config.max_weight,
                allow_short=config.allow_short,
                target_sum=1.0,
                vol_limit=vol_limit_adj,  # Use adjusted constraints
                cvar_limit=cvar_limit_adj,
                alpha=0.05,
                solver_method="SLSQP",
                initial_guess=initial_weights,
                apply_constraints=True,
            )

            # Evaluate portfolio volatility using the risk estimates' covariance matrix.
            port_vol = estimated_portfolio_volatility(final_w, risk_estimates["cov"])

            # Compute CVaR using the historical returns and given alpha.
            computed_cvar = conditional_var(
                pd.Series(risk_estimates["returns"] @ final_w), 0.05
            )

            # For volatility: if port_vol exceeds the original target, record the violation.
            vol_loss = max(0, port_vol - config.portfolio_max_vol)
            # For CVaR: since target CVaR is negative, a computed CVaR that is higher (less negative)
            # than the target indicates a violation.
            cvar_loss = max(0, computed_cvar - config.portfolio_max_cvar)

            # Loss is the sum of squared violations (can weight these terms if desired)
            loss = vol_loss**2 + cvar_loss**2
            return loss

        except ValueError:
            # Infeasible solution: penalize heavily
            return float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)  # You can adjust the number of trials

    best_relax_factor = study.best_params["relax_factor"]
    # logger.info(f"Optimal relaxation factor found: {best_relax_factor:.2f}")

    # Adjust the constraints with the best relaxation factor based on risk_priority.
    if config.portfolio_risk_priority == "vol":
        final_vol_limit = config.portfolio_max_vol * best_relax_factor
        final_cvar_limit = config.portfolio_max_cvar
    elif config.portfolio_risk_priority == "cvar":
        final_vol_limit = config.portfolio_max_vol
        final_cvar_limit = config.portfolio_max_cvar * best_relax_factor
    else:  # "both"
        final_vol_limit = config.portfolio_max_vol * best_relax_factor
        final_cvar_limit = config.portfolio_max_cvar * best_relax_factor

    try:
        final_w = optimize_weights_objective(
            cov=risk_estimates["cov"],
            mu=risk_estimates["mu"],
            returns=risk_estimates["returns"],
            objective=config.optimization_objective,
            order=3,
            target=0.0,
            max_weight=config.max_weight,
            allow_short=config.allow_short,
            target_sum=1.0,
            vol_limit=final_vol_limit,
            cvar_limit=final_cvar_limit,
            alpha=0.05,
            solver_method="SLSQP",
            initial_guess=initial_weights,
            apply_constraints=True,
        )
        return final_w
    except ValueError:
        logger.error("Optimization failed even after relaxing constraints.")
        return None


def apply_risk_constraints(
    initial_weights: pd.Series, risk_estimates: dict, config: Config
) -> pd.Series:
    """
    Given a merged (unconstrained) set of weights, re-optimize using risk constraints.
    Uses Optuna to adaptively adjust constraints based on `config.risk_priority`.

    Args:
        initial_weights (pd.Series): Initial portfolio weights.
        risk_estimates (dict): Dictionary of risk measures (`cov`, `mu`, `returns`).
        config (Config): Configuration object.

    Returns:
        pd.Series: Optimized weights.
    """
    cov_assets = risk_estimates["cov"].index
    initial_weights = initial_weights.reindex(cov_assets, fill_value=0)
    initial_weights_np = initial_weights.values

    logger.info(
        f"Applying risk constraints: vol_limit={config.portfolio_max_vol}, cvar_limit={config.portfolio_max_cvar}"
    )

    final_w = adaptive_risk_constraints(config, risk_estimates, initial_weights_np)

    if final_w is None:
        logger.warning("Final optimization failed. Returning initial weights.")
        return initial_weights

    return convert_weights_to_series(final_w, index=risk_estimates["cov"].index)
