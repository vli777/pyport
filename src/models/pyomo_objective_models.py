# File: objectives.py
import pyomo.environ as pyo
import numpy as np
import pandas as pd


def build_sharpe_model(cov: pd.DataFrame, mu: np.ndarray, target_sum: float, bounds):
    """
    Build a Pyomo model to maximize the Sharpe ratio
    (implemented by minimizing the negative Sharpe).
    """
    model = pyo.ConcreteModel()
    n = len(mu)
    assets = list(range(n))
    model.assets = pyo.Set(initialize=assets)

    # Decision variables: portfolio weights
    def weight_bounds(model, i):
        return bounds[i]

    model.w = pyo.Var(model.assets, domain=pyo.Reals, bounds=weight_bounds)

    # Constraint: weights sum to target_sum.
    model.weight_sum = pyo.Constraint(
        expr=sum(model.w[i] for i in model.assets) == target_sum
    )

    # Define expressions for portfolio return and variance.
    model.port_return = pyo.Expression(
        expr=sum(
            model.w[i] * mu.iloc[i] for i in model.assets
        )  # Use iloc[i] instead of mu[i]
    )
    model.port_variance = pyo.Expression(
        expr=sum(
            model.w[i] * cov.iloc[i, j] * model.w[j]
            for i in model.assets
            for j in model.assets
        )
    )

    model.port_vol = pyo.Expression(expr=pyo.sqrt(model.port_variance))

    # Sharpe ratio: maximize (port_return/port_vol). Here we minimize its negative.
    model.obj = pyo.Objective(
        expr=-model.port_return / model.port_vol, sense=pyo.minimize
    )
    return model


def build_omega_model(
    returns: pd.DataFrame,
    target: float,
    target_sum: float,
    max_weight: float,
    allow_short: bool,
):
    """
    Build a Pyomo model for the Omega ratio using the linear-fractional formulation.

    The formulation uses additional variables y, z, and q:

       max_{y,q,z}  y^T E(r) - target * z
       s.t.
         y^T E(r) >= target * z,
         sum(y) = z,
         sum(q) = 1,
         q[j] >= target * z - y^T r_j  for each observation j,
         (if no shorting) y >= 0,
         y <= max_weight * z,
         z >= 0.

    After solving, the portfolio weights are recovered as w = y / z.
    """
    model = pyo.ConcreteModel()
    T, n = returns.shape
    assets = list(range(n))
    obs = list(range(T))
    model.assets = pyo.Set(initialize=assets)
    model.obs = pyo.Set(initialize=obs)

    # For a robust estimate of expected returns, here we simply use the sample mean.
    # (You could substitute a trimmed mean if desired.)
    mu_robust = returns.mean(axis=0).values

    # Decision variables:
    model.y = pyo.Var(model.assets, domain=pyo.Reals)
    model.z = pyo.Var(domain=pyo.NonNegativeReals)
    model.q = pyo.Var(model.obs, domain=pyo.NonNegativeReals)

    # Constraint: y^T mu_robust >= target * z.
    model.exp_return_constraint = pyo.Constraint(
        expr=sum(model.y[i] * mu_robust[i] for i in model.assets) >= target * model.z
    )

    # Scaling constraint: sum(y) == z.
    model.scaling_constraint = pyo.Constraint(
        expr=sum(model.y[i] for i in model.assets) == model.z
    )

    # Normalize auxiliary variables: sum(q) == 1.
    model.q_norm = pyo.Constraint(expr=sum(model.q[j] for j in model.obs) == 1)

    # For each historical observation, enforce:
    #   q[j] >= target*z - sum(y[i]*r[j, i])
    def obs_constraint_rule(model, j):
        return model.q[j] >= target * model.z - sum(
            model.y[i] * returns.iloc[j, i] for i in model.assets
        )

    model.obs_constraints = pyo.Constraint(model.obs, rule=obs_constraint_rule)

    # Bound the y variables.
    def upper_bound_rule(model, i):
        return model.y[i] <= max_weight * model.z

    model.upper_bound = pyo.Constraint(model.assets, rule=upper_bound_rule)

    if not allow_short:

        def nonnegative_rule(model, i):
            return model.y[i] >= 0

        model.nonnegative = pyo.Constraint(model.assets, rule=nonnegative_rule)

    # Note: The recovered weights will be w = y / z.
    # (You may later choose to normalize these so they sum to target_sum.)

    # Objective: maximize y^T mu_robust - target * z.
    # We express this as minimizing its negative.
    model.obj = pyo.Objective(
        expr=-(sum(model.y[i] * mu_robust[i] for i in model.assets) - target * model.z),
        sense=pyo.minimize,
    )
    return model


def build_sharpe_omega_model(
    cov: pd.DataFrame,
    mu: np.ndarray,
    returns: pd.DataFrame,
    target: float,
    target_sum: float,
    max_weight: float,
    allow_short: bool,
):
    """
    Build a combined Pyomo model that optimizes for a weighted combination
    of the Sharpe and Omega ratios.

    Variables:
      - w[i]: Portfolio weights (common to both objectives)
      - u[j]: Auxiliary variables for each historical observation j,
              representing (target - w^T r_j)_+.

    The Sharpe component is defined as:
          - (w^T mu) / sqrt(w^T cov w)

    The Omega component is defined as:
          - [ (w^T mu - target) / (average_j u_j) + 1 ]

    The overall objective minimizes:
          0.5 * (negative Sharpe) + 0.5 * (negative Omega)

    Note: This model is non-linear and non-convex.
    """
    model = pyo.ConcreteModel()

    n = len(mu)
    T = returns.shape[0]
    assets = list(range(n))
    obs = list(range(T))

    model.assets = pyo.Set(initialize=assets)
    model.obs = pyo.Set(initialize=obs)

    # Define bounds for weights.
    lower_bound = -max_weight if allow_short else 0.0

    # Decision variables: portfolio weights.
    def weight_bounds(model, i):
        return (lower_bound, max_weight)

    model.w = pyo.Var(model.assets, domain=pyo.Reals, bounds=weight_bounds)

    # Constraint: Weights must sum to target_sum.
    model.weight_sum = pyo.Constraint(
        expr=sum(model.w[i] for i in model.assets) == target_sum
    )

    # --- Sharpe Ratio Components ---
    # Portfolio return: w^T mu.
    model.port_return = pyo.Expression(
        expr=sum(model.w[i] * mu[i] for i in model.assets)
    )
    # Portfolio variance: w^T cov w.
    model.port_variance = pyo.Expression(
        expr=sum(
            model.w[i] * cov.iloc[i, j] * model.w[j]
            for i in model.assets
            for j in model.assets
        )
    )
    # Portfolio volatility.
    model.port_vol = pyo.Expression(expr=pyo.sqrt(model.port_variance))
    # Negative Sharpe (since we minimize).
    model.sharpe_obj = pyo.Expression(expr=-model.port_return / model.port_vol)

    # --- Omega Ratio Components ---
    # For each observation j, define u_j >= target - (w^T r_j)
    # (This enforces u_j to capture the shortfall; note that if (w^T r_j) > target then u_j can be 0.)
    model.u = pyo.Var(model.obs, domain=pyo.NonNegativeReals)

    def u_rule(model, j):
        return model.u[j] >= target - sum(
            model.w[i] * returns.iloc[j, i] for i in model.assets
        )

    model.u_constraint = pyo.Constraint(model.obs, rule=u_rule)

    # Omega numerator: w^T mu - target.
    model.omega_num = pyo.Expression(expr=model.port_return - target)
    # Omega denominator: average of the u_j values.
    model.omega_den = pyo.Expression(
        expr=(1.0 / T) * sum(model.u[j] for j in model.obs)
    )
    # Omega ratio: (w^T mu - target) / (average u) + 1.
    model.omega_ratio = pyo.Expression(expr=model.omega_num / model.omega_den + 1)
    # Negative Omega objective.
    model.omega_obj = pyo.Expression(expr=-model.omega_ratio)

    # --- Combined Objective ---
    # For example, we weight the Sharpe and Omega parts equally.
    model.obj = pyo.Objective(
        expr=0.5 * model.sharpe_obj + 0.5 * model.omega_obj, sense=pyo.minimize
    )

    return model
