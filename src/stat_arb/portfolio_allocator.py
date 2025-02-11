import optuna
import pandas as pd

from utils.performance_metrics import sharpe_ratio


class PortfolioAllocator:
    def __init__(self, risk_target: float = 0.15, leverage_cap: float = 3.0):
        """
        Initializes the PortfolioAllocator with risk management parameters.

        Args:
            risk_target (float): Target portfolio volatility.
            leverage_cap (float): Maximum allowable leverage.
        """
        self.risk_target = risk_target
        self.leverage_cap = leverage_cap

    def compute_allocations(
        self, individual_returns: dict, multi_asset_returns: pd.Series
    ) -> pd.Series:
        """
        Computes the final portfolio allocation using Kelly + Risk Parity.

        Args:
            individual_returns (dict): Dictionary of per-ticker strategy returns.
            multi_asset_returns (pd.Series): Multi-asset reversion strategy returns.

        Returns:
            pd.Series: Final portfolio weights.
        """
        # Convert dictionary to DataFrame
        returns_df = pd.DataFrame(individual_returns)
        returns_df["multi_asset"] = multi_asset_returns

        # Compute risk parity weights
        risk_parity_weights = self.compute_risk_parity(returns_df)

        # Compute Kelly scaling
        kelly_weights = self.compute_kelly_sizing(returns_df)

        # Optimize Kelly & Risk Parity jointly using Optuna (pass returns_df)
        optimal_weights = self.optimize_kelly_risk_parity(
            kelly_weights, risk_parity_weights, returns_df
        )

        # Apply adaptive leverage
        final_allocations = self.apply_adaptive_leverage(optimal_weights, returns_df)

        return final_allocations

    def compute_risk_parity(self, returns_df: pd.DataFrame) -> pd.Series:
        """
        Computes risk parity weights based on historical volatilities.

        Args:
            returns_df (pd.DataFrame): Returns of all strategies.

        Returns:
            pd.Series: Risk parity weights.
        """
        vol = returns_df.std()
        risk_parity_weights = 1 / vol
        return risk_parity_weights / risk_parity_weights.sum()

    def compute_kelly_sizing(self, returns_df: pd.DataFrame) -> pd.Series:
        """
        Computes Kelly-optimal bet sizing for each strategy.

        Args:
            returns_df (pd.DataFrame): Returns of all strategies.

        Returns:
            pd.Series: Kelly fractions.
        """
        mean_returns = returns_df.mean()
        variance = returns_df.var()
        kelly_fractions = mean_returns / variance
        return kelly_fractions / kelly_fractions.sum()

    def optimize_kelly_risk_parity(
        self,
        kelly_weights: pd.Series,
        risk_parity_weights: pd.Series,
        returns_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Uses Optuna to jointly optimize Kelly scaling and risk parity allocation.

        Returns:
            pd.Series: Optimized final allocation weights.
        """

        def objective(trial):
            kelly_scaling = trial.suggest_float("kelly_scaling", 0.1, 1.0)
            risk_parity_scaling = trial.suggest_float("risk_parity_scaling", 0.1, 1.0)
            combined_weights = (
                kelly_scaling * kelly_weights
                + risk_parity_scaling * risk_parity_weights
            )
            combined_weights /= combined_weights.sum()

            # Simulate portfolio performance
            portfolio_returns = (combined_weights * returns_df).sum(axis=1)
            return sharpe_ratio(portfolio_returns)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        best_scaling = study.best_params
        return (
            best_scaling["kelly_scaling"] * kelly_weights
            + best_scaling["risk_parity_scaling"] * risk_parity_weights
        ).div((best_scaling["kelly_scaling"] + best_scaling["risk_parity_scaling"]))

    def apply_adaptive_leverage(
        self, weights: pd.Series, returns_df: pd.DataFrame
    ) -> pd.Series:
        """
        Applies dynamic leverage based on market volatility.

        Returns:
            pd.Series: Final leverage-adjusted weights.
        """
        realized_volatility = returns_df.rolling(window=30).std().mean(axis=1).iloc[-1]
        leverage = min(self.risk_target / realized_volatility, self.leverage_cap)

        return weights * leverage
