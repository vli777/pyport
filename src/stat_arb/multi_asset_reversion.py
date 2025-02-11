from typing import Optional
import numpy as np
import pandas as pd
import optuna
from scipy.optimize import minimize


from stat_arb.cointegration import CointegrationAnalyzer
from utils.performance_metrics import kappa_ratio, sharpe_ratio


class MultiAssetReversion:
    def __init__(
        self,
        prices_df: pd.DataFrame,
        det_order=0,
        k_ar_diff=1,
    ):
        """
        Multi-asset mean reversion strategy based on cointegration.

        Args:
            prices_df (pd.DataFrame): Price data DataFrame (regular prices) which may have different history lengths.
            det_order (int): Deterministic trend order for the Johansen test.
            k_ar_diff (int): Number of lag differences.
        """
        # Ensure all prices are positive for the logarithm.
        if (prices_df <= 0).any().any():
            raise ValueError(
                "Price data must be strictly positive to compute logarithms."
            )

        # Align to the overlapping period where all assets have data.
        common_index = prices_df.dropna(axis=0, how="any").index
        if len(common_index) == 0:
            raise ValueError("No overlapping dates among asset histories.")
        prices_df = prices_df.loc[common_index]

        # Convert to log prices.
        self.prices_df = np.log(prices_df)
        # Compute log returns.
        self.returns_df = self.prices_df.diff().dropna()

        # Initialize cointegration analyzer.
        self.cointegration_analyzer = CointegrationAnalyzer(
            self.prices_df, det_order, k_ar_diff
        )
        # Ensure the spread series has no NaN by forward/backward filling.
        self.spread_series = self.cointegration_analyzer.spread_series.fillna(
            method="ffill"
        ).fillna(method="bfill")
        self.hedge_ratios = self.cointegration_analyzer.get_hedge_ratios()

        # Compute Kelly and Risk Parity weights.
        self.kelly_fractions = self.compute_dynamic_kelly()
        self.risk_parity_weights = self.compute_risk_parity_weights()

        # Optimize allocations.
        self.optimal_params = self.optimize_kelly_risk_parity()

    def compute_dynamic_kelly(self, risk_free_rate=0.0):
        """
        Computes dynamic Kelly fractions for each asset.
        """
        kelly_allocations = {}
        for asset in self.returns_df.columns:
            mean_return = self.returns_df[asset].mean() - risk_free_rate
            vol = self.returns_df[asset].std()

            # Guard against zero or NaN volatility.
            if vol == 0 or np.isnan(vol):
                kelly_allocations[asset] = 0.0
            else:
                raw_kelly = mean_return / (vol**2)
                # Use a rolling window with a minimum number of periods.
                rolling_vol = (
                    self.returns_df[asset]
                    .rolling(window=30, min_periods=5)
                    .std()
                    .dropna()
                )
                market_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else vol
                adaptive_kelly = raw_kelly / (1 + market_vol)
                kelly_allocations[asset] = max(0, min(adaptive_kelly, 1))
        return pd.Series(kelly_allocations)

    def compute_risk_parity_weights(self):
        """
        Computes Risk Parity allocations based on inverse volatility.
        """
        volatilities = self.returns_df.std()
        # Replace zeros to avoid division by zero.
        volatilities = volatilities.replace(0, 1e-6)
        inv_vol_weights = 1 / volatilities
        total_weight = inv_vol_weights.sum()
        if total_weight == 0 or np.isnan(total_weight):
            # Fallback to equal weights.
            return pd.Series(
                np.repeat(1 / len(inv_vol_weights), len(inv_vol_weights)),
                index=inv_vol_weights.index,
            )
        return inv_vol_weights / total_weight

    def optimize_kelly_risk_parity(self):
        """
        Jointly optimizes Kelly sizing & Risk Parity for max Sharpe/Kappa ratio.
        """

        def objective(trial):
            kelly_scaling = trial.suggest_float("kelly_scaling", 0.1, 1.0)
            risk_parity_scaling = trial.suggest_float("risk_parity_scaling", 0.1, 1.0)

            kelly_allocations = self.kelly_fractions * kelly_scaling
            risk_parity_allocations = self.risk_parity_weights * risk_parity_scaling

            final_allocations = kelly_allocations + risk_parity_allocations
            total_alloc = final_allocations.sum()
            if total_alloc == 0 or np.isnan(total_alloc):
                return -np.inf  # Avoid division by zero.
            final_allocations /= total_alloc

            portfolio_returns = (self.returns_df * final_allocations).sum(axis=1)
            sharpe = sharpe_ratio(portfolio_returns)
            kappa = kappa_ratio(portfolio_returns, order=3)
            return 0.5 * sharpe + 0.5 * kappa

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)
        return study.best_params

    def generate_trading_signals(
        self, stop_loss: Optional[float] = None, take_profit: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generates buy/sell signals for the mean-reverting basket.

        Returns:
            pd.DataFrame: Signals DataFrame with "Ticker", "Position", "Entry Price", and "Exit Price".
        """
        if stop_loss is None or take_profit is None:
            stop_loss, take_profit = self.calculate_optimal_bounds()

        deviations = self.spread_series - self.spread_series.mean()
        long_positions = deviations < stop_loss
        short_positions = deviations > take_profit

        signals = pd.DataFrame(index=self.spread_series.index)
        signals["Position"] = np.where(
            long_positions, 1, np.where(short_positions, -1, 0)
        )
        signals["Ticker"] = ", ".join(self.prices_df.columns)
        signals["Entry Price"] = np.where(
            signals["Position"] == 1, self.spread_series, np.nan
        )
        signals["Exit Price"] = np.where(
            signals["Position"] == -1, self.spread_series, np.nan
        )
        return signals

    def simulate_strategy(self, signals):
        """
        Backtests the strategy and computes key performance metrics.
        """
        returns = signals["Position"].shift(1) * self.spread_series.pct_change()
        returns = returns.dropna()

        sharpe = sharpe_ratio(returns)
        kappa = kappa_ratio(returns, order=3)

        metrics = {
            "Total Trades": (signals["Position"] != 0).sum(),
            "Sharpe Ratio": sharpe,
            "Kappa Ratio": kappa,
            "Win Rate": (returns > 0).mean(),
            "Avg Return": returns.mean(),
        }
        return returns, metrics

    def calculate_optimal_bounds(self):
        """
        Computes optimal stop-loss and take-profit based on maximizing mean reversion.
        """

        def objective(bounds):
            stop_loss, take_profit = bounds
            signals = self.generate_trading_signals(stop_loss, take_profit)
            _, metrics = self.simulate_strategy(signals)
            sharpe = metrics["Sharpe Ratio"]
            if np.isnan(sharpe):
                return np.inf
            return -sharpe

        std_spread = self.spread_series.std()
        if std_spread == 0 or np.isnan(std_spread):
            std_spread = 1e-6
        bounds = [(-2 * std_spread, 0), (0, 2 * std_spread)]
        result = minimize(objective, x0=[-1, 1], bounds=bounds)
        return result.x

    def optimize_and_trade(self):
        """
        Full pipeline:
          1. Optimize entry/exit bounds.
          2. Generate trading signals.
          3. Simulate strategy & return results.
          4. Return hedge ratios along with signals and metrics.
        """
        stop_loss, take_profit = self.calculate_optimal_bounds()
        signals = self.generate_trading_signals(stop_loss, take_profit)
        returns, metrics = self.simulate_strategy(signals)

        # Ensure hedge ratios are a complete Series without NaN.
        hedge_ratios_series = pd.Series(
            self.hedge_ratios, index=self.prices_df.columns
        ).fillna(0)

        return {
            "Optimal Stop-Loss": stop_loss,
            "Optimal Take-Profit": take_profit,
            "Metrics": metrics,
            "Signals": signals,
            "Hedge Ratios": hedge_ratios_series.to_dict(),
        }
