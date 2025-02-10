import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go

from utils.performance_metrics import kappa_ratio, sharpe_ratio


import numpy as np
import pandas as pd
from scipy.optimize import minimize


class OUHeatPotential:
    def __init__(
        self,
        prices: pd.Series,
        returns_df: pd.DataFrame,
        dt: float = 1.0,
        T: int = 10,
        max_leverage: float = 1.0,
    ):
        """
        Initialize the strategy with price data and returns DataFrame.

        Args:
            prices (pd.Series): Time series of stock prices.
            returns_df (pd.DataFrame): DataFrame of asset returns.
            dt (float): Time step (default 1 day).
            T (int): Lookback period for OU estimation.
            max_leverage (float): Maximum allowed leverage for position sizing.
        """
        self.prices = prices
        self.returns_df = returns_df
        self.dt = dt
        self.T = T
        self.max_leverage = max_leverage
        self.kappa, self.mu, self.sigma = self.estimate_ou_parameters()
        print(
            f"Estimated parameters: kappa={self.kappa:.4f}, mu={self.mu:.4f}, sigma={self.sigma:.4f}"
        )

    def estimate_ou_parameters(self):
        """
        Estimate Ornstein-Uhlenbeck (OU) process parameters using OLS regression on log prices.

        Returns:
            tuple: Estimated (kappa, mu, sigma).
        """
        log_prices = np.log(self.prices)
        delta_x = log_prices.diff().dropna()
        X_t = log_prices.shift(1).dropna()

        # Align response variable Y_t with X_t
        Y_t = delta_x.loc[X_t.index]
        beta, alpha = np.polyfit(X_t, Y_t, 1)

        kappa = -beta / self.dt
        mu = alpha / kappa if kappa != 0 else 0
        residuals = Y_t - (alpha + beta * X_t)
        sigma = np.std(residuals) * np.sqrt(
            2 * kappa / (1 - np.exp(-2 * kappa * self.dt))
        )

        return kappa, mu, sigma

    def generate_trading_signals(self, stop_loss: float, take_profit: float):
        """
        Vectorized trading signal generation using NumPy and Pandas indexing.

        Args:
            stop_loss (float): Lower deviation threshold to enter a long position.
            take_profit (float): Upper deviation threshold to exit the position.

        Returns:
            pd.DataFrame: Signals with "Position", "Entry Price", and "Exit Price".
        """
        log_prices = np.log(self.prices)
        deviations = log_prices - self.mu  # Compute deviations from mean

        signals = pd.DataFrame(
            index=self.prices.index, columns=["Position", "Entry Price", "Exit Price"]
        )
        signals[:] = np.nan  # Initialize with NaN

        # Entry (BUY signal) when deviation < stop_loss
        buy_mask = (deviations < stop_loss) & (signals["Position"].isna())
        signals.loc[buy_mask, "Position"] = "BUY"
        signals.loc[buy_mask, "Entry Price"] = self.prices[buy_mask]

        # Exit (SELL signal) when deviation > take_profit
        sell_mask = (deviations > take_profit) & (signals["Position"].shift(1) == "BUY")
        signals.loc[sell_mask, "Position"] = "SELL"
        signals.loc[sell_mask, "Exit Price"] = self.prices[sell_mask]

        return signals

    def simulate_strategy(self, signals):
        """Simulate strategy with Optimized Kelly + Risk Parity + Adaptive Leverage."""
        trades = signals.dropna(subset=["Entry Price", "Exit Price"])
        if trades.empty:
            return np.array([]), {
                "Total Trades": 0,
                "Sharpe Ratio": np.nan,
                "Win Rate": np.nan,
                "Kappa Ratio": np.nan,
                "Optimized Kelly Fraction": 0,
                "Risk Parity Allocation": {},
            }

        returns = np.log(trades["Exit Price"].values / trades["Entry Price"].values)
        returns_series = pd.Series(returns)

        # Get optimized Kelly & Risk Parity parameters
        best_params = self.compute_optimal_kelly_risk_parity(n_trials=50)
        kelly_fraction = best_params["kelly_fraction"]
        risk_parity_scaling = best_params["risk_parity_scaling"]

        # Compute Dynamic Kelly Fraction with Market Conditions
        rolling_volatility = (
            self.returns_df.rolling(window=30, min_periods=10).std().iloc[-1]
        )
        dynamic_kelly_fraction = kelly_fraction / (1 + rolling_volatility.mean())

        # Compute Risk Parity Allocation
        rolling_vols = self.returns_df.rolling(window=30, min_periods=10).std().iloc[-1]
        inv_vol_weights = 1 / rolling_vols
        risk_parity_allocation = inv_vol_weights / inv_vol_weights.sum()
        final_allocation = (
            dynamic_kelly_fraction * risk_parity_scaling * risk_parity_allocation
        )

        win_rate = (returns > 0).mean()
        sharpe_r = sharpe_ratio(
            returns_series, entries_per_year=252, risk_free_rate=0.0
        )
        kappa_r = kappa_ratio(returns_series, order=3)

        metrics = {
            "Total Trades": len(trades),
            "Sharpe Ratio": sharpe_r,
            "Win Rate": win_rate,
            "Kappa Ratio": kappa_r,
            "Optimized Kelly Fraction": dynamic_kelly_fraction,
            "Risk Parity Allocation": final_allocation.to_dict(),
        }

        return returns, metrics

    def composite_score(self, metrics):
        """
        Compute a composite score to evaluate strategy performance.

        Args:
            metrics (dict): Performance metrics dictionary.

        Returns:
            float: Composite performance score.
        """
        if metrics["Total Trades"] == 0:
            return 0
        return (
            metrics["Sharpe Ratio"]
            * metrics["Win Rate"]
            * np.log(metrics["Total Trades"] + 1)
            * metrics["Kappa Ratio"]
        )

    def compute_optimal_bounds(self):
        """
        Optimize stop-loss and take-profit levels to maximize composite score.

        Returns:
            tuple: (optimal_stop_loss, optimal_take_profit)
        """

        def objective(bounds):
            stop_loss, take_profit = bounds
            positions_df = self.generate_trading_signals(stop_loss, take_profit)
            _, metrics = self.simulate_strategy(positions_df)
            return -self.composite_score(metrics)  # Minimize negative score

        # Search bounds for stop-loss and take-profit
        bounds = [(-2 * self.sigma, 0), (0, 2 * self.sigma)]
        opt_result = minimize(objective, x0=[-1, 1], bounds=bounds)

        return opt_result.x  # Optimal (stop_loss, take_profit)

    def run_strategy(self):
        """
        Run the complete strategy:
            1. Optimize trading bounds.
            2. Generate trading signals.
            3. Backtest strategy and compute performance metrics.

        Returns:
            tuple: (signals DataFrame, metrics dictionary)
        """
        stop_loss, take_profit = self.compute_optimal_bounds()
        print(f"Optimal stop_loss: {stop_loss:.4f}, take_profit: {take_profit:.4f}")

        signals = self.generate_trading_signals(stop_loss, take_profit)
        _, metrics = self.simulate_strategy(signals)

        print("Strategy Metrics:", metrics)
        return signals, metrics

    def compute_optimal_kelly_risk_parity(self, n_trials: int = 50):
        """
        Optimize Kelly Fraction and Risk Parity Allocation jointly using Optuna.
        Also adjusts leverage dynamically based on market volatility.

        Args:
            n_trials (int): Number of Optuna trials.

        Returns:
            dict: Optimized parameters.
        """

        def objective(trial):
            # Suggest Kelly fraction and Risk Parity scaling factor
            kelly_fraction = trial.suggest_float("kelly_fraction", 0.01, 1.0, log=True)
            risk_parity_scaling = trial.suggest_float("risk_parity_scaling", 0.1, 2.0)

            # Apply Kelly Fraction dynamically based on market volatility
            rolling_volatility = (
                self.returns_df.rolling(window=30, min_periods=10).std().iloc[-1]
            )
            dynamic_kelly_fraction = kelly_fraction / (1 + rolling_volatility.mean())

            # Compute Risk Parity Weights
            rolling_vols = (
                self.returns_df.rolling(window=30, min_periods=10).std().iloc[-1]
            )
            inv_vol_weights = 1 / rolling_vols
            risk_parity_allocation = inv_vol_weights / inv_vol_weights.sum()
            adjusted_allocation = (
                dynamic_kelly_fraction * risk_parity_scaling * risk_parity_allocation
            )

            # Generate trading signals and simulate strategy performance
            stop_loss, take_profit = self.compute_optimal_bounds()
            signals = self.generate_trading_signals(stop_loss, take_profit)
            returns, metrics = self.simulate_strategy(signals)

            # Use Composite Score (Sharpe, Kappa, Win Rate) as Objective
            return self.composite_score(metrics)

        # Optimize using Optuna
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)

        best_params = study.best_params
        print(f"Optimized Kelly & Risk Parity: {best_params}")
        return best_params


def plot_ou_signals(
    prices: pd.Series, signals: pd.DataFrame, title: str = "OU Process Trading Signals"
):
    """
    Create an interactive Plotly chart that overlays buy/sell signals on the price series.

    Args:
        prices (pd.Series): Time series of prices.
        signals (pd.DataFrame): DataFrame with trading signals; expected to have a "Position" column
                                with entries "BUY" and "SELL", along with "Entry Price" and "Exit Price".
        title (str): Title of the plot.
    """
    # Create the figure
    fig = go.Figure()

    # Add the price series as a line trace
    fig.add_trace(
        go.Scatter(x=prices.index, y=prices.values, mode="lines", name="Price")
    )

    # Extract buy and sell signals
    buy_signals = signals[signals["Position"] == "BUY"]
    sell_signals = signals[signals["Position"] == "SELL"]

    # Add buy signal markers (using an upward triangle)
    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=prices.loc[buy_signals.index],
            mode="markers",
            name="Buy",
            marker=dict(symbol="triangle-up", color="green", size=10),
        )
    )

    # Add sell signal markers (using a downward triangle)
    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=prices.loc[sell_signals.index],
            mode="markers",
            name="Sell",
            marker=dict(symbol="triangle-down", color="red", size=10),
        )
    )

    # Update layout for better readability
    fig.update_layout(
        title=title, xaxis_title="Time", yaxis_title="Price", template="plotly_white"
    )

    fig.show()

    # strategy = OUHeatPotential(prices, returns_df)
    # signals, metrics = strategy.run_strategy()
    # plot_trading_signals(prices, signals)
