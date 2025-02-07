import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go

from utils.performance_metrics import kappa_ratio, sharpe_ratio


class OUHeatPotentialStrategy:
    def __init__(
        self, prices: pd.Series, returns_df: pd.DataFrame, dt: float = 1.0, T: int = 10
    ):
        """
        Initialize the strategy with price data and a returns DataFrame.
        """
        self.prices = prices
        self.returns_df = returns_df
        self.dt = dt
        self.T = T
        self.kappa, self.mu, self.sigma = self.estimate_ou_parameters()
        print(
            f"Estimated parameters: kappa={self.kappa:.4f}, mu={self.mu:.4f}, sigma={self.sigma:.4f}"
        )

    def estimate_ou_parameters(self):
        """
        Estimate OU parameters using OLS regression on the log-price series.
        This replicates the idea of stacking opportunities and computing
        θ, μ, σ via covariances as in the literature.
        """
        log_prices = np.log(self.prices)
        delta_x = log_prices.diff().dropna()
        X_t = log_prices.shift(1).dropna()
        # Align Y_t with X_t's index
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
        Generate trading signals based on the deviations of log-prices from the estimated mean.
        Entry is triggered when the deviation falls below stop_loss and exit when it exceeds take_profit.
        """
        log_prices = np.log(self.prices)
        deviations = log_prices - self.mu  # deviation from the estimated long-run mean
        position = 0
        signals = pd.DataFrame(
            index=self.prices.index, columns=["Position", "Entry Price", "Exit Price"]
        )
        entry_price = None

        # Loop through prices to apply the trading rule
        for i in range(1, len(self.prices)):
            if position == 0:
                if deviations.iloc[i] < stop_loss:
                    position = 1  # Enter long
                    entry_price = self.prices.iloc[i]
                    signals.iloc[i, signals.columns.get_loc("Position")] = "BUY"
                    signals.iloc[i, signals.columns.get_loc("Entry Price")] = (
                        entry_price
                    )
            elif position == 1:
                if deviations.iloc[i] > take_profit:
                    position = 0  # Exit trade
                    signals.iloc[i, signals.columns.get_loc("Position")] = "SELL"
                    signals.iloc[i, signals.columns.get_loc("Exit Price")] = (
                        self.prices.iloc[i]
                    )
        return signals

    def simulate_strategy(self, signals):
        """
        Simulate strategy performance by extracting trades from signals and computing log returns.
        Returns both an array of trade returns and a metrics dictionary.
        """
        trades = signals.dropna(subset=["Entry Price", "Exit Price"])
        if len(trades) == 0:
            metrics = {
                "Total Trades": 0,
                "Sharpe Ratio": 0,
                "Win Rate": 0,
                "Average Return": 0,
                "Kappa Ratio": 0,
            }
            returns = np.array([])
        else:
            returns = np.log(trades["Exit Price"].values / trades["Entry Price"].values)
            returns_series = pd.Series(returns)
            win_rate = (returns > 0).mean()
            sr = sharpe_ratio(returns_series, entries_per_year=252, risk_free_rate=0.0)
            kp = kappa_ratio(pd.Series(returns), order=3)
            metrics = {
                "Total Trades": len(trades),
                "Sharpe Ratio": sr,
                "Win Rate": win_rate,
                "Average Return": returns.mean(),
                "Kappa Ratio": kp,
            }
        return returns, metrics

    def composite_score(self, metrics):
        """
        Define a composite score to assess strategy performance.
        Here we combine Sharpe ratio, win rate, logarithmic factor of total trades, and Kappa ratio.
        (In a more sophisticated implementation, this score would come directly from solving
        the heat potentials based integral equations.)
        """
        if metrics["Total Trades"] == 0:
            return 0
        # Multiply the Sharpe, win rate, log(trade count+1), and kappa ratio
        score = (
            metrics["Sharpe Ratio"]
            * metrics["Win Rate"]
            * np.log(metrics["Total Trades"] + 1)
            * metrics["Kappa Ratio"]
        )
        return score

    def compute_optimal_bounds(self):
        """
        Compute optimal stop-loss and take-profit levels by maximizing the composite score.
        Uses the provided compute_optimal_bounds function structure.
        """

        def composite_objective(bounds):
            stop_loss, take_profit = bounds
            positions_df = self.generate_trading_signals(stop_loss, take_profit)
            _, metrics = self.simulate_strategy(positions_df)
            score = self.composite_score(metrics)
            return -score  # we minimize the negative score

        # Set bounds relative to the estimated volatility
        bounds = [(-2 * self.sigma, 0), (0, 2 * self.sigma)]
        opt_result = minimize(composite_objective, x0=[-1, 1], bounds=bounds)
        return opt_result.x  # returns (stop_loss, take_profit)

    def run_strategy(self):
        """
        Run the complete strategy:
            1. Optimize the trading bounds.
            2. Generate trading signals using these optimal bounds.
            3. Backtest the strategy and compute performance metrics.
        """
        stop_loss, take_profit = self.compute_optimal_bounds()
        print("Optimal stop_loss:", stop_loss, "take_profit:", take_profit)
        signals = self.generate_trading_signals(stop_loss, take_profit)
        _, metrics = self.simulate_strategy(signals)
        print("Strategy Metrics:", metrics)
        return signals, metrics


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

    # strategy = OUHeatPotentialStrategy(prices, returns_df)
    # signals, metrics = strategy.run_strategy()
    # plot_trading_signals(prices, signals)
