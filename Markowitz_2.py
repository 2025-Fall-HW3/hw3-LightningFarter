"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""

class MyPortfolio:
    """
    Custom portfolio strategy for Task 4.

    Logic is intentionally unchanged from your current version:
    - Uses last 30 days of *entire* price history for technical signals
      (i.e., same behavior / same potential lookahead as original).
    - Daily rebalancing starting at lookback+1.
    - Select top 2 assets by bullish_score and concentrate weights.
    """

    def __init__(self, price: pd.DataFrame, exclude: str, lookback: int = 15, gamma: float = 1):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

        # Filled by calculate_weights / calculate_portfolio_returns
        self.portfolio_weights: pd.DataFrame | None = None
        self.portfolio_returns: pd.DataFrame | None = None

    def enhanced_mv_opt(self, R_n: pd.DataFrame) -> list[float]:
        """
        Technical analysis based momentum strategy (unchanged logic).

        Parameters
        ----------
        R_n : pd.DataFrame
            Lookback window of returns for the tradable assets.

        Returns
        -------
        list[float]
            Portfolio weights aligned with R_n.columns order.
        """
        cols = R_n.columns
        n_assets = len(cols)

        # --- Technical indicators from last 30 days of prices (original behavior) ---
        prices = self.price[cols].iloc[-30:]  # NOTE: logic unchanged
        ma_short = prices.rolling(5).mean().iloc[-1]
        ma_long = prices.rolling(20).mean().iloc[-1]
        current_price = prices.iloc[-1]

        # Trend strength (not directly used downstream, kept for readability parity)
        trend_strength = (current_price - ma_long) / ma_long  # noqa: F841

        # --- Momentum signals ---
        returns_5d = R_n.iloc[-5:].mean()
        returns_20d = R_n.iloc[-20:].mean() if len(R_n) >= 20 else R_n.mean()

        # --- Volatility-adjusted momentum ---
        volatility = R_n.std()
        risk_adjusted_momentum = returns_5d / (volatility + 1e-6)

        # --- Combine signals into bullish scores ---
        bullish_score = pd.Series(0.0, index=cols)
        for asset in cols:
            score = 0.0

            # Trend component
            if current_price[asset] > ma_short[asset] > ma_long[asset]:
                score += 2.0
            elif current_price[asset] > ma_long[asset]:
                score += 1.0

            # Momentum component
            if returns_5d[asset] > 0:
                score += 1.0
            if returns_20d[asset] > 0:
                score += 0.5

            # Risk-adjusted momentum
            score += max(0.0, risk_adjusted_momentum[asset]) * 2.0

            bullish_score[asset] = score

        # Select top 2 assets
        top_n = 2
        top_assets = bullish_score.nlargest(top_n).index.tolist()

        # --- Allocate aggressively to winners (unchanged logic) ---
        weights = pd.Series(0.0, index=cols)

        if len(top_assets) > 0 and bullish_score.loc[top_assets].sum() > 0:
            scores = bullish_score.loc[top_assets].values

            # Exponential emphasis
            max_score = scores.max()
            exp_scores = np.exp(scores / max_score) if max_score > 0 else np.ones_like(scores)
            normalized = exp_scores / exp_scores.sum()

            # Concentration constraints
            max_weight = 0.6
            min_weight = 0.15

            # Apply bounds per selected asset
            bounded = []
            for w in normalized:
                bounded.append(max(min_weight, min(max_weight, w)))

            bounded = np.array(bounded)
            total = bounded.sum()

            if total > 0:
                bounded /= total
            else:
                bounded[:] = 1.0 / len(top_assets)

            weights.loc[top_assets] = bounded
        else:
            # Fallback: equal weight across all tradable assets
            weights[:] = 1.0 / n_assets

        return weights.values.tolist()

    def calculate_weights(self) -> None:
        """
        Compute daily portfolio weights.
        Logic unchanged; only style + assignment cleanliness improved.
        """
        tradable_assets = self.price.columns[self.price.columns != self.exclude]

        self.portfolio_weights = pd.DataFrame(
            0.0, index=self.price.index, columns=self.price.columns, dtype=float
        )

        start_idx = self.lookback + 1
        for i in range(start_idx, len(self.price)):
            # Lookback return window for this day
            R_n = self.returns[tradable_assets].iloc[i - self.lookback:i]

            # Compute weights (aligned to tradable_assets order)
            w = self.enhanced_mv_opt(R_n)

            dt = self.price.index[i]

            # Assign in one shot using aligned Series
            self.portfolio_weights.loc[dt, tradable_assets] = pd.Series(w, index=tradable_assets)
            self.portfolio_weights.loc[dt, self.exclude] = 0.0

        # Fill forward between computed dates
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0.0, inplace=True)

    def calculate_portfolio_returns(self) -> None:
        """
        Compute daily portfolio returns from weights.
        """
        if self.portfolio_weights is None:
            self.calculate_weights()

        tradable_assets = self.price.columns[self.price.columns != self.exclude]

        self.portfolio_returns = self.returns.copy()
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[tradable_assets]
            .mul(self.portfolio_weights[tradable_assets])
            .sum(axis=1)
        )

    def get_results(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns
        -------
        (weights_df, returns_df)
        """
        if self.portfolio_returns is None:
            self.calculate_portfolio_returns()

        # mypy safety: these are set above
        assert self.portfolio_weights is not None
        assert self.portfolio_returns is not None
        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
