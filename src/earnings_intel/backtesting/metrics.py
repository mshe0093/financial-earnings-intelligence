"""Quantitative performance metrics: Sharpe ratio, hit rate, drawdown, and portfolio stats."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PortfolioMetrics(BaseModel):
    """Consolidated performance metrics for a backtested quantitative strategy."""

    total_trades: int = Field(description="Total number of trade signals evaluated")
    winning_trades: int = Field(description="Number of trades with positive return")
    losing_trades: int = Field(description="Number of trades with non-positive return")
    hit_rate: float = Field(description="Fraction of profitable trades (0.0 to 1.0)")
    mean_return: float = Field(description="Average raw strategy return per trade")
    mean_abnormal_return: float = Field(description="Average benchmark-relative abnormal return")
    annualized_return: float = Field(description="Annualized compound/geometric return")
    annualized_volatility: float = Field(description="Annualized return standard deviation")
    sharpe_ratio: float = Field(description="Annualized Sharpe ratio")
    max_drawdown: float = Field(description="Maximum peak-to-trough decline (negative or zero)")
    calmar_ratio: float = Field(description="Annualized return divided by absolute max drawdown")
    equity_curve: list[float] = Field(
        default_factory=list,
        description="Sequential portfolio value progression starting from 1.0",
    )


def compute_sharpe_ratio(
    returns: pd.Series | Sequence[float],
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0,
) -> float:
    """Compute annualized Sharpe ratio of a return series.

    Sharpe = (mean(R - Rf) / std(R - Rf)) * sqrt(annualization_factor)

    Args:
        returns: Series or sequence of fractional period returns.
        risk_free_rate: Annualized risk-free interest rate (default: 0.0).
        annualization_factor: Number of periods in a trading year (default: 252 for daily).

    Returns:
        Annualized Sharpe ratio float. Returns 0.0 if standard deviation is zero.
    """
    s = pd.Series(returns).dropna()
    if len(s) < 2:
        return 0.0

    period_rf = risk_free_rate / annualization_factor
    excess_returns = s - period_rf
    std = excess_returns.std()

    if std == 0 or np.isnan(std):
        return 0.0

    sharpe = (excess_returns.mean() / std) * np.sqrt(annualization_factor)
    return round(float(sharpe), 4)


def compute_hit_rate(returns: pd.Series | Sequence[float]) -> float:
    """Compute proportion of trades with positive return (R > 0).

    Args:
        returns: Sequence of trade returns.

    Returns:
        Fraction between 0.0 and 1.0. Returns 0.0 if empty.
    """
    s = pd.Series(returns).dropna()
    if len(s) == 0:
        return 0.0
    wins = (s > 0).sum()
    return round(float(wins / len(s)), 4)


def compute_max_drawdown(equity_curve: Sequence[float]) -> float:
    """Compute maximum peak-to-trough decline from an equity curve.

    Args:
        equity_curve: Sequential portfolio values (e.g. starting at 1.0).

    Returns:
        Maximum drawdown as a non-positive float (e.g. -0.15 for a 15% drop).
        Returns 0.0 if empty or monotonic gains.
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0

    eq = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(eq)
    # Avoid zero division
    valid_mask = running_max > 0
    if not np.any(valid_mask):
        return 0.0

    drawdowns = np.zeros_like(eq)
    drawdowns[valid_mask] = (eq[valid_mask] - running_max[valid_mask]) / running_max[valid_mask]
    max_dd = float(np.min(drawdowns))
    return round(max_dd, 4)


def compute_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """Compute Calmar ratio: annualized return divided by maximum drawdown magnitude.

    Calmar = annualized_return / abs(max_drawdown)

    Args:
        annualized_return: Strategy's annualized rate of return.
        max_drawdown: Maximum drawdown (negative or zero).

    Returns:
        Calmar ratio. Returns 0.0 if drawdown is zero or return is negative.
    """
    if annualized_return <= 0:
        return 0.0
    abs_dd = abs(max_drawdown)
    if abs_dd == 0:
        return round(annualized_return, 4)
    return round(float(annualized_return / abs_dd), 4)


def calculate_portfolio_metrics(
    strategy_returns: Sequence[float],
    abnormal_returns: Sequence[float],
    risk_free_rate: float = 0.0,
    holding_period_days: int = 30,
) -> PortfolioMetrics:
    """Compute full suite of portfolio performance and risk metrics.

    Args:
        strategy_returns: Sequence of executed trade percentage returns.
        abnormal_returns: Sequence of benchmark-relative abnormal returns.
        risk_free_rate: Annualized risk-free rate.
        holding_period_days: Average holding window in trading days (default: 30).

    Returns:
        Populated `PortfolioMetrics` object.
    """
    s_ret = pd.Series(strategy_returns).dropna()
    ab_ret = pd.Series(abnormal_returns).dropna()

    total_trades = len(s_ret)
    if total_trades == 0:
        return PortfolioMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            hit_rate=0.0,
            mean_return=0.0,
            mean_abnormal_return=0.0,
            annualized_return=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            equity_curve=[1.0],
        )

    winning = int((s_ret > 0).sum())
    losing = total_trades - winning
    hit_rate = round(float(winning / total_trades), 4)
    mean_ret = round(float(s_ret.mean()), 4)
    mean_ab_ret = round(float(ab_ret.mean()) if len(ab_ret) > 0 else 0.0, 4)

    # Build sequential equity curve: starting at 1.0, compounding each trade return
    equity_curve = [1.0]
    for r in s_ret:
        equity_curve.append(round(equity_curve[-1] * (1.0 + r), 6))

    # Annualize returns assuming holding_period_days per trade
    # Number of trade cycles per year
    cycles_per_year = 252.0 / max(holding_period_days, 1)
    if total_trades > 0 and equity_curve[-1] > 0:
        annualized_ret = (equity_curve[-1] ** (cycles_per_year / total_trades)) - 1.0
    else:
        annualized_ret = -1.0

    # Annualized volatility and Sharpe ratio based on trade returns scaled by periods per year
    sharpe = compute_sharpe_ratio(
        s_ret,
        risk_free_rate=risk_free_rate,
        annualization_factor=cycles_per_year,
    )
    ann_vol = float(s_ret.std() * np.sqrt(cycles_per_year)) if len(s_ret) > 1 else 0.0

    max_dd = compute_max_drawdown(equity_curve)
    calmar = compute_calmar_ratio(annualized_ret, max_dd)

    return PortfolioMetrics(
        total_trades=total_trades,
        winning_trades=winning,
        losing_trades=losing,
        hit_rate=hit_rate,
        mean_return=mean_ret,
        mean_abnormal_return=mean_ab_ret,
        annualized_return=round(float(annualized_ret), 4),
        annualized_volatility=round(ann_vol, 4),
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        equity_curve=equity_curve,
    )
