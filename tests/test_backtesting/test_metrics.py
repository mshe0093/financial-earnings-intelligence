"""Known-answer unit tests for quantitative performance metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from earnings_intel.backtesting.metrics import (
    PortfolioMetrics,
    calculate_portfolio_metrics,
    compute_calmar_ratio,
    compute_hit_rate,
    compute_max_drawdown,
    compute_sharpe_ratio,
)


class TestSharpeRatio:
    """Known-answer verification for Sharpe ratio calculation."""

    def test_constant_returns_returns_zero(self) -> None:
        """Standard deviation is zero, so Sharpe ratio must be zero."""
        returns = [0.01, 0.01, 0.01, 0.01]
        assert compute_sharpe_ratio(returns) == 0.0

    def test_empty_or_single_return_returns_zero(self) -> None:
        """Fewer than 2 return observations yields 0.0."""
        assert compute_sharpe_ratio([]) == 0.0
        assert compute_sharpe_ratio([0.05]) == 0.0

    def test_known_sharpe_formula(self) -> None:
        """Verify Sharpe calculation matches manual formula."""
        returns = [0.01, 0.02, 0.03]
        s = pd.Series(returns)
        mean_r = s.mean()  # 0.02
        std_r = s.std()  # 0.01
        annualization = math.sqrt(252.0)
        expected = round((mean_r / std_r) * annualization, 4)

        result = compute_sharpe_ratio(returns, risk_free_rate=0.0, annualization_factor=252.0)
        assert result == pytest.approx(expected, abs=1e-4)

    def test_sharpe_with_risk_free_rate(self) -> None:
        """Verify deduction of risk-free rate."""
        returns = [0.05, 0.06, 0.07]
        rf = 0.02
        factor = 252.0
        period_rf = rf / factor
        s = pd.Series(returns) - period_rf
        expected = round((s.mean() / s.std()) * math.sqrt(factor), 4)

        result = compute_sharpe_ratio(returns, risk_free_rate=rf, annualization_factor=factor)
        assert result == pytest.approx(expected, abs=1e-4)

    def test_negative_mean_returns_negative_sharpe(self) -> None:
        """Negative mean return produces a negative Sharpe ratio."""
        returns = [-0.02, -0.01, -0.03]
        result = compute_sharpe_ratio(returns)
        assert result < 0.0

    def test_handles_nan_and_series(self) -> None:
        """NaN values are properly dropped."""
        s = pd.Series([0.01, np.nan, 0.02, 0.03, None])
        result = compute_sharpe_ratio(s)
        assert result > 0.0


class TestHitRate:
    """Unit tests for hit rate computation."""

    def test_mixed_returns(self) -> None:
        """2 wins out of 4 trades = 50% hit rate."""
        returns = [0.05, -0.02, 0.03, -0.01]
        assert compute_hit_rate(returns) == 0.50

    def test_all_positive_returns(self) -> None:
        """100% winning trades."""
        returns = [0.01, 0.02, 0.03]
        assert compute_hit_rate(returns) == 1.0

    def test_all_losing_returns(self) -> None:
        """0% winning trades."""
        returns = [-0.01, -0.02, 0.0]
        assert compute_hit_rate(returns) == 0.0

    def test_empty_returns(self) -> None:
        """Empty sequence returns 0.0."""
        assert compute_hit_rate([]) == 0.0


class TestMaxDrawdown:
    """Unit tests for maximum peak-to-trough drawdown."""

    def test_known_drawdown_peak_and_trough(self) -> None:
        """Peak = 1.20, trough = 0.90 -> (0.90 - 1.20) / 1.20 = -0.25."""
        equity_curve = [1.0, 1.2, 0.9, 1.1]
        result = compute_max_drawdown(equity_curve)
        assert result == pytest.approx(-0.25, abs=1e-4)

    def test_monotonic_gains_drawdown_is_zero(self) -> None:
        """Purely increasing equity has zero drawdown."""
        equity_curve = [1.0, 1.05, 1.10, 1.20]
        assert compute_max_drawdown(equity_curve) == 0.0

    def test_monotonic_losses(self) -> None:
        """Equity dropping from 1.0 to 0.60 -> -0.40."""
        equity_curve = [1.0, 0.9, 0.8, 0.6]
        result = compute_max_drawdown(equity_curve)
        assert result == pytest.approx(-0.40, abs=1e-4)

    def test_empty_or_single_point(self) -> None:
        """Empty or single point equity curve yields 0.0."""
        assert compute_max_drawdown([]) == 0.0
        assert compute_max_drawdown([1.0]) == 0.0


class TestCalmarRatio:
    """Unit tests for Calmar ratio."""

    def test_standard_calmar(self) -> None:
        """Annualized return 20% with 10% max drawdown -> Calmar = 2.0."""
        assert compute_calmar_ratio(0.20, -0.10) == 2.0

    def test_zero_drawdown_positive_return(self) -> None:
        """Zero drawdown returns the annualized return."""
        assert compute_calmar_ratio(0.15, 0.0) == 0.15

    def test_negative_return(self) -> None:
        """Negative return yields Calmar 0.0."""
        assert compute_calmar_ratio(-0.05, -0.10) == 0.0


class TestCalculatePortfolioMetrics:
    """Tests for full calculate_portfolio_metrics function."""

    def test_empty_trades(self) -> None:
        """Empty inputs should return clean zeroed metrics."""
        m = calculate_portfolio_metrics([], [])
        assert isinstance(m, PortfolioMetrics)
        assert m.total_trades == 0
        assert m.hit_rate == 0.0
        assert m.equity_curve == [1.0]

    def test_multi_trade_performance(self) -> None:
        """Sequence of 5 trades computes all fields."""
        strat_returns = [0.04, -0.02, 0.06, 0.01, -0.03]
        ab_returns = [0.03, -0.01, 0.04, 0.00, -0.02]

        m = calculate_portfolio_metrics(
            strategy_returns=strat_returns,
            abnormal_returns=ab_returns,
            risk_free_rate=0.01,
            holding_period_days=30,
        )

        assert m.total_trades == 5
        assert m.winning_trades == 3
        assert m.losing_trades == 2
        assert m.hit_rate == 0.60
        assert len(m.equity_curve) == 6
        assert m.equity_curve[0] == 1.0
        assert m.max_drawdown < 0.0
        assert isinstance(m.sharpe_ratio, float)
