"""Backtesting engine: PEAD computation, trade signals, and portfolio metrics."""

from earnings_intel.backtesting.engine import (
    BacktestEngine,
    BacktestReport,
    SignalFilter,
    TradeRecord,
    TradeSignal,
)
from earnings_intel.backtesting.metrics import (
    PortfolioMetrics,
    calculate_portfolio_metrics,
    compute_calmar_ratio,
    compute_hit_rate,
    compute_max_drawdown,
    compute_sharpe_ratio,
)
from earnings_intel.backtesting.pead import (
    InsufficientPriceDataError,
    PEADResult,
    compute_pead,
    compute_pead_from_dataframes,
)

__all__ = [
    "BacktestEngine",
    "BacktestReport",
    "InsufficientPriceDataError",
    "PEADResult",
    "PortfolioMetrics",
    "SignalFilter",
    "TradeRecord",
    "TradeSignal",
    "calculate_portfolio_metrics",
    "compute_calmar_ratio",
    "compute_hit_rate",
    "compute_max_drawdown",
    "compute_pead",
    "compute_pead_from_dataframes",
    "compute_sharpe_ratio",
]
