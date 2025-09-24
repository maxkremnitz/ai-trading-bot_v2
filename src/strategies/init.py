"""
Trading Strategies Package
4 spezialisierte Strategien mit AI Consensus Integration
"""

from .base_strategy import BaseStrategy, StrategySignal
from .trend_following import TrendFollowingStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy
from .news_trading import NewsTrading Strategy

__all__ = [
    'BaseStrategy',
    'StrategySignal',
    'TrendFollowingStrategy',
    'MeanReversionStrategy', 
    'BreakoutStrategy',
    'NewsTradingStrategy'
]
