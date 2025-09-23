"""
Trading AI System - Main Package
Multi-Strategy Trading mit AI Consensus Voting
"""

__version__ = "1.0.0"
__author__ = "Trading AI Team"
__description__ = "KI-basiertes Trading System mit 4 Strategien"

from .api.capital_api import CapitalComAPI
from .api.rate_limiter import RateLimiter
from .ai.voting_system import AIVotingSystem
from .strategies.base_strategy import BaseStrategy
from .utils.database import DatabaseManager
from .utils.helpers import safe_execute, safe_float

__all__ = [
    'CapitalComAPI',
    'RateLimiter', 
    'AIVotingSystem',
    'BaseStrategy',
    'DatabaseManager',
    'safe_execute',
    'safe_float'
]
