"""
Configuration Settings für Trading AI System
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class AIConfig:
    """AI Provider Configuration"""
    huggingface_token: str
    gemini_api_key: str
    confidence_threshold: float = 65.0
    consensus_enabled: bool = True
    request_timeout: int = 30
    retry_attempts: int = 2
    rate_limit_per_hour: int = 100
    
    @classmethod
    def from_env(cls):
        return cls(
            huggingface_token=os.getenv('HUGGINGFACE_API_TOKEN', ''),
            gemini_api_key=os.getenv('GOOGLE_GEMINI_API_KEY', ''),
            confidence_threshold=float(os.getenv('AI_CONFIDENCE_THRESHOLD', 65.0)),
            consensus_enabled=os.getenv('TWO_AI_CONSENSUS_ENABLED', 'true').lower() == 'true',
            request_timeout=int(os.getenv('AI_REQUEST_TIMEOUT', 30)),
            retry_attempts=int(os.getenv('AI_RETRY_ATTEMPTS', 2)),
            rate_limit_per_hour=int(os.getenv('AI_RATE_LIMIT_PER_HOUR', 100))
        )

@dataclass
class TradingConfig:
    """Trading Configuration"""
    enabled: bool = True
    dry_run_mode: bool = False
    update_interval_minutes: int = 20
    max_position_size_percent: float = 4.0
    stop_loss_percent: float = 2.0
    take_profit_percent: float = 3.0
    
    # Strategy Enablement
    trend_following_enabled: bool = True
    mean_reversion_enabled: bool = True
    breakout_enabled: bool = True
    news_trading_enabled: bool = True
    
    @classmethod
    def from_env(cls):
        return cls(
            enabled=os.getenv('TRADING_ENABLED', 'true').lower() == 'true',
            dry_run_mode=os.getenv('DRY_RUN_MODE', 'false').lower() == 'true',
            update_interval_minutes=int(os.getenv('UPDATE_INTERVAL_MINUTES', 20)),
            max_position_size_percent=float(os.getenv('MAX_POSITION_SIZE_PERCENT', 4.0)),
            stop_loss_percent=float(os.getenv('STOP_LOSS_PERCENT', 2.0)),
            take_profit_percent=float(os.getenv('TAKE_PROFIT_PERCENT', 3.0)),
            trend_following_enabled=os.getenv('TREND_FOLLOWING_ENABLED', 'true').lower() == 'true',
            mean_reversion_enabled=os.getenv('MEAN_REVERSION_ENABLED', 'true').lower() == 'true',
            breakout_enabled=os.getenv('BREAKOUT_ENABLED', 'true').lower() == 'true',
            news_trading_enabled=os.getenv('NEWS_TRADING_ENABLED', 'true').lower() == 'true'
        )

@dataclass
class CapitalConfig:
    """Capital.com API Configuration"""
    api_key: str
    password: str
    email: str
    base_url: str = "https://demo-api-capital.backend-capital.com"
    
    @classmethod
    def from_env(cls):
        return cls(
            api_key=os.getenv('CAPITAL_API_KEY', ''),
            password=os.getenv('CAPITAL_PASSWORD', ''),
            email=os.getenv('CAPITAL_EMAIL', '')
        )
class Config:
    """Main Configuration Class"""
    
    def __init__(self):
        self.ai = AIConfig.from_env()
        self.trading = TradingConfig.from_env()
        self.capital = CapitalConfig.from_env()
        
        # Flask Configuration
        self.secret_key = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
        self.debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        self.port = int(os.getenv('PORT', 10000))
        self.host = os.getenv('HOST', '0.0.0.0')
        
        # Database Configuration
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///trading_ai_system.db')
        self.database_cleanup_hours = int(os.getenv('DATABASE_CLEANUP_HOURS', 24))
        
        # Logging Configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_file = os.getenv('LOG_FILE', 'trading_ai_system.log')
        self.log_max_bytes = int(os.getenv('LOG_MAX_BYTES', 10485760))
        self.log_backup_count = int(os.getenv('LOG_BACKUP_COUNT', 5))
        
        # Memory Management (für Render)
        self.memory_cleanup_interval = int(os.getenv('MEMORY_CLEANUP_INTERVAL_MINUTES', 30))
        self.max_historical_days = int(os.getenv('MAX_HISTORICAL_DAYS', 30))
        self.max_log_entries = int(os.getenv('MAX_LOG_ENTRIES', 1000))
        
        # Rate Limiting
        self.api_rate_limit_per_minute = int(os.getenv('API_RATE_LIMIT_PER_MINUTE', 50))
        
        # Demo Accounts
        self.demo_accounts = {
            'trend_following': os.getenv('DEMO_ACCOUNT_1_TREND', ''),
            'mean_reversion': os.getenv('DEMO_ACCOUNT_2_MEAN', ''),
            'breakout': os.getenv('DEMO_ACCOUNT_3_BREAKOUT', ''),
            'news_trading': os.getenv('DEMO_ACCOUNT_4_NEWS', '')
        }
        
        # Validation
        self.validate()
    
    def validate(self):
        """Validate Configuration"""
        errors = []
        
        # Capital.com Validation
        if not self.capital.api_key:
            errors.append("CAPITAL_API_KEY missing")
        if not self.capital.password:
            errors.append("CAPITAL_PASSWORD missing")
        if not self.capital.email:
            errors.append("CAPITAL_EMAIL missing")
            
        # AI Validation
        if not self.ai.huggingface_token:
            errors.append("HUGGINGFACE_API_TOKEN missing")
        if not self.ai.gemini_api_key:
            errors.append("GOOGLE_GEMINI_API_KEY missing")
            
        # Trading Validation
        if self.trading.max_position_size_percent <= 0:
            errors.append("MAX_POSITION_SIZE_PERCENT must be > 0")
        if self.trading.update_interval_minutes < 1:
            errors.append("UPDATE_INTERVAL_MINUTES must be >= 1")
            
        if errors:
            logger.error(f"Configuration Errors: {', '.join(errors)}")
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
        
        logger.info("Configuration validation successful")
    
    def get_strategy_assets(self, strategy_name: str) -> List[str]:
        """Get assets for specific strategy"""
        asset_mapping = {
            'trend_following': ['EURUSD', 'BITCOIN'],  # EUR/USD, BTC/USDT 
            'mean_reversion': ['GERMANY30', 'GOLD'],   # DAX, GOLD
            'breakout': ['US100', 'TESLA'],           # NASDAQ, TSLA
            'news_trading': ['EURUSD', 'OIL_CRUDE']   # EUR/USD, WTI
        }
        return asset_mapping.get(strategy_name, [])
    
    def get_strategy_account_id(self, strategy_name: str) -> str:
        """Get demo account ID for strategy"""
        return self.demo_accounts.get(strategy_name, '')
    
    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """Check if strategy is enabled"""
        strategy_flags = {
            'trend_following': self.trading.trend_following_enabled,
            'mean_reversion': self.trading.mean_reversion_enabled,
            'breakout': self.trading.breakout_enabled,
            'news_trading': self.trading.news_trading_enabled
        }
        return strategy_flags.get(strategy_name, False)
