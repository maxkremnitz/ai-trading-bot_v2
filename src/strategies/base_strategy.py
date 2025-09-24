
"""
Base Strategy Class
Abstrakte Basis-Klasse für alle Trading-Strategien
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import pandas as pd
import numpy as np

from ..ai.voting_system import AIVotingSystem, ConsensusSignal
from ..ai.base_provider import MarketData, SignalType
from ..api.capital_api import CapitalComAPI, TradeResult

logger = logging.getLogger(__name__)

class StrategyStatus(Enum):
    """Strategy Status"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    ERROR = "ERROR"
    PAUSED = "PAUSED"

@dataclass
class StrategySignal:
    """Strategy Trading Signal"""
    strategy_name: str
    symbol: str
    action: SignalType
    confidence: float
    price: float
    stop_loss: float
    take_profit: float
    position_size: float
    reasoning: str
    technical_data: Dict[str, float]
    ai_consensus: Optional[ConsensusSignal] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    @property
    def is_executable(self) -> bool:
        """Prüft ob Signal ausführbar ist"""
        return (
            self.action in [SignalType.BUY, SignalType.SELL] and
            self.confidence >= 50.0 and
            self.price > 0 and
            self.position_size > 0
        )

class BaseStrategy(ABC):
    """
    Abstrakte Basis-Klasse für Trading-Strategien
    Integriert AI Consensus Voting und Capital.com API
    """
    
    def __init__(self, name: str, config: Dict[str, Any], ai_voting_system: AIVotingSystem, 
                 capital_api: CapitalComAPI):
        self.name = name
        self.config = config
        self.ai_voting_system = ai_voting_system
        self.capital_api = capital_api
        
        # Strategy Configuration
        self.enabled = config.get('enabled', True)
        self.assets = config.get('assets', [])
        self.max_position_size_percent = config.get('max_position_size_percent', 4.0)
        self.stop_loss_percent = config.get('stop_loss_percent', 2.0)
        self.take_profit_percent = config.get('take_profit_percent', 3.0)
        self.min_confidence = config.get('min_confidence', 65.0)
        
        # Strategy State
        self.status = StrategyStatus.INACTIVE
        self.last_analysis_time = 0
        self.last_trade_time = 0
        self.analysis_interval = config.get('analysis_interval_minutes', 30) * 60
        self.min_trade_interval = config.get('min_trade_interval_minutes', 60) * 60
        
        # Performance Tracking
        self.total_signals = 0
        self.executed_trades = 0
        self.successful_trades = 0
        self.total_pnl = 0.0
        
        # Data Storage
        self.market_data_cache = {}
        self.recent_signals = []
        self.active_positions = []
        
        logger.info(f"Strategy {self.name} initialisiert für Assets: {self.assets}")
    
    @abstractmethod
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Berechnet strategy-spezifische technische Indikatoren
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dict mit technischen Indikatoren
        """
        pass
    
    @abstractmethod
    def generate_base_signal(self, symbol: str, market_data: MarketData, 
                           technical_indicators: Dict[str, float]) -> Optional[StrategySignal]:
        """
        Generiert Basis-Signal ohne AI Consensus
        
        Args:
            symbol: Trading Symbol
            market_data: Marktdaten
            technical_indicators: Technische Indikatoren
            
        Returns:
            StrategySignal oder None
        """
        pass
    
    @abstractmethod
    def get_strategy_context(self) -> str:
        """
        Returns strategy-spezifischen Context für AI Analysis
        """
        pass
    
    def analyze_and_signal(self) -> List[StrategySignal]:
        """
        Hauptanalyse-Funktion: Analysiert alle Assets und generiert Signals
        """
        if not self.enabled:
            return []
        
        current_time = time.time()
        
        # Rate Limiting Check
        if current_time - self.last_analysis_time < self.analysis_interval:
            return []
        
        try:
            self.status = StrategyStatus.ACTIVE
            signals = []
            
            for asset in self.assets:
                try:
                    signal = self._analyze_single_asset(asset)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.error(f"Asset Analysis Error {asset}: {e}")
            
            self.last_analysis_time = current_time
            self.recent_signals.extend(signals)
            
            # Nur letzte 10 Signale behalten
            self.recent_signals = self.recent_signals[-10:]
            
            logger.info(f"{self.name}: {len(signals)} Signale generiert")
            return signals
        
        except Exception as e:
            logger.error(f"Strategy Analysis Error {self.name}: {e}")
            self.status = StrategyStatus.ERROR
            return []
    
    def _analyze_single_asset(self, symbol: str) -> Optional[StrategySignal]:
        """Analysiert einzelnes Asset"""
        try:
            # Market Data holen
            market_data = self._get_market_data(symbol)
            if not market_data:
                return None
            
            # Historical Data für technische Analyse
            historical_data = self._get_historical_data(symbol)
            if historical_data is None or len(historical_data) < 20:
                logger.warning(f"Insufficient historical data for {symbol}")
                return None
            
            # Technische Indikatoren berechnen
            technical_indicators = self.calculate_technical_indicators(historical_data)
            
            # Market Data mit technischen Indikatoren erweitern
            market_data.technical_indicators = technical_indicators
            
            # Basis-Signal generieren
            base_signal = self.generate_base_signal(symbol, market_data, technical_indicators)
            if not base_signal:
                return None
            
            # AI Consensus einholen
            strategy_context = self.get_strategy_context()
            ai_consensus = self.ai_voting_system.get_trading_consensus(market_data, strategy_context)
            
            # AI Consensus in Signal integrieren
            final_signal = self._integrate_ai_consensus(base_signal, ai_consensus)
            
            self.total_signals += 1
            
            return final_signal
        
        except Exception as e:
            logger.error(f"Single Asset Analysis Error {symbol}: {e}")
            return None
    def _integrate_ai_consensus(self, base_signal: StrategySignal, 
                              ai_consensus: ConsensusSignal) -> StrategySignal:
        """
        Integriert AI Consensus in das Basis-Signal
        """
        # AI Consensus anhängen
        base_signal.ai_consensus = ai_consensus
        
        # Signal nur ausführbar wenn AI Consensus stimmt
        if ai_consensus.is_tradeable and ai_consensus.final_action == base_signal.action:
            # Boost confidence wenn AI zustimmt
            combined_confidence = (base_signal.confidence + ai_consensus.consensus_confidence) / 2
            base_signal.confidence = min(100.0, combined_confidence * 1.1)  # 10% Boost
            
            # Erweiterte Reasoning
            base_signal.reasoning += f" | AI Consensus: {ai_consensus.action_summary} - {ai_consensus.reasoning[:100]}"
        else:
            # AI stimmt nicht zu oder nicht tradeable
            base_signal.action = SignalType.HOLD
            base_signal.confidence = min(base_signal.confidence, 40.0)
            base_signal.reasoning += f" | AI Override: {ai_consensus.action_summary}"
        
        return base_signal
    
    def execute_signal(self, signal: StrategySignal) -> Optional[TradeResult]:
        """
        Führt Trading-Signal aus
        """
        if not signal.is_executable:
            logger.info(f"Signal not executable: {signal.action.value} {signal.symbol}")
            return None
        
        current_time = time.time()
        
        # Min Trade Interval Check
        if current_time - self.last_trade_time < self.min_trade_interval:
            logger.info(f"Trade interval not reached: {self.name}")
            return None
        
        try:
            # Account für diese Strategie wechseln
            if not self.capital_api.switch_to_strategy_account(self.name.lower().replace(' ', '_')):
                logger.error(f"Account switch failed for {self.name}")
                return None
            
            # Stop Loss und Take Profit Distanzen berechnen
            stop_distance = int(abs(signal.price - signal.stop_loss))
            profit_distance = int(abs(signal.take_profit - signal.price))
            
            # Order platzieren
            trade_result = self.capital_api.place_order(
                epic=signal.symbol,
                direction=signal.action.value,
                size=signal.position_size,
                stop_distance=max(50, stop_distance),  # Min 50 für Capital.com
                profit_distance=max(50, profit_distance),
                use_percentage_size=True,
                percentage=self.max_position_size_percent
            )
            
            if trade_result and trade_result.success:
                self.executed_trades += 1
                self.last_trade_time = current_time
                logger.info(f"Trade executed: {signal.symbol} {signal.action.value} - {trade_result.deal_reference}")
            
            return trade_result
        
        except Exception as e:
            logger.error(f"Trade Execution Error {self.name}: {e}")
            return None
    
    def _get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Holt aktuelle Marktdaten für Symbol"""
        try:
            # Cache Check
            cache_key = f"{symbol}_{int(time.time() / 300)}"  # 5min cache
            if cache_key in self.market_data_cache:
                return self.market_data_cache[cache_key]
            
            # Hier würdest du echte Market Data API integrieren
            # Für Demo: Simulierte Daten
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            
            if len(hist) == 0:
                return None
            
            current_price = float(hist['Close'].iloc[-1])
            prev_price = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            price_change = ((current_price - prev_price) / prev_price) * 100
            volume = float(hist['Volume'].iloc[-1])
            
            market_data = MarketData(
                symbol=symbol,
                current_price=current_price,
                price_change_24h=price_change,
                volume=volume,
                technical_indicators={}
            )
            
            # Cache speichern
            self.market_data_cache[cache_key] = market_data
            
            return market_data
        
        except Exception as e:
            logger.error(f"Market Data Error {symbol}: {e}")
            return None
    
    def _get_historical_data(self, symbol: str, period: str = "3mo") -> Optional[pd.DataFrame]:
        """Holt historische Daten für technische Analyse"""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if len(data) < 20:
                return None
            
            return data
        
        except Exception as e:
            logger.error(f"Historical Data Error {symbol}: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Performance Statistiken"""
        win_rate = (self.successful_trades / max(1, self.executed_trades)) * 100
        
        return {
            'strategy_name': self.name,
            'status': self.status.value,
            'enabled': self.enabled,
            'assets': self.assets,
            'total_signals': self.total_signals,
            'executed_trades': self.executed_trades,
            'successful_trades': self.successful_trades,
            'win_rate': round(win_rate, 1),
            'total_pnl': round(self.total_pnl, 2),
            'last_analysis_time': self.last_analysis_time,
            'last_trade_time': self.last_trade_time,
            'recent_signals_count': len(self.recent_signals),
            'active_positions_count': len(self.active_positions)
        }
    
    def update_position_status(self):
        """Aktualisiert Status der aktiven Positionen"""
        try:
            current_positions = self.capital_api.get_positions()
            self.active_positions = current_positions
            
            # PnL tracking (vereinfacht)
            total_pnl = sum(pos.pnl for pos in current_positions)
            self.total_pnl = total_pnl
        
        except Exception as e:
            logger.error(f"Position Update Error {self.name}: {e}")
    
    def pause(self):
        """Pausiert Strategy"""
        self.status = StrategyStatus.PAUSED
        logger.info(f"Strategy {self.name} pausiert")
    
    def resume(self):
        """Setzt Strategy fort"""
        if self.enabled:
            self.status = StrategyStatus.ACTIVE
            logger.info(f"Strategy {self.name} fortgesetzt")
