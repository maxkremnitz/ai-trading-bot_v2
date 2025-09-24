"""
Basis AI Provider
Abstrakte Klasse für alle AI Provider
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Signal Types"""
    BUY = "BUY"
    SELL = "SELL"  
    HOLD = "HOLD"

@dataclass
class AISignal:
    """AI Signal Datenstruktur"""
    provider: str
    signal_type: SignalType
    confidence: float  # 0-100
    reasoning: str
    timestamp: float
    processing_time: float
    raw_response: Optional[Dict] = None
    error: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        """Prüft ob Signal valid ist"""
        return (
            self.error is None and
            self.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD] and
            0 <= self.confidence <= 100
        )
    
    @property
    def is_actionable(self) -> bool:
        """Prüft ob Signal handelbar ist (BUY/SELL)"""
        return self.is_valid and self.signal_type in [SignalType.BUY, SignalType.SELL]

@dataclass
class MarketData:
    """Market Data für AI Analysis"""
    symbol: str
    current_price: float
    price_change_24h: float
    volume: float
    technical_indicators: Dict[str, float]
    news_sentiment: Optional[str] = None
    additional_context: Optional[Dict] = None
    
    def to_analysis_text(self) -> str:
        """Konvertiert zu Text für AI Analysis"""
        text = f"Symbol: {self.symbol}\n"
        text += f"Current Price: ${self.current_price:.2f}\n"
        text += f"24h Change: {self.price_change_24h:+.2f}%\n"
        text += f"Volume: {self.volume:,.0f}\n"
        
        if self.technical_indicators:
            text += "Technical Indicators:\n"
            for indicator, value in self.technical_indicators.items():
                text += f"  {indicator}: {value:.2f}\n"
        
        if self.news_sentiment:
            text += f"News Sentiment: {self.news_sentiment}\n"
        
        if self.additional_context:
            text += "Additional Context:\n"
            for key, value in self.additional_context.items():
                text += f"  {key}: {value}\n"
        
        return text

class BaseAIProvider(ABC):
    """
    Basis-Klasse für alle AI Provider
    Definiert Interface für Hugging Face, Gemini, etc.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.last_request_time = 0
        self.request_count = 0
        self.error_count = 0
        self.min_request_interval = config.get('min_interval_seconds', 1.0)
        
        logger.info(f"AI Provider {name} initialisiert")
    
    @abstractmethod
    def analyze_market_data(self, market_data: MarketData, strategy_context: str) -> AISignal:
        """
        Analysiert Marktdaten und gibt Trading-Signal zurück
        
        Args:
            market_data: Marktdaten für Analyse
            strategy_context: Kontext der Trading-Strategie
            
        Returns:
            AISignal: Analyseergebnis mit BUY/SELL/HOLD
        """
        pass
    
    def _rate_limit_check(self):
        """Rate Limiting zwischen Requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.debug(f"{self.name}: Rate limit - warte {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _create_error_signal(self, error_message: str) -> AISignal:
        """Erstellt Error Signal"""
        return AISignal(
            provider=self.name,
            signal_type=SignalType.HOLD,
            confidence=0.0,
            reasoning=f"Error: {error_message}",
            timestamp=time.time(),
            processing_time=0.0,
            error=error_message
        )
    
    def _extract_signal_from_text(self, response_text: str) -> tuple[SignalType, float, str]:
        """
        Extrahiert Signal aus AI Response Text
        
        Returns:
            tuple[SignalType, float, str]: (signal_type, confidence, reasoning)
        """
        text_lower = response_text.lower()
        
        # Signal Type Detection
        buy_keywords = ['buy', 'bullish', 'positive', 'long', 'upward', 'increase']
        sell_keywords = ['sell', 'bearish', 'negative', 'short', 'downward', 'decrease']
        
        buy_score = sum(1 for keyword in buy_keywords if keyword in text_lower)
        sell_score = sum(1 for keyword in sell_keywords if keyword in text_lower)
        
        if buy_score > sell_score and buy_score > 0:
            signal_type = SignalType.BUY
        elif sell_score > buy_score and sell_score > 0:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        # Confidence Extraction (vereinfacht)
        confidence = 50.0  # Default
        
        # Suche nach Zahlen mit % oder confidence indicators
        import re
        confidence_matches = re.findall(r'(\d+(?:\.\d+)?)%', response_text)
        if confidence_matches:
            confidence = min(100.0, max(0.0, float(confidence_matches[0])))
        elif 'strong' in text_lower or 'very' in text_lower:
            confidence = 75.0
        elif 'weak' in text_lower or 'slight' in text_lower:
            confidence = 35.0
        
        # Reasoning (erste 200 Zeichen)
        reasoning = response_text[:200] + ("..." if len(response_text) > 200 else "")
        
        return signal_type, confidence, reasoning
    
    def get_stats(self) -> Dict[str, Any]:
        """Provider Statistiken"""
        return {
            'name': self.name,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'success_rate': ((self.request_count - self.error_count) / max(1, self.request_count)) * 100,
            'last_request_time': self.last_request_time
        }
