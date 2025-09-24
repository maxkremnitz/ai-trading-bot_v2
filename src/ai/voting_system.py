"""
AI Consensus Voting System
2-AI Unanimity Voting zwischen HuggingFace und Gemini
"""

import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import statistics

from .base_provider import BaseAIProvider, AISignal, MarketData, SignalType
from .huggingface_provider import HuggingFaceProvider
from .gemini_provider import GeminiProvider

logger = logging.getLogger(__name__)

class VotingResult(Enum):
    """Voting Results"""
    UNANIMOUS_BUY = "UNANIMOUS_BUY"
    UNANIMOUS_SELL = "UNANIMOUS_SELL"  
    UNANIMOUS_HOLD = "UNANIMOUS_HOLD"
    NO_CONSENSUS = "NO_CONSENSUS"
    ERROR = "ERROR"

@dataclass
class ConsensusSignal:
    """Final Consensus Signal"""
    voting_result: VotingResult
    final_action: SignalType
    consensus_confidence: float
    individual_signals: List[AISignal]
    reasoning: str
    timestamp: float
    processing_time: float
    
    @property
    def is_tradeable(self) -> bool:
        """Prüft ob Signal handelbar ist"""
        return (
            self.voting_result in [VotingResult.UNANIMOUS_BUY, VotingResult.UNANIMOUS_SELL] and
            self.consensus_confidence >= 50.0
        )
    
    @property
    def action_summary(self) -> str:
        """Kurze Zusammenfassung der Aktion"""
        if self.voting_result == VotingResult.UNANIMOUS_BUY:
            return f"BUY ({self.consensus_confidence:.1f}%)"
        elif self.voting_result == VotingResult.UNANIMOUS_SELL:
            return f"SELL ({self.consensus_confidence:.1f}%)"
        else:
            return f"HOLD - {self.voting_result.value}"

class AIVotingSystem:
    """
    AI Consensus Voting System
    Verwaltet 2 AI Provider und führt Unanimity Voting durch
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 65.0)
        self.consensus_enabled = config.get('consensus_enabled', True)
        
        # Provider initialisieren
        self.providers: List[BaseAIProvider] = []
        
        # HuggingFace Provider
        if config.get('huggingface_config'):
            try:
                hf_provider = HuggingFaceProvider(config['huggingface_config'])
                self.providers.append(hf_provider)
                logger.info("HuggingFace Provider hinzugefügt")
            except Exception as e:
                logger.error(f"HuggingFace Provider Init Error: {e}")
        
        # Gemini Provider
        if config.get('gemini_config'):
            try:
                gemini_provider = GeminiProvider(config['gemini_config'])
                self.providers.append(gemini_provider)
                logger.info("Gemini Provider hinzugefügt")
            except Exception as e:
                logger.error(f"Gemini Provider Init Error: {e}")
        
        if len(self.providers) == 0:
            raise ValueError("Keine AI Provider erfolgreich initialisiert")
        
        # Statistics
        self.total_votes = 0
        self.unanimous_votes = 0
        self.buy_votes = 0
        self.sell_votes = 0
        self.hold_votes = 0
        
        logger.info(f"AI Voting System initialisiert mit {len(self.providers)} Providern")
    
    def get_trading_consensus(self, market_data: MarketData, strategy_context: str) -> ConsensusSignal:
        """
        Hauptfunktion: Holt Consensus von allen AI Providern
        """
        start_time = time.time()
        
        if not self.consensus_enabled:
            # Fallback: Nur erster Provider
            if self.providers:
                signal = self.providers[0].analyze_market_data(market_data, strategy_context)
                return self._create_single_provider_result(signal, start_time)
        
        # Alle Provider abfragen
        individual_signals = []
        
        for provider in self.providers:
            try:
                signal = provider.analyze_market_data(market_data, strategy_context)
                individual_signals.append(signal)
                logger.info(f"{provider.name}: {signal.signal_type.value} ({signal.confidence:.1f}%)")
            except Exception as e:
                logger.error(f"Provider {provider.name} Error: {e}")
                # Erstelle Error Signal
                error_signal = provider._create_error_signal(str(e))
                individual_signals.append(error_signal)
        
        # Consensus auswerten
        consensus_result = self._evaluate_consensus(individual_signals)
        processing_time = time.time() - start_time
        
        self.total_votes += 1
        
        return ConsensusSignal(
            voting_result=consensus_result['result'],
            final_action=consensus_result['action'],
            consensus_confidence=consensus_result['confidence'],
            individual_signals=individual_signals,
            reasoning=consensus_result['reasoning'],
            timestamp=time.time(),
            processing_time=processing_time
        )
    def _evaluate_consensus(self, signals: List[AISignal]) -> Dict[str, Any]:
        """
        Evaluiert Consensus zwischen den AI Signals
        """
        valid_signals = [s for s in signals if s.is_valid]
        
        if len(valid_signals) == 0:
            return {
                'result': VotingResult.ERROR,
                'action': SignalType.HOLD,
                'confidence': 0.0,
                'reasoning': "Keine validen AI Signals erhalten"
            }
        
        # Signal Types sammeln
        signal_types = [s.signal_type for s in valid_signals]
        confidences = [s.confidence for s in valid_signals]
        
        # Unanimity Check
        unique_signals = set(signal_types)
        
        if len(unique_signals) == 1:
            # Unanimity erreicht
            unanimous_signal = signal_types[0]
            avg_confidence = statistics.mean(confidences)
            
            # Confidence Threshold prüfen
            if avg_confidence >= self.confidence_threshold:
                if unanimous_signal == SignalType.BUY:
                    self.unanimous_votes += 1
                    self.buy_votes += 1
                    result = VotingResult.UNANIMOUS_BUY
                elif unanimous_signal == SignalType.SELL:
                    self.unanimous_votes += 1
                    self.sell_votes += 1
                    result = VotingResult.UNANIMOUS_SELL
                else:
                    self.hold_votes += 1
                    result = VotingResult.UNANIMOUS_HOLD
                
                reasoning = f"Unanimous {unanimous_signal.value}: "
                reasoning += f"All {len(valid_signals)} AIs agree. "
                reasoning += f"Avg confidence: {avg_confidence:.1f}%. "
                reasoning += f"Individual: {[f'{s.provider}({s.confidence:.0f}%)' for s in valid_signals]}"
                
                return {
                    'result': result,
                    'action': unanimous_signal,
                    'confidence': avg_confidence,
                    'reasoning': reasoning
                }
            else:
                # Unanimous aber zu niedrige Confidence
                reasoning = f"Unanimous {unanimous_signal.value} but low confidence: {avg_confidence:.1f}% < {self.confidence_threshold}%"
                return {
                    'result': VotingResult.NO_CONSENSUS,
                    'action': SignalType.HOLD,
                    'confidence': avg_confidence,
                    'reasoning': reasoning
                }
        else:
            # No Consensus
            self.hold_votes += 1
            
            # Detailed reasoning
            signal_summary = {}
            for signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]:
                count = signal_types.count(signal_type)
                if count > 0:
                    signal_summary[signal_type.value] = count
            
            reasoning = f"No consensus: {signal_summary}. "
            reasoning += f"Individual signals: {[(s.provider, s.signal_type.value, f'{s.confidence:.0f}%') for s in valid_signals]}"
            
            return {
                'result': VotingResult.NO_CONSENSUS,
                'action': SignalType.HOLD,
                'confidence': statistics.mean(confidences),
                'reasoning': reasoning
            }
    
    def _create_single_provider_result(self, signal: AISignal, start_time: float) -> ConsensusSignal:
        """Erstellt Result für Single Provider Mode"""
        if signal.confidence >= self.confidence_threshold and signal.is_actionable:
            if signal.signal_type == SignalType.BUY:
                result = VotingResult.UNANIMOUS_BUY
            else:
                result = VotingResult.UNANIMOUS_SELL
        else:
            result = VotingResult.NO_CONSENSUS
        
        return ConsensusSignal(
            voting_result=result,
            final_action=signal.signal_type,
            consensus_confidence=signal.confidence,
            individual_signals=[signal],
            reasoning=f"Single provider mode: {signal.reasoning}",
            timestamp=time.time(),
            processing_time=time.time() - start_time
        )
    
    def get_voting_stats(self) -> Dict[str, Any]:
        """Voting Statistiken"""
        success_rate = (self.unanimous_votes / max(1, self.total_votes)) * 100
        
        return {
            'total_votes': self.total_votes,
            'unanimous_votes': self.unanimous_votes,
            'consensus_rate': round(success_rate, 1),
            'buy_votes': self.buy_votes,
            'sell_votes': self.sell_votes,
            'hold_votes': self.hold_votes,
            'active_providers': len(self.providers),
            'provider_stats': [provider.get_stats() for provider in self.providers],
            'confidence_threshold': self.confidence_threshold
        }
    
    def test_all_providers(self) -> Dict[str, bool]:
        """Testet alle AI Provider"""
        results = {}
        
        for provider in self.providers:
            try:
                if hasattr(provider, 'test_connection'):
                    results[provider.name] = provider.test_connection()
                else:
                    # Basic test
                    test_data = MarketData(
                        symbol="TEST",
                        current_price=100.0,
                        price_change_24h=2.5,
                        volume=1000000,
                        technical_indicators={"RSI": 65.0, "MACD": 0.5}
                    )
                    signal = provider.analyze_market_data(test_data, "Test context")
                    results[provider.name] = signal.is_valid
            except Exception as e:
                logger.error(f"Provider Test {provider.name}: {e}")
                results[provider.name] = False
        
        return results
