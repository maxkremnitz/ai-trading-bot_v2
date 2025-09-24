"""
Hugging Face API Provider
Verwendet kostenlose Inference API für Sentiment Analysis
"""

import requests
import time
import logging
from typing import Dict, Any, Optional
import json

from .base_provider import BaseAIProvider, AISignal, MarketData, SignalType

logger = logging.getLogger(__name__)

class HuggingFaceProvider(BaseAIProvider):
    """
    Hugging Face Provider für Trading Signal Analysis
    Nutzt verschiedene Modelle für Sentiment und Classification
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("HuggingFace", config)
        
        self.api_token = config.get('api_token') or config.get('huggingface_token')
        if not self.api_token:
            raise ValueError("HuggingFace API Token fehlt")
        
        self.base_url = "https://api-inference.huggingface.co/models"
        
        # Modell-Konfiguration
        self.models = {
            'sentiment': config.get('sentiment_model', 'cardiffnlp/twitter-roberta-base-sentiment-latest'),
            'financial': config.get('financial_model', 'ProsusAI/finbert'),
            'classification': config.get('classification_model', 'facebook/bart-large-mnli')
        }
        
        self.request_timeout = config.get('timeout', 30)
        self.min_request_interval = config.get('min_interval_seconds', 3.0)  # 3 Sekunden zwischen Requests
        
        logger.info(f"HuggingFace Provider initialisiert - Models: {list(self.models.keys())}")
    
    def analyze_market_data(self, market_data: MarketData, strategy_context: str) -> AISignal:
        """
        Analysiert Marktdaten mit HuggingFace Models
        """
        start_time = time.time()
        
        try:
            self._rate_limit_check()
            self.request_count += 1
            
            # Analysis Text erstellen
            analysis_text = self._create_analysis_prompt(market_data, strategy_context)
            
            # Primary Analysis mit Financial Model
            primary_result = self._query_financial_model(analysis_text)
            
            if not primary_result:
                # Fallback auf Sentiment Model
                primary_result = self._query_sentiment_model(analysis_text)
            
            if not primary_result:
                self.error_count += 1
                return self._create_error_signal("Alle HuggingFace Models fehlgeschlagen")
            
            # Signal extrahieren
            signal_type, confidence, reasoning = self._interpret_huggingface_result(
                primary_result, analysis_text
            )
            
            processing_time = time.time() - start_time
            
            return AISignal(
                provider=self.name,
                signal_type=signal_type,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=time.time(),
                processing_time=processing_time,
                raw_response=primary_result
            )
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"HuggingFace Analysis Error: {e}")
            return self._create_error_signal(str(e))
    
    def _create_analysis_prompt(self, market_data: MarketData, strategy_context: str) -> str:
        """Erstellt Analyse-Prompt für HuggingFace"""
        prompt = f"Trading Analysis for {market_data.symbol}:\n\n"
        prompt += f"Current Price: ${market_data.current_price:.2f}\n"
        prompt += f"24h Change: {market_data.price_change_24h:+.2f}%\n"
        
        if market_data.technical_indicators:
            prompt += "Technical Analysis:\n"
            for indicator, value in market_data.technical_indicators.items():
                prompt += f"- {indicator}: {value:.2f}\n"
        
        prompt += f"\nStrategy Context: {strategy_context}\n"
        
        if market_data.news_sentiment:
            prompt += f"News Sentiment: {market_data.news_sentiment}\n"
        
        prompt += "\nBased on this data, what is the trading recommendation?"
        
        return prompt
    
    def _query_financial_model(self, text: str) -> Optional[Dict]:
        """Query Financial Sentiment Model (FinBERT)"""
        try:
            model_url = f"{self.base_url}/{self.models['financial']}"
            
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            payload = {"inputs": text}
            
            response = requests.post(
                model_url,
                headers=headers,
                json=payload,
                timeout=self.request_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"FinBERT Response: {result}")
                return result
            elif response.status_code == 503:
                logger.warning("FinBERT Model loading, trying sentiment model...")
                return None
            else:
                logger.error(f"FinBERT Error: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            logger.error(f"FinBERT Query Error: {e}")
            return None
    
    def _query_sentiment_model(self, text: str) -> Optional[Dict]:
        """Query General Sentiment Model"""
        try:
            model_url = f"{self.base_url}/{self.models['sentiment']}"
            
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            payload = {"inputs": text}
            
            response = requests.post(
                model_url,
                headers=headers,
                json=payload,
                timeout=self.request_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"Sentiment Response: {result}")
                return result
            else:
                logger.error(f"Sentiment Error: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            logger.error(f"Sentiment Query Error: {e}")
            return None
        def _interpret_huggingface_result(self, result: Dict, original_text: str) -> tuple[SignalType, float, str]:
        """
        Interpretiert HuggingFace Model Response
        """
        try:
            # Result kann verschiedene Formate haben
            if isinstance(result, list) and len(result) > 0:
                # Format: [{"label": "POSITIVE", "score": 0.9}]
                if isinstance(result[0], list):
                    result = result[0]  # Nested list
                
                # Sortiere nach Score
                sorted_results = sorted(result, key=lambda x: x.get('score', 0), reverse=True)
                top_result = sorted_results[0]
                
                label = top_result.get('label', '').upper()
                score = float(top_result.get('score', 0))
                
                # Label Mapping
                signal_type = self._map_label_to_signal(label)
                confidence = min(100.0, score * 100)  # 0-1 zu 0-100
                
                reasoning = f"HuggingFace Analysis: {label} (confidence: {confidence:.1f}%)"
                
                # Füge Context hinzu
                if len(sorted_results) > 1:
                    second_label = sorted_results[1].get('label', '')
                    second_score = float(sorted_results[1].get('score', 0)) * 100
                    reasoning += f", Alternative: {second_label} ({second_score:.1f}%)"
                
                return signal_type, confidence, reasoning
            
            else:
                # Fallback: Text-basierte Interpretation
                return self._extract_signal_from_text(str(result))
        
        except Exception as e:
            logger.error(f"HuggingFace Result Interpretation Error: {e}")
            return SignalType.HOLD, 30.0, f"Interpretation Error: {str(e)}"
    
    def _map_label_to_signal(self, label: str) -> SignalType:
        """
        Mapped HuggingFace Labels zu Trading Signals
        """
        label = label.upper()
        
        # FinBERT Labels
        if label in ['POSITIVE', 'BULLISH']:
            return SignalType.BUY
        elif label in ['NEGATIVE', 'BEARISH']:
            return SignalType.SELL
        elif label in ['NEUTRAL']:
            return SignalType.HOLD
        
        # Twitter Sentiment Labels
        elif 'LABEL_2' in label or 'POS' in label:  # Positive
            return SignalType.BUY
        elif 'LABEL_0' in label or 'NEG' in label:  # Negative
            return SignalType.SELL
        elif 'LABEL_1' in label or 'NEU' in label:  # Neutral
            return SignalType.HOLD
        
        # Default
        else:
            return SignalType.HOLD
    
    def test_connection(self) -> bool:
        """Testet HuggingFace API Verbindung"""
        try:
            test_text = "Bitcoin is showing strong momentum today."
            result = self._query_sentiment_model(test_text)
            
            if result:
                logger.info("HuggingFace API Test erfolgreich")
                return True
            else:
                logger.error("HuggingFace API Test fehlgeschlagen")
                return False
        
        except Exception as e:
            logger.error(f"HuggingFace Connection Test Error: {e}")
            return False
