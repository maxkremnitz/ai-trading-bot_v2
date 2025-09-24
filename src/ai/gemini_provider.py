"""
Google Gemini API Provider
Verwendet kostenlose Gemini API für erweiterte Marktanalyse
"""

import google.generativeai as genai
import time
import logging
from typing import Dict, Any, Optional
import json

from .base_provider import BaseAIProvider, AISignal, MarketData, SignalType

logger = logging.getLogger(__name__)

class GeminiProvider(BaseAIProvider):
    """
    Google Gemini Provider für Trading Analysis
    Nutzt Gemini-Pro für erweiterte Marktanalyse
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Gemini", config)
        
        self.api_key = config.get('api_key') or config.get('gemini_api_key')
        if not self.api_key:
            raise ValueError("Google Gemini API Key fehlt")
        
        # Gemini API konfigurieren
        genai.configure(api_key=self.api_key)
        
        # Model Configuration
        self.model_name = config.get('model', 'gemini-pro')
        self.temperature = config.get('temperature', 0.3)
        self.max_output_tokens = config.get('max_output_tokens', 1000)
        
        # Rate Limiting (15 requests/minute kostenlos)
        self.min_request_interval = config.get('min_interval_seconds', 4.0)  # 4 Sekunden = 15/min
        self.request_timeout = config.get('timeout', 30)
        
        # Model initialisieren
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    candidate_count=1
                )
            )
            logger.info(f"Gemini Model {self.model_name} initialisiert")
        except Exception as e:
            logger.error(f"Gemini Model Init Error: {e}")
            raise
    
    def analyze_market_data(self, market_data: MarketData, strategy_context: str) -> AISignal:
        """
        Analysiert Marktdaten mit Google Gemini
        """
        start_time = time.time()
        
        try:
            self._rate_limit_check()
            self.request_count += 1
            
            # Analysis Prompt erstellen
            prompt = self._create_analysis_prompt(market_data, strategy_context)
            
            # Gemini Query
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                self.error_count += 1
                return self._create_error_signal("Gemini: Keine Response erhalten")
            
            response_text = response.text.strip()
            
            # Signal aus Response extrahieren
            signal_type, confidence, reasoning = self._parse_gemini_response(response_text)
            
            processing_time = time.time() - start_time
            
            return AISignal(
                provider=self.name,
                signal_type=signal_type,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=time.time(),
                processing_time=processing_time,
                raw_response={"text": response_text, "model": self.model_name}
            )
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"Gemini Analysis Error: {e}")
            return self._create_error_signal(str(e))
    
    def _create_analysis_prompt(self, market_data: MarketData, strategy_context: str) -> str:
        """Erstellt strukturierten Prompt für Gemini"""
        prompt = f"""You are a professional trading analyst. Analyze the following market data and provide a clear trading recommendation.

MARKET DATA:
Symbol: {market_data.symbol}
Current Price: ${market_data.current_price:.2f}
24h Price Change: {market_data.price_change_24h:+.2f}%
Volume: {market_data.volume:,.0f}

TECHNICAL INDICATORS:"""
        
        if market_data.technical_indicators:
            for indicator, value in market_data.technical_indicators.items():
                prompt += f"\n- {indicator}: {value:.2f}"
        
        prompt += f"\n\nSTRATEGY CONTEXT: {strategy_context}"
        
        if market_data.news_sentiment:
            prompt += f"\nNEWS SENTIMENT: {market_data.news_sentiment}"
        
        if market_data.additional_context:
            prompt += "\nADDITIONAL CONTEXT:"
            for key, value in market_data.additional_context.items():
                prompt += f"\n- {key}: {value}"
        
        prompt += """

INSTRUCTIONS:
1. Analyze the technical indicators and market sentiment
2. Consider the strategy context and current market conditions
3. Provide ONE clear recommendation: BUY, SELL, or HOLD
4. Give a confidence level (0-100%)
5. Explain your reasoning in 2-3 sentences

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
RECOMMENDATION: [BUY/SELL/HOLD]
CONFIDENCE: [0-100]%
REASONING: [Your detailed reasoning here]

Remember: This is for educational purposes. Be analytical and objective."""
        
        return prompt
    def _parse_gemini_response(self, response_text: str) -> tuple[SignalType, float, str]:
        """
        Parst strukturierte Gemini Response
        """
        try:
            lines = response_text.strip().split('\n')
            
            recommendation = None
            confidence = 50.0
            reasoning = response_text  # Fallback
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('RECOMMENDATION:'):
                    rec_text = line.replace('RECOMMENDATION:', '').strip().upper()
                    if 'BUY' in rec_text:
                        recommendation = SignalType.BUY
                    elif 'SELL' in rec_text:
                        recommendation = SignalType.SELL
                    else:
                        recommendation = SignalType.HOLD
                
                elif line.startswith('CONFIDENCE:'):
                    conf_text = line.replace('CONFIDENCE:', '').strip()
                    # Extract number from text like "75%" or "75"
                    import re
                    numbers = re.findall(r'(\d+(?:\.\d+)?)', conf_text)
                    if numbers:
                        confidence = min(100.0, max(0.0, float(numbers[0])))
                
                elif line.startswith('REASONING:'):
                    reasoning = line.replace('REASONING:', '').strip()
                    # Add subsequent lines to reasoning
                    line_index = lines.index(line)
                    if line_index + 1 < len(lines):
                        remaining_lines = lines[line_index + 1:]
                        reasoning += ' ' + ' '.join(remaining_lines)
            
            # Fallback wenn Struktur nicht erkannt wurde
            if recommendation is None:
                recommendation, confidence, reasoning = self._extract_signal_from_text(response_text)
            
            # Reasoning auf angemessene Länge kürzen
            if len(reasoning) > 300:
                reasoning = reasoning[:300] + "..."
            
            return recommendation, confidence, reasoning
        
        except Exception as e:
            logger.error(f"Gemini Response Parsing Error: {e}")
            # Fallback auf Text-Analyse
            return self._extract_signal_from_text(response_text)
    
    def test_connection(self) -> bool:
        """Testet Gemini API Verbindung"""
        try:
            test_prompt = "Analyze this simple market scenario: Bitcoin price is $45,000, up 3% today with high volume. What's your trading recommendation? Format: RECOMMENDATION: [BUY/SELL/HOLD], CONFIDENCE: [0-100]%, REASONING: [brief explanation]"
            
            response = self.model.generate_content(test_prompt)
            
            if response and response.text:
                logger.info("Gemini API Test erfolgreich")
                logger.debug(f"Test Response: {response.text[:100]}...")
                return True
            else:
                logger.error("Gemini API Test: Keine Response")
                return False
        
        except Exception as e:
            logger.error(f"Gemini Connection Test Error: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Model Information"""
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_output_tokens': self.max_output_tokens,
            'rate_limit_interval': self.min_request_interval
        }
