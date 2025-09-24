"""
News/Event Trading Strategy
Demo-Account #4: EUR/USD & WTI Oil
Economic Calendar + News Sentiment + NLP Analysis (BERT/GPT Features)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
import requests
from datetime import datetime, timedelta
import re

from .base_strategy import BaseStrategy, StrategySignal
from ..ai.base_provider import MarketData, SignalType

logger = logging.getLogger(__name__)

class NewsTradingStrategy(BaseStrategy):
    """
    News/Event Trading Strategy mit NLP Sentiment
    
    Features:
    - Economic Calendar Events
    - News Sentiment Analysis
    - Market Impact Scoring
    - BERT/GPT-style Features für AI
    """
    
    def __init__(self, config: Dict, ai_voting_system, capital_api):
        super().__init__("News Trading", config, ai_voting_system, capital_api)
        
        # News Sources Configuration
        self.news_sources = config.get('news_sources', ['reuters', 'bloomberg', 'cnbc'])
        self.economic_calendar_enabled = config.get('economic_calendar', True)
        self.sentiment_lookback_hours = config.get('sentiment_lookback_hours', 24)
        
        # Impact Thresholds
        self.high_impact_threshold = config.get('high_impact_threshold', 0.7)
        self.news_relevance_threshold = config.get('news_relevance_threshold', 0.6)
        self.sentiment_change_threshold = config.get('sentiment_change_threshold', 0.3)
        
        # NLP Features
        self.nlp_keywords = {
            'bullish': ['rally', 'surge', 'breakout', 'bullish', 'upward', 'gains', 'positive', 'strong', 'recovery'],
            'bearish': ['crash', 'plunge', 'bearish', 'downward', 'losses', 'negative', 'weak', 'decline', 'fall'],
            'volatility': ['volatile', 'uncertainty', 'risk', 'concern', 'worry', 'fear', 'panic', 'shock'],
            'economic': ['gdp', 'inflation', 'employment', 'jobs', 'fed', 'central bank', 'interest rates', 'policy']
        }
        
        # Asset-specific news mapping
        self.asset_news_mapping = {
            'EURUSD': ['euro', 'dollar', 'ecb', 'fed', 'europe', 'us economy'],
            'OIL_CRUDE': ['oil', 'crude', 'opec', 'energy', 'petroleum', 'barrel', 'wti']
        }
        
        # Cache for news data
        self.news_cache = {}
        self.economic_events_cache = {}
        
        logger.info(f"News Trading Strategy: Sources({len(self.news_sources)}), Lookback({self.sentiment_lookback_hours}h)")
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Berechnet News/Event-spezifische Indikatoren
        """
        indicators = {}
        
        try:
            # Basic Technical Indicators für Context
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['Volatility'] = data['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            
            # Current Values
            current_price = float(data['Close'].iloc[-1])
            sma_20 = float(data['SMA_20'].iloc[-1])
            volatility = float(data['Volatility'].iloc[-1])
            
            # Price vs SMA
            price_vs_sma = ((current_price - sma_20) / sma_20 * 100) if sma_20 > 0 else 0
            
            # News/Event Features
            news_features = self._calculate_news_features()
            sentiment_features = self._calculate_sentiment_features()
            event_features = self._calculate_economic_event_features()
            
            indicators.update({
                'current_price': current_price,
                'sma_20': sma_20,
                'price_vs_sma': price_vs_sma,
                'volatility': volatility * 100,  # Percentage
                **news_features,
                **sentiment_features,
                **event_features
            })
            
        except Exception as e:
            logger.error(f"News Trading Indicators Error: {e}")
            indicators = {'current_price': float(data['Close'].iloc[-1]) if len(data) > 0 else 0}
        
        return indicators
    
    def _calculate_news_features(self) -> Dict[str, float]:
        """
        Berechnet News-basierte Features
        """
        features = {}
        
        try:
            # Simulierte News Data (In Production: echte News APIs)
            current_time = datetime.now()
            
            # News Volume (Anzahl relevanter News)
            news_volume = np.random.poisson(5)  # Average 5 news per period
            
            # News Recency Score (mehr recent = höher score)
            recency_scores = []
            for i in range(news_volume):
                hours_ago = np.random.exponential(6)  # Average 6 hours ago
                recency_score = max(0, 1 - hours_ago / self.sentiment_lookback_hours)
                recency_scores.append(recency_score)
            
            avg_news_recency = np.mean(recency_scores) if recency_scores else 0
            
            # News Relevance (asset-specific keywords)
            relevance_scores = np.random.beta(2, 3, news_volume)  # Skewed towards lower relevance
            avg_news_relevance = np.mean(relevance_scores) if len(relevance_scores) > 0 else 0.3
            
            # Breaking News Impact
            breaking_news_count = np.random.binomial(news_volume, 0.1)  # 10% chance of breaking news
            breaking_news_impact = min(1.0, breaking_news_count * 0.3)
            
            features = {
                'news_volume': news_volume,
                'news_recency_score': round(avg_news_recency, 3),
                'news_relevance_score': round(avg_news_relevance, 3),
                'breaking_news_impact': round(breaking_news_impact, 3)
            }
            
        except Exception as e:
            logger.error(f"News Features Error: {e}")
            features = {
                'news_volume': 0,
                'news_recency_score': 0.0,
                'news_relevance_score': 0.0,
                'breaking_news_impact': 0.0
            }
        
        return features
    
    def _calculate_sentiment_features(self) -> Dict[str, float]:
        """
        Berechnet NLP Sentiment Features (BERT/GPT-style)
        """
        features = {}
        
        try:
            # Simulierte Sentiment Analysis
            # In Production: Verwende echte NLP Models oder APIs
            
            # Overall Market Sentiment
            sentiment_scores = np.random.normal(0.5, 0.2, 10)  # 10 recent news items
            sentiment_scores = np.clip(sentiment_scores, 0, 1)
            
            overall_sentiment = np.mean(sentiment_scores)
            sentiment_volatility = np.std(sentiment_scores)
            
            # Sentiment Trend (change over time)
            recent_sentiment = np.mean(sentiment_scores[-5:]) if len(sentiment_scores) >= 5 else overall_sentiment
            older_sentiment = np.mean(sentiment_scores[:5]) if len(sentiment_scores) >= 5 else overall_sentiment
            sentiment_trend = recent_sentiment - older_sentiment
            
            # Asset-specific Sentiment
            asset_specific_sentiment = np.random.beta(3, 3)  # More centered distribution
            
            # Economic Policy Sentiment
            policy_sentiment = np.random.normal(0.5, 0.15)
            policy_sentiment = np.clip(policy_sentiment, 0, 1)
            
            # Fear/Greed Index (simplified)
            fear_greed_score = 1 - sentiment_volatility  # High volatility = more fear
            
            features = {
                'overall_sentiment': round(overall_sentiment, 3),
                'sentiment_volatility': round(sentiment_volatility, 3),
                'sentiment_trend': round(sentiment_trend, 3),
                'asset_sentiment': round(asset_specific_sentiment, 3),
                'policy_sentiment': round(policy_sentiment, 3),
                'fear_greed_index': round(fear_greed_score, 3)
            }
            
        except Exception as e:
            logger.error(f"Sentiment Features Error: {e}")
            features = {
                'overall_sentiment': 0.5,
                'sentiment_volatility': 0.1,
                'sentiment_trend': 0.0,
                'asset_sentiment': 0.5,
                'policy_sentiment': 0.5,
                'fear_greed_index': 0.5
            }
        
        return features
    
    def _calculate_economic_event_features(self) -> Dict[str, float]:
        """
        Berechnet Economic Calendar Features
        """
        features = {}
        
        try:
            current_time = datetime.now()
            
            # Upcoming High-Impact Events (next 24h)
            upcoming_events = []
            for i in range(np.random.poisson(3)):  # Average 3 events
                hours_ahead = np.random.exponential(12)  # Average 12 hours ahead
                if hours_ahead <= 24:  # Only next 24 hours
                    impact_level = np.random.choice([0.3, 0.6, 0.9], p=[0.5, 0.3, 0.2])  # Low, Med, High
                    upcoming_events.append({
                        'hours_ahead': hours_ahead,
                        'impact': impact_level
                    })
            
            # Recent Events Impact (last 24h)
            recent_events = []
            for i in range(np.random.poisson(2)):  # Average 2 recent events
                hours_ago = np.random.exponential(8)  # Average 8 hours ago
                if hours_ago <= 24:
                    impact_level = np.random.choice([0.3, 0.6, 0.9], p=[0.4, 0.4, 0.2])
                    actual_vs_forecast = np.random.normal(0, 0.3)  # Surprise factor
                    recent_events.append({
                        'hours_ago': hours_ago,
                        'impact': impact_level,
                        'surprise': actual_vs_forecast
                    })
            
            # Event Impact Scores
            upcoming_impact = np.mean([e['impact'] for e in upcoming_events]) if upcoming_events else 0
            recent_impact = np.mean([e['impact'] for e in recent_events]) if recent_events else 0
            
            # Surprise Index (how much recent events deviated from expectations)
            surprise_index = np.mean([abs(e['surprise']) for e in recent_events]) if recent_events else 0
            
            # High Impact Event Proximity (closer events = higher score)
            high_impact_proximity = 0
            for event in upcoming_events:
                if event['impact'] > 0.7:  # High impact
                    proximity_score = max(0, 1 - event['hours_ahead'] / 24)
                    high_impact_proximity = max(high_impact_proximity, proximity_score)
            
            # Central Bank Events (special category)
            cb_event_probability = 0.1 if np.random.random() < 0.1 else 0  # 10% chance
            
            features = {
                'upcoming_event_impact': round(upcoming_impact, 3),
                'recent_event_impact': round(recent_impact, 3),
                'economic_surprise_index': round(surprise_index, 3),
                'high_impact_proximity': round(high_impact_proximity, 3),
                'central_bank_event': round(cb_event_probability, 3)
            }
            
        except Exception as e:
            logger.error(f"Economic Event Features Error: {e}")
            features = {
                'upcoming_event_impact': 0.0,
                'recent_event_impact': 0.0,
                'economic_surprise_index': 0.0,
                'high_impact_proximity': 0.0,
                'central_bank_event': 0.0
            }
        
        return features
    def generate_base_signal(self, symbol: str, market_data: MarketData,
                           technical_indicators: Dict[str, float]) -> Optional[StrategySignal]:
        """
        Generiert News/Event Trading Signal
        """
        try:
            current_price = technical_indicators.get('current_price', 0)
            if current_price <= 0:
                return None
            
            # Signal Score Berechnung
            signal_score = 50.0  # Neutral start
            
            # 1. Overall Sentiment Impact (30% Weight)
            overall_sentiment = technical_indicators.get('overall_sentiment', 0.5)
            sentiment_trend = technical_indicators.get('sentiment_trend', 0.0)
            
            # Strong positive sentiment
            if overall_sentiment > 0.65:
                signal_score += 30 * (overall_sentiment - 0.5) * 2
            elif overall_sentiment < 0.35:  # Strong negative sentiment
                signal_score -= 30 * (0.5 - overall_sentiment) * 2
            
            # Sentiment trend momentum
            if abs(sentiment_trend) > self.sentiment_change_threshold:
                if sentiment_trend > 0:  # Improving sentiment
                    signal_score += 15
                else:  # Deteriorating sentiment
                    signal_score -= 15
            
            # 2. Economic Events Impact (25% Weight)
            upcoming_impact = technical_indicators.get('upcoming_event_impact', 0)
            recent_impact = technical_indicators.get('recent_event_impact', 0)
            surprise_index = technical_indicators.get('economic_surprise_index', 0)
            high_impact_proximity = technical_indicators.get('high_impact_proximity', 0)
            
            # Pre-event positioning
            if upcoming_impact > self.high_impact_threshold and high_impact_proximity > 0.5:
                # High volatility expected - reduce position or avoid
                signal_score -= 10
            
            # Post-event reaction
            if recent_impact > 0.6:
                if surprise_index > 0.3:  # Big surprise
                    # Direction depends on sentiment
                    if overall_sentiment > 0.6:
                        signal_score += 20
                    elif overall_sentiment < 0.4:
                        signal_score -= 20
            
            # Central Bank events
            cb_event = technical_indicators.get('central_bank_event', 0)
            if cb_event > 0:
                signal_score += cb_event * 15  # Boost for CB events
            
            # 3. News Flow and Relevance (25% Weight)
            news_volume = technical_indicators.get('news_volume', 0)
            news_relevance = technical_indicators.get('news_relevance_score', 0)
            breaking_news = technical_indicators.get('breaking_news_impact', 0)
            
            if news_volume > 8:  # High news flow
                if news_relevance > self.news_relevance_threshold:
                    signal_score += 15 * news_relevance
            
            if breaking_news > 0.3:  # Breaking news impact
                signal_score += 10 * breaking_news
            
            # 4. Asset-Specific Sentiment (20% Weight)
            asset_sentiment = technical_indicators.get('asset_sentiment', 0.5)
            fear_greed = technical_indicators.get('fear_greed_index', 0.5)
            
            if asset_sentiment > 0.7:  # Strong positive asset sentiment
                signal_score += 15
            elif asset_sentiment < 0.3:  # Strong negative asset sentiment
                signal_score -= 15
            
            # Fear/Greed adjustment
            if fear_greed < 0.3:  # High fear - contrarian signal
                signal_score += 10
            elif fear_greed > 0.7:  # High greed - contrarian signal
                signal_score -= 10
            
            # Signal Decision
            signal_score = max(0, min(100, signal_score))
            
            if signal_score >= 70:  # Strong Buy based on news
                action = SignalType.BUY
                confidence = signal_score
            elif signal_score <= 30:  # Strong Sell based on news
                action = SignalType.SELL
                confidence = 100 - signal_score
            else:  # Hold - unclear news impact
                action = SignalType.HOLD
                confidence = abs(signal_score - 50) * 2
            
            # Position Sizing and Risk Management
            position_size = self._calculate_position_size(signal_score, technical_indicators)
            stop_loss = self._calculate_stop_loss(current_price, action, technical_indicators)
            take_profit = self._calculate_take_profit(current_price, action, technical_indicators)
            
            # Reasoning
            reasoning = f"News Trading: Sentiment({overall_sentiment:.2f}), "
            reasoning += f"Events({upcoming_impact:.2f}/{recent_impact:.2f}), "
            reasoning += f"News({news_volume}vol/{news_relevance:.2f}rel), Score: {signal_score:.1f}"
            
            return StrategySignal(
                strategy_name=self.name,
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=reasoning,
                technical_data=technical_indicators
            )
            
        except Exception as e:
            logger.error(f"News Trading Signal Error {symbol}: {e}")
            return None
    
    def _calculate_position_size(self, signal_score: float, indicators: Dict[str, float]) -> float:
        """Position Size basierend auf News Impact und Confidence"""
        base_size = self.max_position_size_percent
        
        # High Impact Event Adjustment
        high_impact_proximity = indicators.get('high_impact_proximity', 0)
        if high_impact_proximity > 0.7:  # Very close to high impact event
            base_size *= 0.5  # Reduce size due to uncertainty
        
        # Breaking News Boost
        breaking_news = indicators.get('breaking_news_impact', 0)
        if breaking_news > 0.5:
            base_size *= 1.2
        
        # Sentiment Volatility Adjustment
        sentiment_vol = indicators.get('sentiment_volatility', 0.1)
        if sentiment_vol > 0.3:  # High sentiment volatility
            base_size *= 0.8
        
        return min(5.0, max(0.5, base_size))  # News trading: smaller positions
    
    def _calculate_stop_loss(self, price: float, action: SignalType, indicators: Dict[str, float]) -> float:
        """News-Event angepasster Stop Loss"""
        base_sl_percent = self.stop_loss_percent
        
        # High volatility events - wider stops
        upcoming_impact = indicators.get('upcoming_event_impact', 0)
        if upcoming_impact > 0.7:
            base_sl_percent *= 1.5
        
        # Recent surprise factor
        surprise_index = indicators.get('economic_surprise_index', 0)
        if surprise_index > 0.4:
            base_sl_percent *= 1.3
        
        # Market volatility
        volatility = indicators.get('volatility', 2.0) / 100
        vol_adjusted_sl = max(base_sl_percent, volatility * 2)
        
        final_sl = min(4.0, max(1.0, vol_adjusted_sl))  # Cap between 1-4%
        
        if action == SignalType.BUY:
            return price * (1 - final_sl / 100)
        else:
            return price * (1 + final_sl / 100)
    
    def _calculate_take_profit(self, price: float, action: SignalType, indicators: Dict[str, float]) -> float:
        """Take Profit für News Events"""
        base_tp_percent = self.take_profit_percent
        
        # Strong sentiment trend - extend targets
        sentiment_trend = abs(indicators.get('sentiment_trend', 0))
        if sentiment_trend > 0.4:
            base_tp_percent *= 1.5
        
        # Breaking news - quick profits
        breaking_news = indicators.get('breaking_news_impact', 0)
        if breaking_news > 0.3:
            base_tp_percent *= 0.8  # Take profits quicker
        
        final_tp = min(6.0, max(1.5, base_tp_percent))
        
        if action == SignalType.BUY:
            return price * (1 + final_tp / 100)
        else:
            return price * (1 - final_tp / 100)
    
    def get_strategy_context(self) -> str:
        """Strategy Context für AI Analysis"""
        return f"""News/Event Trading Strategy Analysis:
        - Primary focus: Trading market reactions to news events and sentiment shifts
        - Assets: {', '.join(self.assets)} (EUR/USD for forex/policy news, WTI for energy/economic news)
        - Key factors: Economic calendar events, news sentiment analysis, market impact scoring
        - NLP features: Sentiment trends, breaking news impact, economic surprises
        - Risk management: Event-aware position sizing, volatility-adjusted stops
        - Objective: Capture immediate market reactions to fundamental developments
        - Best conditions: High-impact events, clear sentiment shifts, breaking news
        
        Please analyze if current news flow and sentiment justify directional bias.
        Consider economic event timing, sentiment momentum, and market surprise factors."""
