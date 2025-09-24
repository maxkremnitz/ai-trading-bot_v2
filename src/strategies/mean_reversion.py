"""
Mean Reversion Strategy  
Demo-Account #2: DAX & GOLD
Bollinger Bands + RSI + RandomForest Classifier
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .base_strategy import BaseStrategy, StrategySignal
from ..ai.base_provider import MarketData, SignalType

logger = logging.getLogger(__name__)

class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy mit ML Classification
    
    Technische Indikatoren:
    - Bollinger Bands (20, 2)
    - RSI (14) für Oversold/Overbought
    - Standard Deviation für Volatilität
    - RandomForest für Trend vs Range Classification
    """
    
    def __init__(self, config: Dict, ai_voting_system, capital_api):
        super().__init__("Mean Reversion", config, ai_voting_system, capital_api)
        
        # Bollinger Bands Parameter
        self.bb_period = config.get('bb_period', 20)
        self.bb_std_dev = config.get('bb_std_dev', 2.0)
        self.rsi_period = config.get('rsi_period', 14)
        
        # Mean Reversion Thresholds
        self.oversold_threshold = config.get('oversold_threshold', 30)
        self.overbought_threshold = config.get('overbought_threshold', 70)
        self.bb_squeeze_threshold = config.get('bb_squeeze_threshold', 0.02)  # 2%
        
        # RandomForest Classifier
        self.rf_classifier = None
        self.scaler = StandardScaler()
        self.rf_features = []
        self.min_training_samples = config.get('min_training_samples', 100)
        
        logger.info(f"Mean Reversion Strategy: BB({self.bb_period}, {self.bb_std_dev}), RSI({self.rsi_period})")
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Berechnet Mean Reversion Indikatoren
        """
        indicators = {}
        
        try:
            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=self.bb_period).mean()
            bb_std = data['Close'].rolling(window=self.bb_period).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * self.bb_std_dev)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * self.bb_std_dev)
            
            # Bollinger Band Position
            data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
            data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss.replace(0, 0.0001)
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Volatility Measures
            data['Price_STD'] = data['Close'].rolling(window=20).std()
            data['Returns'] = data['Close'].pct_change()
            data['Return_STD'] = data['Returns'].rolling(window=20).std()
            
            # Mean Reversion Strength
            data['Distance_from_MA'] = (data['Close'] - data['BB_Middle']) / data['BB_Middle']
            data['MA_Slope'] = data['BB_Middle'].diff(5) / data['BB_Middle'].shift(5)
            
            # Current Values
            current_price = float(data['Close'].iloc[-1])
            bb_upper = float(data['BB_Upper'].iloc[-1])
            bb_middle = float(data['BB_Middle'].iloc[-1])
            bb_lower = float(data['BB_Lower'].iloc[-1])
            bb_position = float(data['BB_Position'].iloc[-1])
            bb_width = float(data['BB_Width'].iloc[-1])
            rsi = float(data['RSI'].iloc[-1])
            price_std = float(data['Price_STD'].iloc[-1])
            distance_from_ma = float(data['Distance_from_MA'].iloc[-1])
            ma_slope = float(data['MA_Slope'].iloc[-1]) if not pd.isna(data['MA_Slope'].iloc[-1]) else 0
            
            # RandomForest Features
            rf_features = self._calculate_rf_features(data)
            
            indicators.update({
                'current_price': current_price,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'bb_position': bb_position,  # 0-1, where 0.5 is middle
                'bb_width': bb_width,
                'rsi': rsi,
                'price_volatility': price_std / current_price if current_price > 0 else 0,
                'distance_from_mean': distance_from_ma * 100,  # Percentage
                'ma_slope': ma_slope * 100,
                **rf_features
            })
            
        except Exception as e:
            logger.error(f"Mean Reversion Indicators Error: {e}")
            indicators = {'current_price': float(data['Close'].iloc[-1]) if len(data) > 0 else 0}
        
        return indicators
    
    def _calculate_rf_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Berechnet Features für RandomForest Classifier
        """
        features = {}
        
        try:
            if len(data) < 50:
                return {
                    'rf_trend_probability': 0.5,
                    'rf_range_probability': 0.5,
                    'market_regime': 0.5,  # 0 = trend, 1 = range
                    'reversion_strength': 0.0
                }
            
            # Feature Engineering für RF
            lookback = min(50, len(data))
            recent_data = data.tail(lookback).copy()
            
            # 1. Price Pattern Features
            high_low_ratio = recent_data['High'].max() / recent_data['Low'].min() if recent_data['Low'].min() > 0 else 1
            price_range_stability = recent_data['Close'].std() / recent_data['Close'].mean() if recent_data['Close'].mean() > 0 else 0
            
            # 2. Trend vs Range Classification Features
            # Linear trend strength
            prices = recent_data['Close'].values
            x = np.arange(len(prices))
            if len(prices) > 1:
                slope, _ = np.polyfit(x, prices, 1)
                trend_strength = abs(slope / prices[-1]) if prices[-1] > 0 else 0
            else:
                trend_strength = 0
            
            # Price Channel Analysis
            rolling_max = recent_data['High'].rolling(window=10).max()
            rolling_min = recent_data['Low'].rolling(window=10).min()
            channel_breaks = ((recent_data['Close'] > rolling_max.shift(1)) | 
                            (recent_data['Close'] < rolling_min.shift(1))).sum()
            
            # 3. Mean Reversion Signals
            bb_touches = ((recent_data['Close'] <= recent_data['BB_Lower']) | 
                         (recent_data['Close'] >= recent_data['BB_Upper'])).sum()
            
            rsi_extremes = ((recent_data['RSI'] <= 30) | (recent_data['RSI'] >= 70)).sum()
            
            # 4. Volatility Regime
            current_vol = recent_data['Returns'].std() * np.sqrt(252)  # Annualized
            long_vol = data['Returns'].tail(200).std() * np.sqrt(252) if len(data) >= 200 else current_vol
            vol_regime = current_vol / long_vol if long_vol > 0 else 1
            
            # SimpleRF Classification (ohne sklearn training für Demo)
            # In Production würdest du hier ein trainiertes Model verwenden
            trend_score = 0
            range_score = 0
            
            # Trend Indicators
            if trend_strength > 0.02:  # Strong trend
                trend_score += 0.3
            if channel_breaks > len(prices) * 0.1:  # Frequent breakouts
                trend_score += 0.2
            if bb_touches < len(prices) * 0.05:  # Few BB touches
                trend_score += 0.2
            
            # Range Indicators  
            if price_range_stability < 0.05:  # Stable range
                range_score += 0.3
            if bb_touches > len(prices) * 0.1:  # Many BB touches
                range_score += 0.3
            if rsi_extremes > len(prices) * 0.1:  # Many RSI extremes
                range_score += 0.2
            
            # Normalisieren
            total_score = trend_score + range_score
            if total_score > 0:
                trend_prob = trend_score / total_score
                range_prob = range_score / total_score
            else:
                trend_prob = 0.5
                range_prob = 0.5
            
            market_regime = range_prob  # 0 = trend, 1 = range
            reversion_strength = range_prob * (bb_touches + rsi_extremes) / len(prices)
            
            features = {
                'rf_trend_probability': round(trend_prob, 3),
                'rf_range_probability': round(range_prob, 3),
                'market_regime': round(market_regime, 3),
                'reversion_strength': round(reversion_strength, 3)
            }
            
        except Exception as e:
            logger.error(f"RF Features Error: {e}")
            features = {
                'rf_trend_probability': 0.5,
                'rf_range_probability': 0.5,
                'market_regime': 0.5,
                'reversion_strength': 0.0
            }
        
        return features
    def generate_base_signal(self, symbol: str, market_data: MarketData,
                           technical_indicators: Dict[str, float]) -> Optional[StrategySignal]:
        """
        Generiert Mean Reversion Signal
        """
        try:
            current_price = technical_indicators.get('current_price', 0)
            if current_price <= 0:
                return None
            
            # Signal Score Berechnung
            signal_score = 50.0  # Neutral start
            
            # 1. Bollinger Band Position (40% Weight)
            bb_position = technical_indicators.get('bb_position', 0.5)
            bb_width = technical_indicators.get('bb_width', 0.05)
            
            # Mean Reversion Signals nur bei ausreichender Volatilität
            if bb_width > self.bb_squeeze_threshold:
                if bb_position <= 0.1:  # Near lower band - Buy signal
                    signal_score += 40 * (1 - bb_position * 10)  # Stronger near 0
                elif bb_position >= 0.9:  # Near upper band - Sell signal
                    signal_score -= 40 * (bb_position - 0.8) * 10
                elif 0.4 <= bb_position <= 0.6:  # Near middle - weak signal
                    signal_score += 5  # Slight bias to mean reversion
            
            # 2. RSI Oversold/Overbought (30% Weight)
            rsi = technical_indicators.get('rsi', 50)
            if rsi <= self.oversold_threshold:  # Oversold - Buy signal
                oversold_strength = (self.oversold_threshold - rsi) / self.oversold_threshold
                signal_score += 30 * oversold_strength
            elif rsi >= self.overbought_threshold:  # Overbought - Sell signal
                overbought_strength = (rsi - self.overbought_threshold) / (100 - self.overbought_threshold)
                signal_score -= 30 * overbought_strength
            
            # 3. RandomForest Market Regime (20% Weight)
            market_regime = technical_indicators.get('market_regime', 0.5)
            reversion_strength = technical_indicators.get('reversion_strength', 0)
            
            if market_regime > 0.6:  # Range-bound market favors mean reversion
                signal_score += 15 * market_regime
                signal_score += 5 * reversion_strength
            elif market_regime < 0.4:  # Trending market - avoid mean reversion
                signal_score -= 10 * (1 - market_regime)
            
            # 4. Distance from Mean (10% Weight)
            distance_from_mean = technical_indicators.get('distance_from_mean', 0)
            ma_slope = technical_indicators.get('ma_slope', 0)
            
            # Strong deviation from mean in sideways market
            if abs(ma_slope) < 0.5:  # Sideways market
                if distance_from_mean < -2:  # Below mean
                    signal_score += 10
                elif distance_from_mean > 2:  # Above mean
                    signal_score -= 10
            
            # Signal Decision
            signal_score = max(0, min(100, signal_score))
            
            if signal_score >= 70:  # Strong Buy (Mean Reversion Up)
                action = SignalType.BUY
                confidence = signal_score
            elif signal_score <= 30:  # Strong Sell (Mean Reversion Down)  
                action = SignalType.SELL
                confidence = 100 - signal_score
            else:  # Hold
                action = SignalType.HOLD
                confidence = abs(signal_score - 50) * 2
            
            # Position Sizing and Risk Management
            position_size = self._calculate_position_size(signal_score, technical_indicators)
            stop_loss = self._calculate_stop_loss(current_price, action, technical_indicators)
            take_profit = self._calculate_take_profit(current_price, action, technical_indicators)
            
            # Reasoning
            reasoning = f"Mean Reversion: BB_Pos({bb_position:.2f}), RSI({rsi:.1f}), "
            reasoning += f"Regime({market_regime:.2f}), Distance({distance_from_mean:.1f}%), Score: {signal_score:.1f}"
            
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
            logger.error(f"Mean Reversion Signal Error {symbol}: {e}")
            return None
    
    def _calculate_position_size(self, signal_score: float, indicators: Dict[str, float]) -> float:
        """Position Size basierend auf Mean Reversion Stärke"""
        base_size = self.max_position_size_percent
        
        # BB Width Adjustment - weniger Size bei niedriger Volatilität
        bb_width = indicators.get('bb_width', 0.05)
        if bb_width < self.bb_squeeze_threshold:  # Low volatility
            base_size *= 0.5
        elif bb_width > 0.08:  # High volatility
            base_size *= 1.3
        
        # Reversion Strength Adjustment
        reversion_strength = indicators.get('reversion_strength', 0)
        if reversion_strength > 0.1:
            base_size *= 1.2
        
        return min(6.0, max(1.0, base_size))
    
    def _calculate_stop_loss(self, price: float, action: SignalType, indicators: Dict[str, float]) -> float:
        """Dynamischer Stop Loss für Mean Reversion"""
        base_sl_percent = self.stop_loss_percent
        
        # BB-basierter Stop
        bb_width = indicators.get('bb_width', 0.05) * 100
        bb_stop = max(1.0, bb_width * 0.7)  # 70% of BB width
        
        # Volatility Adjustment
        volatility = indicators.get('price_volatility', 0.02) * 100
        vol_stop = max(1.5, volatility * 1.5)
        
        # Use wider of the two
        dynamic_sl = max(base_sl_percent, bb_stop, vol_stop)
        dynamic_sl = min(4.0, dynamic_sl)  # Cap at 4%
        
        if action == SignalType.BUY:
            return price * (1 - dynamic_sl / 100)
        else:
            return price * (1 + dynamic_sl / 100)
    
    def _calculate_take_profit(self, price: float, action: SignalType, indicators: Dict[str, float]) -> float:
        """Take Profit für Mean Reversion (meist zur Mitte)"""
        # Target: Bollinger Band Middle
        bb_middle = indicators.get('bb_middle', price)
        
        if action == SignalType.BUY:
            # Target BB middle or above
            distance_to_middle = (bb_middle - price) / price
            if distance_to_middle > 0.005:  # At least 0.5% profit potential
                return bb_middle * 0.98  # Slightly before middle
            else:
                return price * (1 + self.take_profit_percent / 100)
        else:
            # Target BB middle or below
            distance_to_middle = (price - bb_middle) / price  
            if distance_to_middle > 0.005:
                return bb_middle * 1.02  # Slightly above middle
            else:
                return price * (1 - self.take_profit_percent / 100)
    
    def get_strategy_context(self) -> str:
        """Strategy Context für AI Analysis"""
        return f"""Mean Reversion Strategy Analysis:
        - Primary focus: Trading reversals from extreme price levels back to mean
        - Assets: {', '.join(self.assets)} (DAX for index mean reversion, GOLD for commodity cycles)
        - Key indicators: Bollinger Bands position, RSI extremes, volatility regime
        - RandomForest classifier: Identifies whether market is in trending or ranging regime
        - Risk management: Volatility-adjusted stops, target mean reversion to BB middle
        - Objective: Profit from temporary price deviations with statistical edge
        - Best conditions: Range-bound markets, high volatility, clear support/resistance
        
        Please analyze if current price action shows mean reversion characteristics.
        Consider whether the market regime favors range-bound behavior over trending."""
