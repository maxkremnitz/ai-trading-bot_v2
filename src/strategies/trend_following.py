"""
Trend Following Strategy
Demo-Account #1: EUR/USD & BTC/USDT
SMA50/SMA200 + MACD + LSTM-Features für AI Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime, time as dt_time
from pytz import timezone

from .base_strategy import BaseStrategy, StrategySignal
from ..ai.base_provider import MarketData, SignalType

logger = logging.getLogger(__name__)

class TrendFollowingStrategy(BaseStrategy):
    """
    Trendfolge-Strategie mit AI Consensus
    Technische Indikatoren:
    - SMA 50/200 (Golden Cross / Death Cross)
    - MACD (12, 26, 9)
    - RSI für Momentum
    - LSTM-Features für AI
    """

    # Handelszeiten für diese Strategie (UTC)
    TRADING_HOURS = {
        'EURUSD=X': {
            'open': dt_time(0, 0),    # 24/5
            'close': dt_time(23, 59),
            'weekdays': [0, 1, 2, 3, 4]
        },
        'BTC-USD': {
            'open': dt_time(0, 0),    # 24/7
            'close': dt_time(23, 59),
            'weekdays': [0, 1, 2, 3, 4, 5, 6]
        }
    }

    def __init__(self, config: Dict, ai_voting_system, capital_api):
        super().__init__("Trend Following", config, ai_voting_system, capital_api)

        # Strategy-spezifische Parameter
        self.sma_short_period = config.get('sma_short', 50)
        self.sma_long_period = config.get('sma_long', 200)
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        self.rsi_period = config.get('rsi_period', 14)

        # Trend Thresholds
        self.strong_trend_threshold = config.get('strong_trend_threshold', 2.0)  # %
        self.trend_confirmation_bars = config.get('trend_confirmation_bars', 3)

        # LSTM Feature Engineering
        self.lstm_lookback = config.get('lstm_lookback', 60)  # 60 Perioden für LSTM

        # Überschreibe Asset-Klassifizierung für spezifische Handelszeiten
        self.asset_classes = {
            'EURUSD=X': 'forex',
            'BTC-USD': 'crypto'
        }

        logger.info(f"Trend Following Strategy: SMA({self.sma_short_period}/{self.sma_long_period}), "
                   f"MACD({self.macd_fast},{self.macd_slow},{self.macd_signal})")

    def _is_market_open(self, asset: str) -> bool:
        """Asset-spezifische Handelszeiten-Prüfung"""
        if asset in self.TRADING_HOURS:
            hours_config = self.TRADING_HOURS[asset]
        else:
            return super()._is_market_open(asset)

        now_utc = datetime.now(timezone('UTC'))
        current_time = now_utc.time()
        current_weekday = now_utc.weekday()

        # Prüfe Wochentag
        if current_weekday not in hours_config['weekdays']:
            return False

        # Prüfe Uhrzeit
        if hours_config['open'] <= hours_config['close']:
            if not (hours_config['open'] <= current_time <= hours_config['close']):
                return False
        else:
            if not (current_time >= hours_config['open'] or current_time <= hours_config['close']):
                return False

        return True

    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Berechnet Trend-Following Indikatoren
        """
        indicators = {}

        try:
            if len(data) < max(self.sma_long_period, self.macd_slow + 10):
                logger.warning(f"Nicht genug Daten für Indikatoren (nur {len(data)} Punkte)")
                return {'current_price': float(data['Close'].iloc[-1]) if len(data) > 0 else 0}

            # Moving Averages
            data[f'SMA_{self.sma_short_period}'] = data['Close'].rolling(window=self.sma_short_period).mean()
            data[f'SMA_{self.sma_long_period}'] = data['Close'].rolling(window=self.sma_long_period).mean()

            # MACD
            ema_fast = data['Close'].ewm(span=self.macd_fast, adjust=False).mean()
            ema_slow = data['Close'].ewm(span=self.macd_slow, adjust=False).mean()
            data['MACD'] = ema_fast - ema_slow
            data['MACD_Signal'] = data['MACD'].ewm(span=self.macd_signal, adjust=False).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

            # RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_period).mean()
            rs = gain / loss.replace(0, 0.0001)
            data['RSI'] = 100 - (100 / (1 + rs))

            # Current values
            current_price = float(data['Close'].iloc[-1])
            sma_short = float(data[f'SMA_{self.sma_short_period}'].iloc[-1])
            sma_long = float(data[f'SMA_{self.sma_long_period}'].iloc[-1])
            macd = float(data['MACD'].iloc[-1])
            macd_signal = float(data['MACD_Signal'].iloc[-1])
            macd_histogram = float(data['MACD_Histogram'].iloc[-1])
            rsi = float(data['RSI'].iloc[-1])

            # Trend Analysis
            trend_strength = ((sma_short - sma_long) / sma_long) * 100 if sma_long > 0 else 0
            price_vs_sma_short = ((current_price - sma_short) / sma_short) * 100 if sma_short > 0 else 0
            price_vs_sma_long = ((current_price - sma_long) / sma_long) * 100 if sma_long > 0 else 0

            # LSTM Features für AI
            lstm_features = self._calculate_lstm_features(data)

            indicators.update({
                'current_price': current_price,
                'sma_50': sma_short,
                'sma_200': sma_long,
                'trend_strength': trend_strength,
                'price_vs_sma_50': price_vs_sma_short,
                'price_vs_sma_200': price_vs_sma_long,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'rsi': rsi,
                **lstm_features
            })

        except Exception as e:
            logger.error(f"Technical Indicators Error: {e}")
            indicators = {'current_price': float(data['Close'].iloc[-1]) if len(data) > 0 else 0}

        return indicators

    def _calculate_lstm_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Berechnet LSTM-Features für AI Analysis
        """
        features = {}

        try:
            if len(data) < self.lstm_lookback:
                return {
                    'lstm_trend_score': 0.0,
                    'lstm_momentum': 0.0,
                    'lstm_volatility': 1.0,
                    'price_sequence_trend': 0.0
                }

            # Price Sequence für LSTM
            prices = data['Close'].tail(self.lstm_lookback).values

            # Trend Score (Linear Regression Slope)
            x = np.arange(len(prices))
            if len(prices) > 1:
                slope, _ = np.polyfit(x, prices, 1)
                trend_score = (slope / prices[-1]) * 100 if prices[-1] > 0 else 0
            else:
                trend_score = 0

            # Momentum (Price acceleration)
            if len(prices) >= 3:
                recent_change = (prices[-1] - prices[-3]) / prices[-3] * 100 if prices[-3] > 0 else 0
                earlier_change = (prices[-3] - prices[-6]) / prices[-6] * 100 if len(prices) >= 6 and prices[-6] > 0 else 0
                momentum = recent_change - earlier_change
            else:
                momentum = 0

            # Volatility (Rolling Standard Deviation)
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * 100 if len(returns) > 0 else 1.0

            # Price Pattern Recognition (simplified)
            if len(prices) >= 10:
                recent_high = np.max(prices[-10:])
                recent_low = np.min(prices[-10:])
                current_position = (prices[-1] - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
                pattern_trend = (current_position - 0.5) * 2  # -1 to 1
            else:
                pattern_trend = 0

            features = {
                'lstm_trend_score': round(trend_score, 3),
                'lstm_momentum': round(momentum, 3),
                'lstm_volatility': round(volatility, 3),
                'price_sequence_trend': round(pattern_trend, 3)
            }

        except Exception as e:
            logger.error(f"LSTM Features Error: {e}")
            features = {
                'lstm_trend_score': 0.0,
                'lstm_momentum': 0.0,
                'lstm_volatility': 1.0,
                'price_sequence_trend': 0.0
            }

        return features

    def generate_base_signal(self, symbol: str, market_data: MarketData,
                           technical_indicators: Dict[str, float]) -> Optional[StrategySignal]:
        """
        Generiert Trend-Following Signal
        """
        try:
            current_price = technical_indicators.get('current_price', 0)
            if current_price <= 0:
                return None

            # Signal Score Berechnung
            signal_score = 50.0  # Neutral start

            # 1. Golden Cross / Death Cross (35% Weight)
            sma_50 = technical_indicators.get('sma_50', 0)
            sma_200 = technical_indicators.get('sma_200', 0)
            trend_strength = technical_indicators.get('trend_strength', 0)

            if sma_50 > sma_200 and abs(trend_strength) > 0.5:  # Golden Cross
                signal_score += 35 * min(1.0, abs(trend_strength) / self.strong_trend_threshold)
            elif sma_50 < sma_200 and abs(trend_strength) > 0.5:  # Death Cross
                signal_score -= 35 * min(1.0, abs(trend_strength) / self.strong_trend_threshold)

            # 2. MACD Confirmation (25% Weight)
            macd = technical_indicators.get('macd', 0)
            macd_signal = technical_indicators.get('macd_signal', 0)
            macd_histogram = technical_indicators.get('macd_histogram', 0)

            if macd > macd_signal and macd_histogram > 0:  # Bullish MACD
                signal_score += 25
            elif macd < macd_signal and macd_histogram < 0:  # Bearish MACD
                signal_score -= 25

            # 3. RSI Momentum (20% Weight)
            rsi = technical_indicators.get('rsi', 50)
            if 40 <= rsi <= 60:  # Neutral zone, good for trend following
                signal_score += 10
            elif rsi < 30:  # Oversold, potential reversal
                signal_score += 15
            elif rsi > 70:  # Overbought, potential reversal
                signal_score -= 15

            # 4. LSTM Features (20% Weight)
            lstm_trend = technical_indicators.get('lstm_trend_score', 0)
            lstm_momentum = technical_indicators.get('lstm_momentum', 0)

            if lstm_trend > 0.1:  # Positive LSTM trend
                signal_score += 15
            elif lstm_trend < -0.1:  # Negative LSTM trend
                signal_score -= 15

            if lstm_momentum > 0.05:  # Positive momentum
                signal_score += 5
            elif lstm_momentum < -0.05:  # Negative momentum
                signal_score -= 5

            # Signal Decision
            signal_score = max(0, min(100, signal_score))

            if signal_score >= 65:  # Strong Buy
                action = SignalType.BUY
                confidence = signal_score
            elif signal_score <= 35:  # Strong Sell
                action = SignalType.SELL
                confidence = 100 - signal_score
            else:  # Hold
                action = SignalType.HOLD
                confidence = abs(signal_score - 50) * 2

            # Position Sizing und Risk Management
            position_size = self._calculate_position_size(signal_score, technical_indicators)
            stop_loss = self._calculate_stop_loss(current_price, action, technical_indicators)
            take_profit = self._calculate_take_profit(current_price, action, technical_indicators)

            # Reasoning
            reasoning = f"Trend Following: SMA({sma_50:.2f}/{sma_200:.2f}), "
            reasoning += f"MACD({macd:.3f}), RSI({rsi:.1f}), "
            reasoning += f"LSTM Trend({lstm_trend:.3f}), Score: {signal_score:.1f}"

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
            logger.error(f"Trend Following Signal Error {symbol}: {e}")
            return None

    def _calculate_position_size(self, signal_score: float, indicators: Dict[str, float]) -> float:
        """Berechnet Position Size basierend auf Signal Strength"""
        base_size = self.max_position_size_percent

        # Volatility Adjustment
        volatility = indicators.get('lstm_volatility', 1.0)
        if volatility > 3.0:  # High volatility
            base_size *= 0.7
        elif volatility < 1.0:  # Low volatility
            base_size *= 1.2

        # Confidence Adjustment
        if signal_score > 80 or signal_score < 20:  # Very strong signal
            base_size *= 1.3
        elif 45 <= signal_score <= 55:  # Weak signal
            base_size *= 0.6

        return min(8.0, max(1.0, base_size))  # Cap zwischen 1-8%

    def _calculate_stop_loss(self, price: float, action: SignalType, indicators: Dict[str, float]) -> float:
        """Berechnet dynamischen Stop Loss"""
        base_sl_percent = self.stop_loss_percent

        # ATR-basierte Anpassung (simplified)
        volatility = indicators.get('lstm_volatility', 1.0)
        dynamic_sl = base_sl_percent * (1 + volatility / 5.0)  # Adjust for volatility

        # SMA Support/Resistance
        sma_50 = indicators.get('sma_50', price)
        sma_distance = abs(price - sma_50) / price * 100
        if sma_distance > 2.0:  # Far from SMA, wider stop
            dynamic_sl *= 1.2

        dynamic_sl = min(5.0, max(1.0, dynamic_sl))  # Cap zwischen 1-5%

        if action == SignalType.BUY:
            return price * (1 - dynamic_sl / 100)
        else:
            return price * (1 + dynamic_sl / 100)

    def _calculate_take_profit(self, price: float, action: SignalType, indicators: Dict[str, float]) -> float:
        """Berechnet Take Profit Target"""
        base_tp_percent = self.take_profit_percent

        # Trend Strength Adjustment
        trend_strength = abs(indicators.get('trend_strength', 0))
        if trend_strength > 2.0:  # Strong trend
            base_tp_percent *= 1.5
        elif trend_strength < 0.5:  # Weak trend
            base_tp_percent *= 0.8

        base_tp_percent = min(8.0, max(1.5, base_tp_percent))  # Cap zwischen 1.5-8%

        if action == SignalType.BUY:
            return price * (1 + base_tp_percent / 100)
        else:
            return price * (1 - base_tp_percent / 100)

    def get_strategy_context(self) -> str:
        """Strategy Context für AI Analysis"""
        return f"""Trend Following Strategy Analysis:
        - Primary focus: Long-term trend identification using SMA crossovers
        - Assets: {', '.join(self.assets)} (EUR/USD for forex trends, BTC/USDT for crypto trends)
        - Key indicators: SMA 50/200 crossover, MACD confirmation, RSI momentum
        - LSTM features: Price sequence analysis, trend scoring, momentum detection
        - Risk management: Dynamic stops based on volatility, trend strength position sizing
        - Objective: Capture sustained directional moves with AI confirmation
        - Timeframe: Medium to long-term positions (hours to days)
        - Current market hours: {self._get_current_market_status()}

        Please analyze if current market conditions favor trend continuation or reversal.
        Consider the LSTM trend features and overall momentum for directional bias."""

    def _get_current_market_status(self) -> str:
        """Gibt aktuellen Marktstatus zurück"""
        now_utc = datetime.now(timezone('UTC'))
        status = []

        for asset in self.assets:
            is_open = self._is_market_open(asset)
            status.append(f"{asset}: {'OPEN' if is_open else 'CLOSED'}")

        return ", ".join(status)
