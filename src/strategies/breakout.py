"""
Breakout Strategy
Demo-Account #3: NASDAQ & TESLA
Donchian Channels + Volume Filter + XGBoost for True Breakout Detection
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

class BreakoutStrategy(BaseStrategy):
    """
    Breakout Strategy mit XGBoost ML Detection
    Technische Indikatoren:
    - Donchian Channels (20 periods)
    - Volume Analysis (relative volume)
    - ATR für Volatility
    - XGBoost Features für True vs False Breakout
    """

    # Handelszeiten für diese Strategie (UTC)
    TRADING_HOURS = {
        '^NDX': {
            'open': dt_time(13, 30),  # 9:30 ET = 13:30 UTC
            'close': dt_time(20, 0),   # 16:00 ET = 20:00 UTC
            'weekdays': [0, 1, 2, 3, 4]
        },
        'TSLA': {
            'open': dt_time(13, 30),  # 9:30 ET = 13:30 UTC
            'close': dt_time(20, 0),   # 16:00 ET = 20:00 UTC
            'weekdays': [0, 1, 2, 3, 4]
        }
    }

    def __init__(self, config: Dict, ai_voting_system, capital_api):
        super().__init__("Breakout", config, ai_voting_system, capital_api)

        # Donchian Channel Parameter
        self.donchian_period = config.get('donchian_period', 20)
        self.volume_lookback = config.get('volume_lookback', 20)
        self.atr_period = config.get('atr_period', 14)

        # Breakout Thresholds
        self.min_volume_multiple = config.get('min_volume_multiple', 1.5)  # 150% of avg volume
        self.min_price_move_percent = config.get('min_price_move_percent', 0.5)  # 0.5%
        self.consolidation_period = config.get('consolidation_period', 10)  # bars

        # XGBoost Features (simplified without actual XGBoost for demo)
        self.xgb_features = []
        self.breakout_confidence_threshold = config.get('xgb_confidence_threshold', 0.7)

        # Überschreibe Asset-Klassifizierung für spezifische Handelszeiten
        self.asset_classes = {
            '^NDX': 'stocks',
            'TSLA': 'stocks'
        }

        logger.info(f"Breakout Strategy: Donchian({self.donchian_period}), "
                   f"Vol({self.volume_lookback}), ATR({self.atr_period})")

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
        Berechnet Breakout-spezifische Indikatoren
        """
        indicators = {}

        try:
            if len(data) < self.donchian_period * 2:
                logger.warning(f"Nicht genug Daten für Donchian Channels (nur {len(data)} Punkte)")
                return {'current_price': float(data['Close'].iloc[-1]) if len(data) > 0 else 0}

            # Donchian Channels
            data['Donchian_High'] = data['High'].rolling(window=self.donchian_period).max()
            data['Donchian_Low'] = data['Low'].rolling(window=self.donchian_period).min()
            data['Donchian_Middle'] = (data['Donchian_High'] + data['Donchian_Low']) / 2

            # Donchian Position
            data['Donchian_Position'] = (data['Close'] - data['Donchian_Low']) / (data['Donchian_High'] - data['Donchian_Low'])
            data['Donchian_Width'] = (data['Donchian_High'] - data['Donchian_Low']) / data['Donchian_Middle']

            # Volume Analysis
            data['Volume_SMA'] = data['Volume'].rolling(window=self.volume_lookback).mean()
            data['Relative_Volume'] = data['Volume'] / data['Volume_SMA']
            data['Volume_Spike'] = (data['Relative_Volume'] > self.min_volume_multiple).astype(int)

            # ATR (Average True Range)
            data['High_Low'] = data['High'] - data['Low']
            data['High_Close'] = abs(data['High'] - data['Close'].shift(1))
            data['Low_Close'] = abs(data['Low'] - data['Close'].shift(1))
            data['True_Range'] = data[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
            data['ATR'] = data['True_Range'].rolling(window=self.atr_period).mean()

            # Breakout Detection
            data['Price_Range'] = data['High'] - data['Low']
            data['Range_Expansion'] = data['Price_Range'] / data['Price_Range'].rolling(window=10).mean()

            # Consolidation Analysis
            data['Price_Volatility'] = data['Close'].rolling(window=self.consolidation_period).std()
            data['Consolidation_Score'] = 1 / (1 + data['Price_Volatility'])  # Higher = more consolidation

            # Current Values
            current_price = float(data['Close'].iloc[-1])
            donchian_high = float(data['Donchian_High'].iloc[-1])
            donchian_low = float(data['Donchian_Low'].iloc[-1])
            donchian_middle = float(data['Donchian_Middle'].iloc[-1])
            donchian_position = float(data['Donchian_Position'].iloc[-1])
            donchian_width = float(data['Donchian_Width'].iloc[-1])
            relative_volume = float(data['Relative_Volume'].iloc[-1])
            atr = float(data['ATR'].iloc[-1])
            range_expansion = float(data['Range_Expansion'].iloc[-1])
            consolidation_score = float(data['Consolidation_Score'].iloc[-1])

            # XGBoost Features
            xgb_features = self._calculate_xgboost_features(data)

            indicators.update({
                'current_price': current_price,
                'donchian_high': donchian_high,
                'donchian_low': donchian_low,
                'donchian_middle': donchian_middle,
                'donchian_position': donchian_position,
                'donchian_width': donchian_width,
                'relative_volume': relative_volume,
                'atr': atr,
                'atr_percent': (atr / current_price * 100) if current_price > 0 else 0,
                'range_expansion': range_expansion,
                'consolidation_score': consolidation_score,
                **xgb_features
            })

        except Exception as e:
            logger.error(f"Breakout Indicators Error: {e}")
            indicators = {'current_price': float(data['Close'].iloc[-1]) if len(data) > 0 else 0}

        return indicators

    def _calculate_xgboost_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Berechnet XGBoost Features für True Breakout Detection
        """
        features = {}

        try:
            if len(data) < 50:
                return {
                    'xgb_breakout_probability': 0.5,
                    'xgb_direction_confidence': 0.5,
                    'consolidation_strength': 0.5,
                    'volume_confirmation': 0.5,
                    'momentum_alignment': 0.5
                }

            # Feature Engineering für XGBoost
            lookback = min(50, len(data))
            recent_data = data.tail(lookback).copy()

            # 1. Consolidation Analysis
            price_std = recent_data['Close'].std()
            price_mean = recent_data['Close'].mean()
            consolidation_strength = 1 - (price_std / price_mean) if price_mean > 0 else 0

            # 2. Volume Pattern Analysis
            volume_trend = recent_data['Relative_Volume'].tail(5).mean()
            volume_acceleration = recent_data['Relative_Volume'].diff().tail(3).mean()
            volume_confirmation = min(1.0, (volume_trend + volume_acceleration) / 2)

            # 3. Price Momentum Features
            short_ma = recent_data['Close'].rolling(window=5).mean()
            long_ma = recent_data['Close'].rolling(window=20).mean()
            momentum_alignment = ((short_ma.iloc[-1] > long_ma.iloc[-1]).astype(int) +
                                (recent_data['Close'].iloc[-1] > recent_data['Close'].iloc[-5]).astype(int)) / 2

            # 4. Breakout Pattern Recognition
            # Check for squeeze before breakout
            bb_width_recent = recent_data['Donchian_Width'].tail(10).mean()
            bb_width_historical = recent_data['Donchian_Width'].mean()
            squeeze_factor = bb_width_recent / bb_width_historical if bb_width_historical > 0 else 1

            # Time since last breakout
            breakout_signals = ((recent_data['Close'] >= recent_data['Donchian_High'].shift(1)) |
                              (recent_data['Close'] <= recent_data['Donchian_Low'].shift(1))).astype(int)
            bars_since_breakout = 0
            for i in range(len(breakout_signals) - 1, -1, -1):
                if breakout_signals.iloc[i] == 1:
                    break
                bars_since_breakout += 1

            breakout_timing_score = min(1.0, bars_since_breakout / 20) if bars_since_breakout > 5 else 0

            # 5. Support/Resistance Strength
            resistance_touches = (recent_data['High'] >= recent_data['Donchian_High'] * 0.99).sum()
            support_touches = (recent_data['Low'] <= recent_data['Donchian_Low'] * 1.01).sum()
            sr_strength = (resistance_touches + support_touches) / len(recent_data)

            # Simplified XGBoost Prediction (In Production: trained model)
            breakout_score = 0

            # Positive breakout indicators
            if consolidation_strength > 0.7:  # Strong consolidation
                breakout_score += 0.25
            if volume_confirmation > 0.6:  # Volume support
                breakout_score += 0.25
            if breakout_timing_score > 0.3:  # Good timing
                breakout_score += 0.2
            if squeeze_factor < 0.8:  # Squeeze pattern
                breakout_score += 0.15
            if sr_strength > 0.1:  # Tested levels
                breakout_score += 0.15

            breakout_probability = min(1.0, max(0.0, breakout_score))

            # Direction confidence based on momentum
            if momentum_alignment > 0.7:
                direction_confidence = 0.8
            elif momentum_alignment < 0.3:
                direction_confidence = 0.2
            else:
                direction_confidence = 0.5

            features = {
                'xgb_breakout_probability': round(breakout_probability, 3),
                'xgb_direction_confidence': round(direction_confidence, 3),
                'consolidation_strength': round(consolidation_strength, 3),
                'volume_confirmation': round(volume_confirmation, 3),
                'momentum_alignment': round(momentum_alignment, 3)
            }

        except Exception as e:
            logger.error(f"XGBoost Features Error: {e}")
            features = {
                'xgb_breakout_probability': 0.5,
                'xgb_direction_confidence': 0.5,
                'consolidation_strength': 0.5,
                'volume_confirmation': 0.5,
                'momentum_alignment': 0.5
            }

        return features

    def generate_base_signal(self, symbol: str, market_data: MarketData,
                           technical_indicators: Dict[str, float]) -> Optional[StrategySignal]:
        """
        Generiert Breakout Signal
        """
        try:
            current_price = technical_indicators.get('current_price', 0)
            if current_price <= 0:
                return None

            # Breakout Detection
            donchian_high = technical_indicators.get('donchian_high', current_price)
            donchian_low = technical_indicators.get('donchian_low', current_price)
            donchian_position = technical_indicators.get('donchian_position', 0.5)

            # Signal Score Berechnung
            signal_score = 50.0  # Neutral start
            action = SignalType.HOLD

            # 1. Donchian Breakout Detection (35% Weight)
            if current_price >= donchian_high * 0.999:  # Bullish Breakout
                breakout_strength = min(1.0, (current_price - donchian_high) / donchian_high * 100)
                signal_score += 35 * (0.5 + breakout_strength)
                action = SignalType.BUY
            elif current_price <= donchian_low * 1.001:  # Bearish Breakout
                breakout_strength = min(1.0, (donchian_low - current_price) / donchian_low * 100)
                signal_score -= 35 * (0.5 + breakout_strength)
                action = SignalType.SELL
            elif donchian_position > 0.8:  # Near upper bound
                signal_score += 10
            elif donchian_position < 0.2:  # Near lower bound
                signal_score -= 10

            # 2. Volume Confirmation (25% Weight)
            relative_volume = technical_indicators.get('relative_volume', 1.0)
            volume_confirmation = technical_indicators.get('volume_confirmation', 0.5)

            if relative_volume >= self.min_volume_multiple:  # Strong volume
                signal_score += 25 * min(1.0, relative_volume / 3.0)  # Cap at 3x volume
            elif relative_volume < 0.8:  # Weak volume
                signal_score -= 15

            if volume_confirmation > 0.6:
                signal_score += 10

            # 3. XGBoost Breakout Probability (25% Weight)
            xgb_probability = technical_indicators.get('xgb_breakout_probability', 0.5)
            direction_confidence = technical_indicators.get('xgb_direction_confidence', 0.5)

            if xgb_probability > self.breakout_confidence_threshold:
                signal_score += 25 * xgb_probability

                # Direction Adjustment
                if action == SignalType.BUY and direction_confidence > 0.6:
                    signal_score += 5
                elif action == SignalType.SELL and direction_confidence < 0.4:
                    signal_score -= 5
            else:
                signal_score -= 10  # Low breakout probability

            # 4. Consolidation and Timing (15% Weight)
            consolidation_strength = technical_indicators.get('consolidation_strength', 0.5)
            range_expansion = technical_indicators.get('range_expansion', 1.0)

            if consolidation_strength > 0.7:  # Strong prior consolidation
                signal_score += 10
            if range_expansion > 1.5:  # Range expansion
                signal_score += 5

            # Adjust score based on action
            if action == SignalType.SELL:
                signal_score = 100 - signal_score

            signal_score = max(0, min(100, signal_score))

            # Final Action Decision
            if signal_score >= 75 and action == SignalType.BUY:
                final_action = SignalType.BUY
                confidence = signal_score
            elif signal_score <= 25 and action == SignalType.SELL:
                final_action = SignalType.SELL
                confidence = 100 - signal_score
            else:
                final_action = SignalType.HOLD
                confidence = abs(signal_score - 50) * 2

            # Position Sizing and Risk Management
            position_size = self._calculate_position_size(signal_score, technical_indicators)
            stop_loss = self._calculate_stop_loss(current_price, final_action, technical_indicators)
            take_profit = self._calculate_take_profit(current_price, final_action, technical_indicators)

            # Reasoning
            reasoning = f"Breakout: Donchian({donchian_position:.2f}), Vol({relative_volume:.1f}x), "
            reasoning += f"XGB({xgb_probability:.2f}), Consol({consolidation_strength:.2f}), Score: {signal_score:.1f}"

            return StrategySignal(
                strategy_name=self.name,
                symbol=symbol,
                action=final_action,
                confidence=confidence,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=reasoning,
                technical_data=technical_indicators
            )

        except Exception as e:
            logger.error(f"Breakout Signal Error {symbol}: {e}")
            return None

    def _calculate_position_size(self, signal_score: float, indicators: Dict[str, float]) -> float:
        """Position Size basierend auf Breakout Stärke"""
        base_size = self.max_position_size_percent

        # XGBoost Confidence Adjustment
        xgb_prob = indicators.get('xgb_breakout_probability', 0.5)
        if xgb_prob > 0.8:  # Very confident breakout
            base_size *= 1.5
        elif xgb_prob < 0.4:  # Low confidence
            base_size *= 0.6

        # Volume Confirmation
        rel_volume = indicators.get('relative_volume', 1.0)
        if rel_volume > 2.0:  # Strong volume
            base_size *= 1.3
        elif rel_volume < 1.0:  # Weak volume
            base_size *= 0.7

        return min(8.0, max(1.0, base_size))

    def _calculate_stop_loss(self, price: float, action: SignalType, indicators: Dict[str, float]) -> float:
        """ATR-basierter Stop Loss für Breakouts"""
        atr_percent = indicators.get('atr_percent', 2.0)
        donchian_high = indicators.get('donchian_high', price)
        donchian_low = indicators.get('donchian_low', price)

        if action == SignalType.BUY:
            # Stop below Donchian Low or ATR-based
            donchian_stop = donchian_low * 0.995  # Small buffer
            atr_stop = price * (1 - max(0.015, atr_percent * 1.5 / 100))  # 1.5x ATR
            return max(donchian_stop, atr_stop)  # Use closer stop
        else:
            # Stop above Donchian High or ATR-based
            donchian_stop = donchian_high * 1.005
            atr_stop = price * (1 + max(0.015, atr_percent * 1.5 / 100))
            return min(donchian_stop, atr_stop)

    def _calculate_take_profit(self, price: float, action: SignalType, indicators: Dict[str, float]) -> float:
        """Take Profit basierend auf ATR und Momentum"""
        atr_percent = indicators.get('atr_percent', 2.0)
        momentum_alignment = indicators.get('momentum_alignment', 0.5)

        # Base target: 2-4x ATR
        base_target = max(3.0, atr_percent * 3.0)

        # Adjust for momentum
        if momentum_alignment > 0.7:  # Strong momentum
            base_target *= 1.5
        elif momentum_alignment < 0.3:  # Weak momentum
            base_target *= 0.8

        base_target = min(8.0, base_target)  # Cap at 8%

        if action == SignalType.BUY:
            return price * (1 + base_target / 100)
        else:
            return price * (1 - base_target / 100)

    def get_strategy_context(self) -> str:
        """Strategy Context für AI Analysis"""
        return f"""Breakout Strategy Analysis:
        - Primary focus: Trading price breakouts from consolidation periods
        - Assets: {', '.join(self.assets)} (NASDAQ for tech momentum, TESLA for volatile breakouts)
        - Key indicators: Donchian Channels, volume confirmation, range expansion
        - XGBoost classifier: Distinguishes true breakouts from false breakouts
        - Risk management: ATR-based stops, momentum-adjusted targets
        - Objective: Capture strong directional moves after consolidation
        - Best conditions: High volume, range expansion, strong momentum alignment
        - Current market hours: {self._get_current_market_status()}

        Please analyze if current price action shows genuine breakout characteristics.
        Consider volume patterns, consolidation quality, and momentum sustainability."""

    def _get_current_market_status(self) -> str:
        """Gibt aktuellen Marktstatus zurück"""
        now_utc = datetime.now(timezone('UTC'))
        status = []

        for asset in self.assets:
            is_open = self._is_market_open(asset)
            status.append(f"{asset}: {'OPEN' if is_open else 'CLOSED'}")

        return ", ".join(status)
