"""
Helper Functions für Trading AI System
Memory Management, Safe Operations und Utilities
"""

import gc
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import wraps
from typing import Any, Optional, Union
import time
import traceback

logger = logging.getLogger(__name__)

def safe_execute(func):
    """Decorator für sichere Funktionsausführung mit Error Handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    return wrapper

def safe_float(value: Any, default: float = 0.0) -> float:
    """Sichere Float-Konvertierung mit Fallback"""
    try:
        if value is None or value == "" or pd.isna(value):
            return default
        if isinstance(value, (pd.Series, np.ndarray)):
            return default
        if isinstance(value, str):
            # Remove common non-numeric characters
            cleaned = value.replace(',', '').replace('$', '').replace('%', '').strip()
            if not cleaned:
                return default
            return float(cleaned)
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return default

def safe_round(value: Any, decimals: int = 2) -> float:
    """Sichere Round-Funktion mit Error Handling"""
    try:
        return round(safe_float(value), decimals)
    except (ValueError, TypeError):
        return 0.0

def safe_percentage(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    """Sichere Prozentberechnung"""
    try:
        num = safe_float(numerator)
        den = safe_float(denominator)
        if den == 0:
            return default
        return round((num / den) * 100, 2)
    except:
        return default

def cleanup_memory():
    """Aggressive Memory Cleanup für Render Environment"""
    try:
        # Standard Garbage Collection
        collected = gc.collect()
        
        # Matplotlib cleanup
        if 'matplotlib.pyplot' in sys.modules:
            plt.close('all')
            plt.clf()
            plt.cla()
        
        # Clear module cache (selective)
        modules_to_clear = ['yfinance', 'pandas', 'numpy']
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                if hasattr(module, '__dict__'):
                    # Clear specific caches
                    if module_name == 'yfinance' and hasattr(module, '_CACHE'):
                        module._CACHE.clear()
        
        logger.debug(f"Memory cleanup: {collected} objects collected")
        return True
        
    except Exception as e:
        logger.error(f"Memory cleanup error: {e}")
        return False

def format_currency(value: Union[float, int], currency: str = "USD") -> str:
    """Formatiert Währungswerte"""
    try:
        val = safe_float(value)
        if currency == "USD":
            return f"${val:,.2f}"
        elif currency == "EUR":
            return f"€{val:,.2f}"
        else:
            return f"{val:,.2f} {currency}"
    except:
        return f"0.00 {currency}"

def format_percentage(value: Union[float, int], include_sign: bool = True) -> str:
    """Formatiert Prozentuale Werte"""
    try:
        val = safe_float(value)
        if include_sign:
            return f"{val:+.1f}%"
        else:
            return f"{val:.1f}%"
    except:
        return "0.0%"

def format_time_ago(timestamp: float) -> str:
    """Formatiert Zeit als 'X ago' String"""
    try:
        now = time.time()
        diff = now - timestamp
        
        if diff < 60:
            return f"{int(diff)}s ago"
        elif diff < 3600:
            return f"{int(diff/60)}m ago"
        elif diff < 86400:
            return f"{int(diff/3600)}h ago"
        else:
            return f"{int(diff/86400)}d ago"
    except:
        return "unknown"

def validate_signal_data(signal_data: dict) -> bool:
    """Validiert Trading Signal Data"""
    required_fields = ['symbol', 'action', 'price', 'confidence']
    
    try:
        for field in required_fields:
            if field not in signal_data:
                return False
        
        # Validate ranges
        if not 0 <= signal_data.get('confidence', 0) <= 100:
            return False
        
        if signal_data.get('price', 0) <= 0:
            return False
        
        if signal_data.get('action') not in ['BUY', 'SELL', 'HOLD']:
            return False
        
        return True
    except:
        return False

@safe_execute
def calculate_win_rate(trades: list) -> float:
    """Berechnet Win Rate aus Trade List"""
    if not trades:
        return 0.0
    
    profitable_trades = sum(1 for trade in trades if safe_float(trade.get('pnl', 0)) > 0)
    return safe_percentage(profitable_trades, len(trades))

@safe_execute
def calculate_total_pnl(trades: list) -> float:
    """Berechnet Total PnL aus Trade List"""
    return sum(safe_float(trade.get('pnl', 0)) for trade in trades)

@safe_execute
def get_system_memory_usage() -> dict:
    """System Memory Usage (für Monitoring)"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except ImportError:
        # psutil not available (Render Environment)
        return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}

def rate_limit_decorator(calls_per_minute: int = 60):
    """Rate Limiting Decorator"""
    def decorator(func):
        last_called = [0.0]
        call_times = []
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove old calls outside the minute window
            call_times[:] = [t for t in call_times if now - t < 60]
            
            if len(call_times) >= calls_per_minute:
                sleep_time = 60 - (now - call_times[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limiting {func.__name__}: sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)
            
            call_times.append(now)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

class PerformanceTimer:
    """Context Manager für Performance Timing"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = time.time() - self.start_time
            logger.debug(f"{self.operation_name} took {elapsed:.3f}s")

def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate String mit Suffix"""
    if not isinstance(text, str):
        text = str(text)
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def health_check_status() -> dict:
    """System Health Check Status"""
    return {
        'timestamp': time.time(),
        'memory_mb': get_system_memory_usage().get('rss_mb', 0),
        'gc_objects': len(gc.get_objects()),
        'modules_loaded': len(sys.modules)
    }
