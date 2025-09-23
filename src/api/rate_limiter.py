"""
Rate Limiter für API Calls
Optimiert für Capital.com API Limits und AI Provider
"""

import time
import threading
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Rate Limit Konfiguration"""
    max_requests: int
    time_window: float  # in Sekunden
    name: str

class RateLimiter:
    """
    Multi-Provider Rate Limiter
    Verwaltet Rate Limits für verschiedene APIs
    """
    
    def __init__(self):
        self.request_history: Dict[str, List[float]] = {}
        self.lock = threading.Lock()
        
        # Rate Limit Konfigurationen
        self.configs = {
            'capital_trading': RateLimitConfig(1, 0.1, 'Capital Trading'),     # 10/sec max
            'capital_general': RateLimitConfig(10, 1.0, 'Capital General'),    # 10/sec
            'capital_session': RateLimitConfig(1, 1.0, 'Capital Session'),     # 1/sec
            'huggingface': RateLimitConfig(30, 3600.0, 'HuggingFace'),        # 30/hour
            'gemini': RateLimitConfig(15, 60.0, 'Gemini'),                    # 15/min
            'general': RateLimitConfig(100, 60.0, 'General')                  # 100/min fallback
        }
        
        # Cleanup Timer
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 Minuten
        
        logger.info("Rate Limiter initialisiert")
    
    def can_make_request(self, request_type: str = "general") -> Tuple[bool, float]:
        """
        Prüft ob Request gemacht werden darf
        
        Args:
            request_type: Art des Requests (capital_trading, huggingface, etc.)
            
        Returns:
            Tuple[bool, float]: (kann_request_machen, warte_zeit_sekunden)
        """
        config = self.configs.get(request_type, self.configs['general'])
        
        with self.lock:
            current_time = time.time()
            
            # History für diesen Request-Type initialisieren
            if request_type not in self.request_history:
                self.request_history[request_type] = []
            
            # Alte Requests außerhalb des Zeitfensters entfernen
            cutoff_time = current_time - config.time_window
            self.request_history[request_type] = [
                req_time for req_time in self.request_history[request_type] 
                if req_time > cutoff_time
            ]
            
            # Prüfen ob Limit erreicht
            if len(self.request_history[request_type]) >= config.max_requests:
                # Warte bis zum ältesten Request + time_window
                oldest_request = min(self.request_history[request_type])
                wait_time = (oldest_request + config.time_window) - current_time
                return False, max(0, wait_time)
            
            # Request hinzufügen
            self.request_history[request_type].append(current_time)
            return True, 0
    
    def wait_if_needed(self, request_type: str = "general"):
        """
        Wartet falls Rate Limit erreicht
        
        Args:
            request_type: Art des Requests
        """
        can_proceed, wait_time = self.can_make_request(request_type)
        
        if not can_proceed and wait_time > 0:
            config = self.configs.get(request_type, self.configs['general'])
            logger.info(f"Rate Limit {config.name}: Warte {wait_time:.2f}s")
            time.sleep(wait_time + 0.01)  # Kleiner Buffer
    
    def cleanup_old_requests(self):
        """Cleanup alter Request History für Memory Management"""
        current_time = time.time()
        
        if current_time - self.last_cleanup > self.cleanup_interval:
            with self.lock:
                for request_type, config in self.configs.items():
                    if request_type in self.request_history:
                        cutoff_time = current_time - config.time_window * 2  # 2x Zeitfenster
                        self.request_history[request_type] = [
                            req_time for req_time in self.request_history[request_type]
                            if req_time > cutoff_time
                        ]
                
                self.last_cleanup = current_time
                logger.debug("Rate Limiter History cleanup durchgeführt")
    
    def get_status(self) -> Dict[str, Dict]:
        """Status aller Rate Limits"""
        status = {}
        current_time = time.time()
        
        with self.lock:
            for request_type, config in self.configs.items():
                history = self.request_history.get(request_type, [])
                
                # Aktive Requests im Zeitfenster
                active_requests = [
                    req_time for req_time in history 
                    if current_time - req_time <= config.time_window
                ]
                
                can_request, wait_time = self.can_make_request(request_type)
                
                status[request_type] = {
                    'name': config.name,
                    'active_requests': len(active_requests),
                    'max_requests': config.max_requests,
                    'time_window_seconds': config.time_window,
                    'can_make_request': can_request,
                    'wait_time_seconds': wait_time,
                    'utilization_percent': round((len(active_requests) / config.max_requests) * 100, 1)
                }
        
        return status
    
    def reset_limits(self, request_type: str = None):
        """Reset Rate Limits (für Testing)"""
        with self.lock:
            if request_type:
                if request_type in self.request_history:
                    self.request_history[request_type] = []
                    logger.info(f"Rate Limit für {request_type} zurückgesetzt")
            else:
                self.request_history = {}
                logger.info("Alle Rate Limits zurückgesetzt")
