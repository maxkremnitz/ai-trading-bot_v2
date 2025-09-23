"""
Capital.com API Client
Erweitert für Multi-Account Management (4 Demo Accounts)
WICHTIG: Vermeidet Demo-Account 0 und 1 (für alten Bot reserviert)
"""

import os
import requests
import time
import logging
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)

@dataclass
class TradeResult:
    """Trade Execution Result"""
    success: bool
    deal_reference: str = ""
    deal_id: str = ""
    message: str = ""
    details: Dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

@dataclass
class Position:
    """Position Data"""
    deal_id: str
    epic: str
    direction: str
    size: float
    open_level: float
    current_level: float
    pnl: float
    currency: str

class CapitalComAPI:
    """
    Capital.com API Client mit Multi-Account Support
    Verwaltet Demo Accounts #2, #3, #4, #5 für verschiedene Strategien
    """
    
    def __init__(self, rate_limiter, strategy_name: str = "general"):
        self.rate_limiter = rate_limiter
        self.strategy_name = strategy_name
        
        # API Credentials
        self.api_key = os.getenv('CAPITAL_API_KEY')
        self.password = os.getenv('CAPITAL_PASSWORD')
        self.email = os.getenv('CAPITAL_EMAIL')
        
        # API Configuration
        self.base_url = "https://demo-api-capital.backend-capital.com"
        self.session_timeout = 600  # 10 Minuten
        
        # Session Management
        self.cst_token: Optional[str] = None
        self.security_token: Optional[str] = None
        self.account_id: Optional[str] = None
        self.last_auth_time: float = 0
        
        # Account Management
        self.available_accounts: List[Dict] = []
        self.current_account: Optional[str] = None
        self.target_account_id: Optional[str] = None
        
        # Thread Safety
        self.api_lock = threading.Lock()
        
        # Epic Mapping für Capital.com
        self.epic_mapping = {
            # Forex
            'EURUSD': 'EURUSD',
            'EUR/USD': 'EURUSD',
            
            # Crypto  
            'BITCOIN': 'BITCOIN',
            'BTC/USDT': 'BITCOIN',
            'BTCUSDT': 'BITCOIN',
            
            # Indices
            'GERMANY30': 'GERMANY30',
            'DAX': 'GERMANY30',
            'US100': 'US100', 
            'NASDAQ': 'US100',
            
            # Commodities
            'GOLD': 'GOLD',
            'OIL_CRUDE': 'OIL_CRUDE',
            'WTI': 'OIL_CRUDE',
            
            # Stocks
            'TESLA': 'TESLA',
            'TSLA': 'TESLA'
        }
        
        logger.info(f"Capital.com API initialisiert für Strategie: {strategy_name}")
    
    def is_authenticated(self) -> bool:
        """Prüft ob Session aktiv ist"""
        return (
            self.cst_token is not None and 
            self.security_token is not None and
            (time.time() - self.last_auth_time) < self.session_timeout
        )
    
    def authenticate(self) -> bool:
        """
        Authentifizierung bei Capital.com
        """
        if not self.api_key or not self.password or not self.email:
            logger.error("Capital.com Credentials fehlen")
            return False
        
        if self.is_authenticated():
            return True
        
        try:
            with self.api_lock:
                self.rate_limiter.wait_if_needed("capital_session")
                
                headers = {
                    'X-CAP-API-KEY': self.api_key,
                    'Content-Type': 'application/json'
                }
                
                data = {
                    'identifier': self.email,
                    'password': self.password,
                    'encryptedPassword': False
                }
                
                response = requests.post(
                    f"{self.base_url}/api/v1/session",
                    headers=headers,
                    json=data,
                    timeout=15
                )
                
                if response.status_code == 200:
                    self.cst_token = response.headers.get('CST')
                    self.security_token = response.headers.get('X-SECURITY-TOKEN')
                    
                    if self.cst_token and self.security_token:
                        self.last_auth_time = time.time()
                        
                        # Account Information extrahieren
                        session_data = response.json()
                        self.available_accounts = session_data.get('accounts', [])
                        self.current_account = session_data.get('currentAccountId')
                        
                        logger.info(f"Capital.com authentifiziert - {len(self.available_accounts)} Accounts verfügbar")
                        return True
                else:
                    logger.error(f"Authentifizierung fehlgeschlagen: {response.status_code} - {response.text}")
        
        except Exception as e:
            logger.error(f"Authentifizierung Error: {e}")
        
        return False
    
    def log_available_accounts(self):
        """Loggt alle verfügbaren Accounts für Debugging"""
        if not self.available_accounts:
            logger.warning("Keine Accounts verfügbar")
            return
        
        logger.info("=== Verfügbare Capital.com Demo Accounts ===")
        for i, acc in enumerate(self.available_accounts):
            acc_name = acc.get('accountName', 'Unknown')
            acc_id = acc.get('accountId', 'Unknown')
            balance_info = acc.get('balance', {})
            balance = balance_info.get('balance', 'Unknown')
            
            # Markiere reservierte Accounts
            reserved = "🚫 RESERVIERT (Alter Bot)" if ('account 0' in acc_name.lower() or 'account 1' in acc_name.lower()) else "✅ VERFÜGBAR"
            
            logger.info(f"  {i}: {acc_name} (ID: {acc_id}) - Balance: {balance} - {reserved}")
        logger.info("=" * 50)
    
    def switch_to_strategy_account(self, strategy_name: str) -> bool:
        """
        Wechselt zum Account für eine bestimmte Strategie
        WICHTIG: Vermeidet Demo-Account 0 und 1 (für alten Bot reserviert)
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return False
        
        # Account Mapping - STARTET BEI ACCOUNT 2!
        account_mapping = {
            'trend_following': ['Account 2', 'account 2'],      # Demo-Account #2
            'mean_reversion': ['Account 3', 'account 3'],       # Demo-Account #3  
            'breakout': ['Account 4', 'account 4'],             # Demo-Account #4
            'news_trading': ['Account 5', 'account 5']          # Demo-Account #5
        }
        
        target_account_names = account_mapping.get(strategy_name, [])
        if not target_account_names:
            logger.error(f"Unbekannte Strategie: {strategy_name}")
            return False
        
        # Suche passenden Account
        target_account = None
        for acc in self.available_accounts:
            acc_name = acc.get('accountName', '').lower()
            
            # WICHTIG: Account 0 und Account 1 ÜBERSPRINGEN
            if 'account 0' in acc_name or 'account 1' in acc_name:
                logger.info(f"Überspringe {acc_name} (reserviert für alten Bot)")
                continue
            
            # Suche nach passendem Account Namen
            for target_name in target_account_names:
                if target_name.lower() in acc_name:
                    target_account = acc
                    logger.info(f"Gefunden: {acc_name} für {strategy_name}")
                    break
            
            if target_account:
                break
        
        if not target_account:
            logger.error(f"Kein passender Account für {strategy_name} gefunden")
            logger.info(f"Verfügbare Accounts: {[acc.get('accountName') for acc in self.available_accounts]}")
            return False
        
        target_account_id = target_account.get('accountId')
        
        if target_account_id == self.current_account:
            logger.info(f"Bereits im Account {target_account.get('accountName')}")
            return True
        
        try:
            with self.api_lock:
                self.rate_limiter.wait_if_needed("capital_general")
                
                headers = self._get_auth_headers()
                if not headers:
                    return False
                
                data = {'accountId': target_account_id}
                
                response = requests.put(
                    f"{self.base_url}/api/v1/session",
                    headers=headers,
                    json=data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    self.current_account = target_account_id
                    self.account_id = target_account_id
                    logger.info(f"✅ Account gewechselt zu: {target_account.get('accountName')} (ID: {target_account_id})")
                    return True
                else:
                    logger.error(f"Account Switch fehlgeschlagen: {response.status_code} - {response.text}")
        
        except Exception as e:
            logger.error(f"Account Switch Error: {e}")
        
        return False
    def place_order(self, epic: str, direction: str, size: float = None, 
                   stop_distance: int = None, profit_distance: int = None,
                   use_percentage_size: bool = False, percentage: float = 4.0) -> Optional[TradeResult]:
        """
        Order bei Capital.com platzieren
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return TradeResult(False, message="Authentifizierung fehlgeschlagen")
        
        # Epic Mapping
        mapped_epic = self.epic_mapping.get(epic, epic)
        
        try:
            with self.api_lock:
                # Position Size berechnen
                if use_percentage_size or size is None:
                    size = self._calculate_position_size_percentage(percentage)
                else:
                    size = max(0.5, float(size))  # Minimum 0.5
                
                self.rate_limiter.wait_if_needed("capital_trading")
                
                headers = self._get_auth_headers()
                if not headers:
                    return TradeResult(False, message="Auth Headers fehlen")
                
                order_data = {
                    'epic': mapped_epic,
                    'direction': str(direction).upper(),
                    'size': size,
                    'guaranteedStop': False
                }
                
                # Stop Loss und Take Profit
                if stop_distance:
                    order_data['stopDistance'] = max(50, int(stop_distance))
                if profit_distance:
                    order_data['profitDistance'] = max(50, int(profit_distance))
                
                logger.info(f"Order: {mapped_epic} {direction} Size: {size} (Account: {self.get_current_account_name()})")
                
                response = requests.post(
                    f"{self.base_url}/api/v1/positions",
                    headers=headers,
                    json=order_data,
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    deal_reference = result.get('dealReference', '')
                    
                    logger.info(f"✅ Order erfolgreich platziert: {deal_reference}")
                    
                    # Deal Confirmation prüfen
                    if deal_reference:
                        time.sleep(1)
                        confirmation = self._check_deal_confirmation(deal_reference)
                        
                        return TradeResult(
                            success=True,
                            deal_reference=deal_reference,
                            deal_id=confirmation.get('dealId', '') if confirmation else '',
                            message="Order erfolgreich",
                            details={'confirmation': confirmation, 'order_data': order_data}
                        )
                    
                    return TradeResult(True, deal_reference, message="Order platziert")
                else:
                    error_msg = f"Order fehlgeschlagen ({response.status_code}): {response.text}"
                    logger.error(error_msg)
                    return TradeResult(False, message=error_msg)
        
        except Exception as e:
            error_msg = f"Order Error: {e}"
            logger.error(error_msg)
            return TradeResult(False, message=error_msg)
    
    def get_positions(self) -> List[Position]:
        """Aktuelle Positionen abrufen"""
        if not self.is_authenticated():
            return []
        
        try:
            self.rate_limiter.wait_if_needed("capital_general")
            
            headers = self._get_auth_headers()
            if not headers:
                return []
            
            response = requests.get(
                f"{self.base_url}/api/v1/positions",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                positions = []
                
                for pos_data in data.get('positions', []):
                    position = Position(
                        deal_id=pos_data.get('dealId', ''),
                        epic=pos_data.get('epic', ''),
                        direction=pos_data.get('direction', ''),
                        size=float(pos_data.get('size', 0)),
                        open_level=float(pos_data.get('openLevel', 0)),
                        current_level=float(pos_data.get('level', 0)),
                        pnl=float(pos_data.get('pnl', 0)),
                        currency=pos_data.get('currency', 'USD')
                    )
                    positions.append(position)
                
                return positions
        
        except Exception as e:
            logger.error(f"Get Positions Error: {e}")
        
        return []
    
    def get_account_balance(self) -> float:
        """Account Balance abrufen"""
        try:
            account_info = self.get_account_info()
            if account_info and 'accounts' in account_info:
                for acc in account_info['accounts']:
                    if acc.get('accountId') == self.current_account:
                        balance_info = acc.get('balance', {})
                        available = balance_info.get('available', 0)
                        return float(available) if available else 0.0
        except Exception as e:
            logger.error(f"Balance Error: {e}")
        
        return 0.0
    
    def _calculate_position_size_percentage(self, percentage: float) -> float:
        """Position Size basierend auf Prozent des verfügbaren Kapitals"""
        try:
            balance = self.get_account_balance()
            if balance <= 0:
                return 1.0  # Fallback
            
            target_amount = balance * (percentage / 100)
            
            # Größenberechnung für Capital.com
            if target_amount < 50:
                position_size = 0.5
            elif target_amount < 100:
                position_size = 1.0
            elif target_amount < 500:
                position_size = round(target_amount / 100, 1)
            else:
                position_size = round(target_amount / 200, 1)
            
            position_size = max(0.5, position_size)
            logger.info(f"Position Size: {percentage}% von ${balance} = {position_size}")
            return position_size
        
        except Exception as e:
            logger.error(f"Position Size Calculation Error: {e}")
            return 1.0
    
    def get_account_info(self) -> Optional[Dict]:
        """Account Information abrufen"""
        if not self.is_authenticated():
            return None
        
        try:
            self.rate_limiter.wait_if_needed("capital_general")
            
            headers = self._get_auth_headers()
            if not headers:
                return None
            
            response = requests.get(
                f"{self.base_url}/api/v1/accounts",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
        
        except Exception as e:
            logger.error(f"Account Info Error: {e}")
        
        return None
    
    def _check_deal_confirmation(self, deal_reference: str) -> Optional[Dict]:
        """Deal Confirmation Status prüfen"""
        try:
            self.rate_limiter.wait_if_needed("capital_general")
            
            headers = self._get_auth_headers()
            if not headers:
                return None
            
            response = requests.get(
                f"{self.base_url}/api/v1/confirms/{deal_reference}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                confirmation = response.json()
                deal_status = confirmation.get('dealStatus', 'UNKNOWN')
                logger.info(f"Deal Confirmation: {deal_status}")
                return confirmation
        
        except Exception as e:
            logger.error(f"Deal Confirmation Error: {e}")
        
        return None
    
    def _get_auth_headers(self) -> Optional[Dict]:
        """Authentication Headers erstellen"""
        if not self.cst_token or not self.security_token:
            return None
        
        return {
            'X-CAP-API-KEY': self.api_key,
            'CST': self.cst_token,
            'X-SECURITY-TOKEN': self.security_token,
            'Content-Type': 'application/json'
        }
    
    def get_current_account_name(self) -> str:
        """Name des aktuellen Accounts"""
        for acc in self.available_accounts:
            if acc.get('accountId') == self.current_account:
                return acc.get('accountName', 'Unknown')
        return 'Unknown'
    
    def ensure_authenticated(self) -> bool:
        """Auto-Reconnect"""
        if not self.is_authenticated():
            logger.info(f"Session abgelaufen - neu authentifizieren...")
            return self.authenticate()
        return True
