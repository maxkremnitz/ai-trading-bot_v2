"""
Trading AI System - Main Flask Application
Multi-Strategy Trading mit AI Consensus Voting
WICHTIG: Verwendet Demo-Accounts #2, #3, #4, #5 (nicht 0 & 1)
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

# Load Environment
load_dotenv()

# Import our modules
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from config.settings import Config
    from src.api.capital_api import CapitalComAPI
    from src.api.rate_limiter import RateLimiter
    from src.ai.voting_system import AIVotingSystem
    from src.ai.huggingface_provider import HuggingFaceProvider
    from src.ai.gemini_provider import GeminiProvider
    from src.strategies.trend_following import TrendFollowingStrategy
    from src.strategies.mean_reversion import MeanReversionStrategy
    from src.strategies.breakout import BreakoutStrategy
    from src.strategies.news_trading import NewsTradingStrategy
    from src.utils.database import DatabaseManager
    from src.utils.helpers import safe_execute, cleanup_memory, format_currency, format_percentage
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading_ai_system.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Flask App
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

class TradingAISystem:
    """
    Main Trading AI System Controller
    Verwaltet 4 Strategien mit AI Consensus Voting
    WICHTIG: Verwendet nur Demo-Accounts #2, #3, #4, #5
    """
    
    def __init__(self):
        self.config = Config()
        self.running = False
        self.start_time = datetime.now()
        
        # Core Components
        self.rate_limiter = RateLimiter()
        self.database = DatabaseManager()
        
        # AI System
        self.ai_voting_system = None
        
        # Capital.com APIs (4 separate instances for demo accounts #2, #3, #4, #5)
        self.capital_apis = {}
        
        # Trading Strategies
        self.strategies = {}
        
        # System State
        self.last_analysis_time = 0
        self.analysis_count = 0
        self.trade_count = 0
        self.error_count = 0
        self.system_status = {
            'strategies': {},
            'ai_voting': {},
            'apis': {},
            'recent_signals': [],
            'active_positions': []
        }
        
        # Background Thread
        self.analysis_thread = None
        self.thread_lock = threading.Lock()
        
        logger.info("Trading AI System initialisiert - Version 2.0")
    
    def initialize_ai_system(self) -> bool:
        """Initialisiert AI Voting System"""
        try:
            logger.info("🧠 Initialisiere AI Voting System...")
            
            # Provider Configurations
            hf_config = {
                'api_token': self.config.ai.huggingface_token,
                'timeout': self.config.ai.request_timeout,
                'min_interval_seconds': 4.0  # 15 requests/minute
            }
            
            gemini_config = {
                'api_key': self.config.ai.gemini_api_key,
                'timeout': self.config.ai.request_timeout,
                'min_interval_seconds': 4.0  # 15 requests/minute
            }
            
            # AI Voting System Configuration
            voting_config = {
                'confidence_threshold': self.config.ai.confidence_threshold,
                'consensus_enabled': self.config.ai.consensus_enabled,
                'huggingface_config': hf_config,
                'gemini_config': gemini_config
            }
            
            self.ai_voting_system = AIVotingSystem(voting_config)
            
            # Test AI Providers
            test_results = self.ai_voting_system.test_all_providers()
            logger.info(f"AI Provider Tests: {test_results}")
            
            success = all(test_results.values())
            if success:
                logger.info("✅ AI System erfolgreich initialisiert")
            else:
                logger.warning(f"⚠️ Einige AI Provider haben Probleme: {test_results}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ AI System Init Error: {e}")
            return False
    
    def initialize_capital_apis(self) -> bool:
        """Initialisiert Capital.com APIs für Demo Accounts #2, #3, #4, #5"""
        try:
            logger.info("💰 Initialisiere Capital.com APIs...")
            
            # WICHTIG: Nur Accounts #2, #3, #4, #5 verwenden!
            strategy_names = ['trend_following', 'mean_reversion', 'breakout', 'news_trading']
            
            for strategy_name in strategy_names:
                try:
                    logger.info(f"  - Initialisiere {strategy_name}...")
                    
                    api = CapitalComAPI(self.rate_limiter, strategy_name)
                    
                    if api.authenticate():
                        # Log verfügbare Accounts (für Debugging)
                        api.log_available_accounts()
                        
                        # Switch zu Strategy-spezifischem Account  
                        if api.switch_to_strategy_account(strategy_name):
                            self.capital_apis[strategy_name] = api
                            account_name = api.get_current_account_name()
                            balance = api.get_account_balance()
                            logger.info(f"  ✅ {strategy_name}: {account_name} (Balance: ${balance:.2f})")
                        else:
                            logger.warning(f"  ⚠️ Account Switch für {strategy_name} fehlgeschlagen")
                    else:
                        logger.error(f"  ❌ Authentifizierung für {strategy_name} fehlgeschlagen")
                
                except Exception as e:
                    logger.error(f"  ❌ API Init Error {strategy_name}: {e}")
            
            success = len(self.capital_apis) > 0
            logger.info(f"Capital APIs: {len(self.capital_apis)}/4 erfolgreich initialisiert")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Capital APIs Init Error: {e}")
            return False
    
    def initialize_strategies(self) -> bool:
        """Initialisiert alle 4 Trading Strategien"""
        try:
            logger.info("📊 Initialisiere Trading Strategies...")
            
            if not self.ai_voting_system:
                logger.error("❌ AI System nicht initialisiert")
                return False
            
            # Strategy Configurations
            strategy_configs = {
                'trend_following': {
                    'enabled': self.config.trading.trend_following_enabled,
                    'assets': ['EURUSD', 'BITCOIN'],
                    'max_position_size_percent': 4.0,
                    'sma_short': 50,
                    'sma_long': 200,
                    'analysis_interval_minutes': 30
                },
                'mean_reversion': {
                    'enabled': self.config.trading.mean_reversion_enabled,
                    'assets': ['GERMANY30', 'GOLD'],
                    'max_position_size_percent': 4.0,
                    'bb_period': 20,
                    'bb_std_dev': 2.0,
                    'analysis_interval_minutes': 25
                },
                'breakout': {
                    'enabled': self.config.trading.breakout_enabled,
                    'assets': ['US100', 'TESLA'],
                    'max_position_size_percent': 5.0,
                    'donchian_period': 20,
                    'analysis_interval_minutes': 35
                },
                'news_trading': {
                    'enabled': self.config.trading.news_trading_enabled,
                    'assets': ['EURUSD', 'OIL_CRUDE'],
                    'max_position_size_percent': 3.0,
                    'sentiment_lookback_hours': 24,
                    'analysis_interval_minutes': 20
                }
            }
            
            # Initialize Strategies
            for strategy_name, config in strategy_configs.items():
                try:
                    logger.info(f"  - Initialisiere {strategy_name}...")
                    
                    capital_api = self.capital_apis.get(strategy_name)
                    if not capital_api:
                        logger.warning(f"  ⚠️ Keine Capital API für {strategy_name}")
                        continue
                    
                    # Strategy Factory
                    if strategy_name == 'trend_following':
                        strategy = TrendFollowingStrategy(config, self.ai_voting_system, capital_api)
                    elif strategy_name == 'mean_reversion':
                        strategy = MeanReversionStrategy(config, self.ai_voting_system, capital_api)
                    elif strategy_name == 'breakout':
                        strategy = BreakoutStrategy(config, self.ai_voting_system, capital_api)
                    elif strategy_name == 'news_trading':
                        strategy = NewsTradingStrategy(config, self.ai_voting_system, capital_api)
                    
                    self.strategies[strategy_name] = strategy
                    
                    # Account Info für Logging
                    account_name = capital_api.get_current_account_name()
                    logger.info(f"  ✅ {strategy.name}: {account_name} | Assets: {strategy.assets}")
                    
                except Exception as e:
                    logger.error(f"  ❌ Strategy Init Error {strategy_name}: {e}")
            
            success = len(self.strategies) > 0
            logger.info(f"Strategies: {len(self.strategies)}/4 erfolgreich initialisiert")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Strategies Init Error: {e}")
            return False
    @safe_execute
    def run_analysis_cycle(self) -> Dict[str, any]:
        """Hauptanalyse-Zyklus für alle Strategien"""
        if not self.running:
            return {'success': False, 'message': 'System nicht aktiv'}
        
        analysis_start = time.time()
        results = {
            'timestamp': datetime.now().isoformat(),
            'strategies': {},
            'total_signals': 0,
            'executed_trades': 0,
            'ai_consensus_rate': 0,
            'processing_time': 0
        }
        
        try:
            with self.thread_lock:
                logger.info("🔄 Starte Analysis Cycle...")
                
                # Memory cleanup vor Analyse
                cleanup_memory()
                
                all_signals = []
                executed_trades = []
                ai_consensus_results = []
                
                # Analyse jeder Strategie
                for strategy_name, strategy in self.strategies.items():
                    if not strategy.enabled:
                        logger.info(f"  - {strategy_name}: DISABLED")
                        continue
                    
                    try:
                        logger.info(f"  📊 Analysiere {strategy_name}...")
                        
                        # Ensure API ist authenticated
                        if not strategy.capital_api.ensure_authenticated():
                            logger.error(f"  ❌ {strategy_name}: API nicht authentifiziert")
                            continue
                        
                        # Strategy Analysis
                        strategy_signals = strategy.analyze_and_signal()
                        
                        results['strategies'][strategy_name] = {
                            'signals_generated': len(strategy_signals),
                            'signals': [],
                            'trades_executed': 0,
                            'status': strategy.status.value,
                            'account': strategy.capital_api.get_current_account_name()
                        }
                        
                        for signal in strategy_signals:
                            # AI Consensus Data sammeln
                            if signal.ai_consensus:
                                ai_consensus_results.append(signal.ai_consensus)
                                
                                # Database speichern
                                self.database.save_ai_voting_result({
                                    'strategy_name': strategy_name,
                                    'symbol': signal.symbol,
                                    'hf_signal': signal.ai_consensus.individual_signals[0].signal_type.value if len(signal.ai_consensus.individual_signals) > 0 else '',
                                    'hf_confidence': signal.ai_consensus.individual_signals[0].confidence if len(signal.ai_consensus.individual_signals) > 0 else 0,
                                    'gemini_signal': signal.ai_consensus.individual_signals[1].signal_type.value if len(signal.ai_consensus.individual_signals) > 1 else '',
                                    'gemini_confidence': signal.ai_consensus.individual_signals[1].confidence if len(signal.ai_consensus.individual_signals) > 1 else 0,
                                    'consensus_result': signal.ai_consensus.voting_result.value,
                                    'final_action': signal.ai_consensus.final_action.value,
                                    'processing_time': signal.ai_consensus.processing_time
                                })
                            
                            # Strategy Analysis speichern
                            self.database.save_strategy_analysis(
                                strategy_name,
                                signal.symbol,
                                {
                                    'action': signal.action.value,
                                    'confidence': signal.confidence,
                                    'price': signal.price,
                                    'reasoning': signal.reasoning
                                },
                                signal.technical_data,
                                {
                                    'voting_result': signal.ai_consensus.voting_result.value if signal.ai_consensus else None,
                                    'consensus_confidence': signal.ai_consensus.consensus_confidence if signal.ai_consensus else 0
                                }
                            )
                            
                            # Trade Execution
                            if signal.is_executable and self.config.trading.enabled and not self.config.trading.dry_run_mode:
                                logger.info(f"  💰 Executing Trade: {signal.symbol} {signal.action.value}")
                                
                                trade_result = strategy.execute_signal(signal)
                                
                                if trade_result and trade_result.success:
                                    executed_trades.append({
                                        'strategy': strategy_name,
                                        'signal': signal,
                                        'result': trade_result
                                    })
                                    
                                    # Trade in Database speichern
                                    self.database.save_trade_execution({
                                        'strategy_name': strategy_name,
                                        'symbol': signal.symbol,
                                        'action': signal.action.value,
                                        'price': signal.price,
                                        'position_size': signal.position_size,
                                        'stop_loss': signal.stop_loss,
                                        'take_profit': signal.take_profit,
                                        'confidence': signal.confidence,
                                        'deal_reference': trade_result.deal_reference,
                                        'deal_id': trade_result.deal_id,
                                        'account_id': strategy.capital_api.current_account,
                                        'status': 'executed',
                                        'result_data': trade_result.details
                                    })
                                    
                                    results['strategies'][strategy_name]['trades_executed'] += 1
                                    self.trade_count += 1
                                    
                                    logger.info(f"  ✅ Trade executed: {trade_result.deal_reference}")
                                else:
                                    logger.warning(f"  ❌ Trade failed: {signal.symbol}")
                            elif self.config.trading.dry_run_mode:
                                logger.info(f"  🔸 DRY RUN: {signal.symbol} {signal.action.value}")
                            
                            # Signal für Results
                            results['strategies'][strategy_name]['signals'].append({
                                'symbol': signal.symbol,
                                'action': signal.action.value,
                                'confidence': signal.confidence,
                                'ai_consensus': signal.ai_consensus.action_summary if signal.ai_consensus else 'No consensus',
                                'reasoning': signal.reasoning[:100] + '...' if len(signal.reasoning) > 100 else signal.reasoning
                            })
                        
                        all_signals.extend(strategy_signals)
                        
                        # Strategy Performance Update
                        strategy.update_position_status()
                        
                        logger.info(f"  ✅ {strategy_name}: {len(strategy_signals)} signals, {results['strategies'][strategy_name]['trades_executed']} trades")
                        
                    except Exception as e:
                        logger.error(f"  ❌ Strategy Analysis Error {strategy_name}: {e}")
                        self.error_count += 1
                        results['strategies'][strategy_name] = {
                            'error': str(e),
                            'status': 'ERROR'
                        }
                
                # AI Consensus Rate berechnen
                if ai_consensus_results:
                    unanimous_results = sum(1 for result in ai_consensus_results if result.is_tradeable)
                    results['ai_consensus_rate'] = round((unanimous_results / len(ai_consensus_results)) * 100, 1)
                
                # Update System Status
                self.system_status['recent_signals'] = all_signals[-20:]  # Last 20
                results['total_signals'] = len(all_signals)
                results['executed_trades'] = len(executed_trades)
                
                self.analysis_count += 1
                self.last_analysis_time = time.time()
                
            results['processing_time'] = round(time.time() - analysis_start, 2)
            results['success'] = True
            
            logger.info(f"✅ Analysis Cycle #{self.analysis_count}: {results['total_signals']} signals, {results['executed_trades']} trades, {results['ai_consensus_rate']}% AI consensus")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Analysis Cycle Error: {e}")
            self.error_count += 1
            return {
                'success': False,
                'error': str(e),
                'processing_time': round(time.time() - analysis_start, 2)
            }
    
    def start_system(self) -> bool:
        """Startet das komplette Trading AI System"""
        try:
            logger.info("="*60)
            logger.info("🚀 Trading AI System v2.0 wird gestartet...")
            logger.info("   WICHTIG: Verwendet Demo-Accounts #2, #3, #4, #5")
            logger.info("   (Account 0 & 1 sind für den alten Bot reserviert)")
            logger.info("="*60)
            
            # 1. AI System initialisieren
            if not self.initialize_ai_system():
                logger.error("❌ AI System Initialisierung fehlgeschlagen")
                return False
            
            # 2. Capital.com APIs initialisieren
            if not self.initialize_capital_apis():
                logger.error("❌ Capital APIs Initialisierung fehlgeschlagen")
                return False
            
            # 3. Trading Strategies initialisieren
            if not self.initialize_strategies():
                logger.error("❌ Strategies Initialisierung fehlgeschlagen")
                return False
            
            # 4. Background Analysis Thread starten
            if self.config.trading.enabled:
                logger.info("🔄 Starte Background Analysis...")
                self.running = True
                self.analysis_thread = threading.Thread(target=self._background_analysis_loop, daemon=True)
                self.analysis_thread.start()
                logger.info("✅ Background Analysis Thread gestartet")
            else:
                logger.info("⚠️  Trading deaktiviert - nur Analysis Mode")
            
            logger.info("="*60)
            logger.info("🎉 Trading AI System erfolgreich gestartet!")
            logger.info(f"📊 Strategien: {list(self.strategies.keys())}")
            logger.info(f"🧠 AI Provider: {len(self.ai_voting_system.providers)} aktiv")
            logger.info(f"💰 Demo Accounts: {len(self.capital_apis)} verbunden")
            
            # Account Summary
            for strategy_name, api in self.capital_apis.items():
                account_name = api.get_current_account_name()
                logger.info(f"   - {strategy_name}: {account_name}")
            
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ System Start Error: {e}")
            return False
    
    def _background_analysis_loop(self):
        """Background Thread für kontinuierliche Analyse"""
        logger.info("🔄 Background Analysis Loop gestartet")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check ob Analyse-Interval erreicht
                interval_seconds = self.config.trading.update_interval_minutes * 60
                if current_time - self.last_analysis_time >= interval_seconds:
                    logger.info("⏰ Führe scheduled Background Analysis durch...")
                    
                    analysis_result = self.run_analysis_cycle()
                    
                    if analysis_result.get('success'):
                        logger.info(f"✅ Background Analysis erfolgreich: {analysis_result.get('total_signals', 0)} Signale")
                    else:
                        logger.error(f"❌ Background Analysis Error: {analysis_result.get('error', 'Unknown')}")
                
                # Health Check & Cleanup alle 5 Minuten
                if int(current_time) % 300 == 0:
                    self._health_check()
                    cleanup_memory()
                
                # Sleep for 30 seconds
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"❌ Background Loop Error: {e}")
                time.sleep(120)  # Longer sleep on error
    
    def _health_check(self):
        """System Health Check"""
        try:
            logger.info("🔍 Führe Health Check durch...")
            
            # API Status prüfen
            for strategy_name, api in self.capital_apis.items():
                if not api.is_authenticated():
                    logger.warning(f"⚠️ API {strategy_name} nicht authentifiziert - versuche Reconnect")
                    if api.authenticate():
                        api.switch_to_strategy_account(strategy_name)
                        logger.info(f"✅ {strategy_name} reconnected")
            
            # AI Provider Status
            if self.ai_voting_system:
                provider_tests = self.ai_voting_system.test_all_providers()
                failed_providers = [name for name, status in provider_tests.items() if not status]
                if failed_providers:
                    logger.warning(f"⚠️ AI Provider Probleme: {failed_providers}")
                else:
                    logger.info("✅ Alle AI Provider funktionsfähig")
            
            # Database Cleanup
            self.database.cleanup_old_data()
            
            logger.info("✅ Health Check abgeschlossen")
            
        except Exception as e:
            logger.error(f"❌ Health Check Error: {e}")
    
    def stop_system(self):
        """Stoppt das System"""
        logger.info("🛑 Stoppe Trading AI System...")
        self.running = False
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=10)
        
        logger.info("✅ Trading AI System gestoppt")
    
    def get_system_status(self) -> Dict:
        """Umfassender System Status für Dashboard"""
        try:
            # Strategy Status
            strategy_status = {}
            for name, strategy in self.strategies.items():
                api = self.capital_apis.get(name)
                strategy_status[name] = {
                    'name': strategy.name,
                    'enabled': strategy.enabled,
                    'status': strategy.status.value,
                    'assets': strategy.assets,
                    'performance': strategy.get_performance_stats(),
                    'account_name': api.get_current_account_name() if api else 'Unknown',
                    'account_balance': api.get_account_balance() if api else 0.0
                }
            
            # AI Voting Status
            ai_status = {}
            if self.ai_voting_system:
                ai_status = self.ai_voting_system.get_voting_stats()
                ai_status['recent_db_stats'] = self.database.get_ai_voting_stats(days=1)
            
            # API Status
            api_status = {}
            for strategy_name, api in self.capital_apis.items():
                api_status[strategy_name] = {
                    'authenticated': api.is_authenticated(),
                    'current_account': api.get_current_account_name(),
                    'balance': api.get_account_balance()
                }
            
            # Recent Activity
            recent_trades = self.database.get_recent_trades(limit=20)
            
            # System Metrics
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'running': self.running,
                    'uptime_hours': round(uptime_seconds / 3600, 1),
                    'analysis_count': self.analysis_count,
                    'trade_count': self.trade_count,
                    'error_count': self.error_count,
                    'last_analysis': datetime.fromtimestamp(self.last_analysis_time).isoformat() if self.last_analysis_time > 0 else None,
                    'next_analysis_minutes': max(0, ((self.config.trading.update_interval_minutes * 60) - (time.time() - self.last_analysis_time)) / 60) if self.last_analysis_time > 0 else 0
                },
                'strategies': strategy_status,
                'ai_voting': ai_status,
                'apis': api_status,
                'recent_trades': recent_trades,
                'config': {
                    'trading_enabled': self.config.trading.enabled,
                    'dry_run_mode': self.config.trading.dry_run_mode,
                    'update_interval_minutes': self.config.trading.update_interval_minutes,
                    'ai_confidence_threshold': self.config.ai.confidence_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Get System Status Error: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global System Instance
trading_system: Optional[TradingAISystem] = None

def initialize_system():
    """Initialisiert das globale Trading System"""
    global trading_system
    try:
        if trading_system is None:
            trading_system = TradingAISystem()
            if trading_system.start_system():
                logger.info("✅ Global Trading System erfolgreich initialisiert")
                return trading_system
            else:
                logger.error("❌ Global Trading System Initialisierung fehlgeschlagen")
                return None
        return trading_system
    except Exception as e:
        logger.error(f"❌ System Initialization Error: {e}")
        return None

# Flask Routes
@app.before_request
def ensure_system_running():
    """Stellt sicher dass System läuft"""
    global trading_system
    if not trading_system:
        trading_system = initialize_system()

@app.route("/")
def dashboard():
    """Haupt-Dashboard"""
    try:
        system = initialize_system()
        if not system:
            return f"""
            <html><head><title>System Error</title></head>
            <body style="font-family: Arial; padding: 20px; text-align: center;">
            <h1>🚨 System Initialisierung fehlgeschlagen</h1>
            <p>Das Trading AI System konnte nicht gestartet werden.</p>
            <p>Zeit: {datetime.now()}</p>
            <a href="/logs" style="color: blue;">Logs anzeigen</a>
            </body></html>
            """, 500
        
        status = system.get_system_status()
        return render_template('dashboard.html', status=status)
        
    except Exception as e:
        logger.error(f"❌ Dashboard Error: {e}")
        return f"""
        <html><head><title>Dashboard Error</title></head>
        <body style="font-family: Arial; padding: 20px; background: #f5f5f5;">
        <div style="background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h1 style="color: #e74c3c;">🚨 Dashboard Error</h1>
        <p><strong>Error:</strong> {str(e)}</p>
        <p><strong>Zeit:</strong> {datetime.now()}</p>
        <a href="/" style="background: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin-top: 10px;">Reload Dashboard</a>
        </div>
        </body></html>
        """, 500

@app.route("/api/status")
def api_status():
    """API Status Endpoint"""
    try:
        system = initialize_system()
        if not system:
            return jsonify({
                'error': 'System nicht verfügbar',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        status = system.get_system_status()
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"❌ API Status Error: {e}")
        return jsonify({
            'error': 'Status Error',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route("/api/analysis", methods=["POST"])
def trigger_analysis():
    """Manueller Analysis Trigger"""
    try:
        system = initialize_system()
        if not system:
            return jsonify({'error': 'System nicht verfügbar'}), 500
        
        if not system.running:
            return jsonify({'error': 'System nicht aktiv'}), 400
        
        logger.info("🔄 Manual Analysis durch API getriggert")
        result = system.run_analysis_cycle()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Manual Analysis Error: {e}")
        return jsonify({
            'error': 'Analysis Error',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route("/health")
def health_check():
    """Health Check für Render"""
    try:
        system = initialize_system()
        
        health_data = {
            'status': 'healthy' if system and system.running else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'uptime_hours': (datetime.now() - system.start_time).total_seconds() / 3600 if system else 0,
            'version': '2.0.0',
            'account_usage': 'Demo Accounts #2, #3, #4, #5',
            'components': {}
        }
        
        if system:
            health_data['components'] = {
                'ai_system': system.ai_voting_system is not None,
                'strategies': len(system.strategies),
                'capital_apis': len(system.capital_apis),
                'database': system.database is not None,
                'analysis_count': system.analysis_count,
                'trade_count': system.trade_count,
                'error_count': system.error_count
            }
        
        status_code = 200 if system and system.running else 503
        return jsonify(health_data), status_code
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route("/logs")
def view_logs():
    """Log Viewer"""
    try:
        log_file = 'trading_ai_system.log'
        if not os.path.exists(log_file):
            return "<h1>No Log File Found</h1>"
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        recent_lines = lines[-100:] if len(lines) > 100 else lines
        
        log_html = f"""
        <html>
        <head>
            <title>Trading AI System v2.0 Logs</title>
            <style>
                body {{ font-family: 'Courier New', monospace; background: #1a1a1a; color: #e0e0e0; padding: 20px; margin: 0; }}
                .header {{ background: #333; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .log-line {{ padding: 2px 0; border-left: 3px solid transparent; padding-left: 10px; }}
                .error {{ color: #ff6b6b; border-left-color: #ff6b6b; }}
                .warning {{ color: #feca57; border-left-color: #feca57; }}
                .info {{ color: #48dbfb; border-left-color: #48dbfb; }}
                .debug {{ color: #a4b0be; }}
                .nav {{ margin-bottom: 20px; }}
                .nav a {{ color: #48dbfb; text-decoration: none; margin-right: 20px; padding: 8px 16px; background: #333; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 Trading AI System v2.0 Logs</h1>
                <p>Demo Accounts: #2, #3, #4, #5 | Last {len(recent_lines)} lines</p>
                <p>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="nav">
                <a href="/">← Dashboard</a>
                <a href="/logs">🔄 Refresh</a>
                <a href="/api/status">📊 Status</a>
            </div>
            
            <div class="logs">
        """
        
        for line in recent_lines:
            css_class = ""
            if "ERROR" in line or "❌" in line:
                css_class = "error"
            elif "WARNING" in line or "⚠️" in line:
                css_class = "warning"
            elif "INFO" in line or "✅" in line:
                css_class = "info"
            else:
                css_class = "debug"
            
            log_html += f'<div class="log-line {css_class}">{line.strip()}</div>\n'
        
        log_html += """
            </div>
        </body>
        </html>
        """
        
        return log_html
        
    except Exception as e:
        return f"<h1>Log Error</h1><p>{str(e)}</p>"

# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not found',
        'available_endpoints': [
            '/ - Dashboard',
            '/api/status - System Status', 
            '/api/analysis - Trigger Analysis',
            '/health - Health Check',
            '/logs - Log Viewer'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

# Startup Logic
def startup_validation():
    """Startup Validation"""
    logger.info("🚀 Trading AI System v2.0 - Startup Validation")
    logger.info("   Verwendet Demo-Accounts #2, #3, #4, #5")
    
    # Environment Variables Check
    required_vars = [
        'CAPITAL_API_KEY', 'CAPITAL_PASSWORD', 'CAPITAL_EMAIL',
        'HUGGINGFACE_API_TOKEN', 'GOOGLE_GEMINI_API_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing Environment Variables: {', '.join(missing_vars)}")
        return False
    
    logger.info("✅ Environment Variables validated")
    return True

# Main Execution
if __name__ == "__main__":
    if not startup_validation():
        logger.error("❌ Startup validation fehlgeschlagen")
        sys.exit(1)
    
    try:
        port = int(os.getenv("PORT", 5000))
        debug = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        
        logger.info(f"🌐 Starting Flask app on port {port}")
        
        app.run(
            host="0.0.0.0",
            port=port,
            debug=debug,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Manual stop")
        if trading_system:
            trading_system.stop_system()
    except Exception as e:
        logger.error(f"🚨 Startup Error: {e}")
        sys.exit(1)
else:
    # WSGI Mode (Render)
    logger.info("🔄 WSGI Mode detected - initializing for Render")
    if startup_validation():
        # Initialize system in background
        threading.Timer(2.0, initialize_system).start()
        logger.info("✅ WSGI initialization scheduled")
