"""
Database Manager für Trading AI System
SQLite mit Performance-Optimierung für Render
"""

import sqlite3
import logging
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
import json
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

def safe_db_operation(func):
    """Decorator für sichere DB-Operationen"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Database Operation Error in {func.__name__}: {e}")
            return None
    return wrapper

class DatabaseManager:
    """
    SQLite Database Manager mit Performance-Optimierung
    """
    
    def __init__(self, db_path: str = "trading_ai_system.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        
        # Performance Settings
        self.cleanup_interval = 3600  # 1 Stunde
        self.last_cleanup = time.time()
        self.max_records_per_table = 10000
        
        self.init_database()
        logger.info(f"Database Manager initialisiert: {db_path}")
    
    @contextmanager
    def get_connection(self):
        """Thread-sichere Datenbankverbindung"""
        conn = None
        try:
            with self.lock:
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=30,
                    check_same_thread=False
                )
                conn.row_factory = sqlite3.Row
                
                # Performance Optimizations
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")  
                conn.execute("PRAGMA cache_size=-64000")  # 64MB Cache
                conn.execute("PRAGMA temp_store=MEMORY")
                
                yield conn
                
        except Exception as e:
            logger.error(f"Database Connection Error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def init_database(self):
        """Erstellt Datenbank-Schema"""
        try:
            with self.get_connection() as conn:
                # Strategy Analysis Table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        current_price REAL,
                        signal_type TEXT,
                        confidence REAL,
                        technical_data TEXT,
                        ai_consensus TEXT,
                        reasoning TEXT
                    )
                """)
                
                # Trade Execution Table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        price REAL,
                        position_size REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        confidence REAL,
                        deal_reference TEXT,
                        deal_id TEXT,
                        account_id TEXT,
                        status TEXT,
                        result_data TEXT,
                        pnl REAL DEFAULT 0
                    )
                """)
                
                # AI Voting Results Table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ai_voting (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        huggingface_signal TEXT,
                        huggingface_confidence REAL,
                        gemini_signal TEXT,
                        gemini_confidence REAL,
                        consensus_result TEXT,
                        final_action TEXT,
                        processing_time REAL
                    )
                """)
                
                # Performance Metrics Table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        total_trades INTEGER,
                        winning_trades INTEGER,
                        total_pnl REAL,
                        win_rate REAL,
                        avg_trade_duration REAL,
                        max_drawdown REAL,
                        sharpe_ratio REAL
                    )
                """)
                
                # System Status Table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_status (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        component TEXT NOT NULL,
                        status TEXT NOT NULL,
                        details TEXT,
                        error_count INTEGER DEFAULT 0
                    )
                """)
                
                # Indices for Performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_timestamp ON strategy_analysis(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_voting_timestamp ON ai_voting(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_strategy_name ON trades(strategy_name)")
                
                conn.commit()
                logger.info("Database Schema initialisiert")
                
        except Exception as e:
            logger.error(f"Database Init Error: {e}")
            raise
    
    @safe_db_operation
    def save_strategy_analysis(self, strategy_name: str, symbol: str, signal_data: Dict[str, Any], 
                              technical_data: Dict[str, float], ai_consensus: Optional[Dict] = None):
        """Speichert Strategy Analysis"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO strategy_analysis 
                    (timestamp, strategy_name, symbol, current_price, signal_type, confidence, 
                     technical_data, ai_consensus, reasoning)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    strategy_name,
                    symbol,
                    signal_data.get('price', 0),
                    signal_data.get('action', 'HOLD'),
                    signal_data.get('confidence', 0),
                    json.dumps(technical_data),
                    json.dumps(ai_consensus) if ai_consensus else None,
                    signal_data.get('reasoning', '')
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Save Strategy Analysis Error: {e}")
    
    @safe_db_operation  
    def save_trade_execution(self, trade_
        """Speichert Trade Execution"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO trades
                    (timestamp, strategy_name, symbol, action, price, position_size, 
                     stop_loss, take_profit, confidence, deal_reference, deal_id, 
                     account_id, status, result_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    trade_data.get('strategy_name', ''),
                    trade_data.get('symbol', ''),
                    trade_data.get('action', ''),
                    trade_data.get('price', 0),
                    trade_data.get('position_size', 0),
                    trade_data.get('stop_loss', 0),
                    trade_data.get('take_profit', 0),
                    trade_data.get('confidence', 0),
                    trade_data.get('deal_reference', ''),
                    trade_data.get('deal_id', ''),
                    trade_data.get('account_id', ''),
                    trade_data.get('status', 'pending'),
                    json.dumps(trade_data.get('result_data', {}))
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Save Trade Error: {e}")
    
    @safe_db_operation
    def save_ai_voting_result(self, voting_data: Dict[str, Any]):
        """Speichert AI Voting Results"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO ai_voting
                    (timestamp, strategy_name, symbol, huggingface_signal, huggingface_confidence,
                     gemini_signal, gemini_confidence, consensus_result, final_action, processing_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    voting_data.get('strategy_name', ''),
                    voting_data.get('symbol', ''),
                    voting_data.get('hf_signal', ''),
                    voting_data.get('hf_confidence', 0),
                    voting_data.get('gemini_signal', ''),
                    voting_data.get('gemini_confidence', 0),
                    voting_data.get('consensus_result', ''),
                    voting_data.get('final_action', ''),
                    voting_data.get('processing_time', 0)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Save AI Voting Error: {e}")
    
    @safe_db_operation
    def get_recent_trades(self, strategy_name: str = None, limit: int = 50) -> List[Dict]:
        """Holt recent trades"""
        try:
            with self.get_connection() as conn:
                query = "SELECT * FROM trades"
                params = []
                
                if strategy_name:
                    query += " WHERE strategy_name = ?"
                    params.append(strategy_name)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                
                trades = []
                for row in cursor:
                    trade = dict(row)
                    if trade['result_data']:
                        try:
                            trade['result_data'] = json.loads(trade['result_data'])
                        except:
                            trade['result_data'] = {}
                    trades.append(trade)
                
                return trades
                
        except Exception as e:
            logger.error(f"Get Recent Trades Error: {e}")
            return []
    
    @safe_db_operation
    def get_ai_voting_stats(self, days: int = 7) -> Dict[str, Any]:
        """AI Voting Statistiken"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with self.get_connection() as conn:
                # Total Votes
                cursor = conn.execute(
                    "SELECT COUNT(*) as total FROM ai_voting WHERE timestamp > ?",
                    (cutoff_date,)
                )
                total_votes = cursor.fetchone()['total']
                
                # Consensus Rate
                cursor = conn.execute(
                    "SELECT COUNT(*) as consensus FROM ai_voting WHERE timestamp > ? AND consensus_result LIKE 'UNANIMOUS%'",
                    (cutoff_date,)
                )
                consensus_votes = cursor.fetchone()['consensus']
                
                # Average Processing Time
                cursor = conn.execute(
                    "SELECT AVG(processing_time) as avg_time FROM ai_voting WHERE timestamp > ?",
                    (cutoff_date,)
                )
                avg_processing_time = cursor.fetchone()['avg_time'] or 0
                
                consensus_rate = (consensus_votes / max(1, total_votes)) * 100
                
                return {
                    'total_votes': total_votes,
                    'consensus_votes': consensus_votes,
                    'consensus_rate': round(consensus_rate, 1),
                    'avg_processing_time': round(avg_processing_time, 2),
                    'period_days': days
                }
                
        except Exception as e:
            logger.error(f"AI Voting Stats Error: {e}")
            return {}
    
    @safe_db_operation
    def cleanup_old_data(self):
        """Cleanup alter Daten für Performance"""
        current_time = time.time()
        
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
            
            with self.get_connection() as conn:
                # Cleanup old analysis data
                conn.execute(
                    "DELETE FROM strategy_analysis WHERE id NOT IN (SELECT id FROM strategy_analysis ORDER BY timestamp DESC LIMIT ?)",
                    (self.max_records_per_table,)
                )
                
                # Cleanup old voting data
                conn.execute(
                    "DELETE FROM ai_voting WHERE timestamp < ?",
                    (cutoff_date,)
                )
                
                # Keep all trades (for performance analysis)
                # But cleanup very old system status
                old_cutoff = (datetime.now() - timedelta(days=7)).isoformat()
                conn.execute(
                    "DELETE FROM system_status WHERE timestamp < ?",
                    (old_cutoff,)
                )
                
                conn.commit()
                
                # Vacuum for space recovery
                conn.execute("VACUUM")
                
                self.last_cleanup = current_time
                logger.info("Database cleanup completed")
                
        except Exception as e:
            logger.error(f"Database Cleanup Error: {e}")
