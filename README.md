# 🤖 Trading AI System - Multi-Strategy with AI Consensus

Ein KI-basiertes Trading-System mit 4 spezialisierten Strategien, die auf Capital.com Demo-Accounts laufen und ein 2-AI Consensus-Voting verwenden.

## 🎯 Features

- **4 Trading-Strategien** auf separaten Demo-Accounts
- **AI Consensus Voting** (Hugging Face + Google Gemini)
- **Capital.com Integration** für alle Trades
- **Memory-optimiert** für Render Deployment
- **Real-time Dashboard** mit AI-Voting Visualisierung

## 📊 Strategien

| Strategie | Demo-Account | Assets | KI-Analyse |
|-----------|-------------|---------|------------|
| **Trendfolge** | Account #1 | EUR/USD, BTC/USDT | SMA + MACD + LSTM |
| **Mean Reversion** | Account #2 | DAX, GOLD | Bollinger + RSI + RandomForest |
| **Breakout** | Account #3 | NASDAQ, TSLA | Donchian + Volume + XGBoost |
| **News Trading** | Account #4 | EUR/USD, WTI | News Sentiment + NLP |

## 🧠 AI Consensus System

- **Hugging Face API**: Sentiment Analysis & Classification
- **Google Gemini API**: Advanced Market Analysis
- **Voting Rule**: Beide AIs müssen übereinstimmen (2/2 Consensus)
- **Execution**: Nur bei identischen Labels (BUY/SELL) + Confidence >65%

## 🚀 Quick Start

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/trading-ai-system.git
cd trading-ai-system
