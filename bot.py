import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from openai import OpenAI
from google import genai
from google.genai import types

# ==========================================
# 1. CREDENTIALS & CLIENT SETUP
# ==========================================
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

try:
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
except Exception as e:
    print(f"[WARNING] Alpaca client init warning: {e}")

try:
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )
except Exception as e:
    print(f"[WARNING] Groq client init warning: {e}")

try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"[WARNING] Gemini client init warning: {e}")

SYMBOLS = ["QQQ", "AAPL", "NVDA", "MSFT", "META", "GOOGL"]
NOTIONAL_PER_TRADE = 1000.0

# ==========================================
# 2. MACRO NEWS & GEMINI ENGINE
# ==========================================
def fetch_and_verify_macro_news():
    print("[MACRO AGENT] Querying live global news & politics via Gemini...")
    
    prompt = (
        "Search the live web for breaking global news, geopolitical events, and major economic factors "
        "over the last 24-48 hours that impact global stock markets. "
        "Summarize the macro sentiment score from -1.0 (extremely bearish) to +1.0 (extremely bullish), "
        "and provide a concise synthesis.\n\n"
        "Respond strictly in JSON format with keys: 'macro_sentiment_score' (float), "
        "'verified_headlines' (list of strings), and 'summary_reasoning' (string)."
    )
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        data = json.loads(response.text)
        print(f"[MACRO AGENT] Macro Sentiment Score: {data.get('macro_sentiment_score', 0.0)}")
        return data
    except Exception as e:
        print(f"[WARNING] Failed to fetch macro news with Gemini (using neutral fallback): {e}")
        return {"macro_sentiment_score": 0.0, "verified_headlines": [], "summary_reasoning": "Fallback neutral sentiment."}

# ==========================================
# 3. ICT / FVG & ADVANCED QUANT METRICS
# ==========================================
def get_market_data(symbol):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=120)
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        bars = data_client.get_stock_bars(request_params)
        df = bars.df
        if df is None or df.empty:
            return None
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
        return df
    except Exception as e:
        print(f"[WARNING] Could not fetch market bars for {symbol}: {e}")
        return None

def detect_fair_value_gaps_and_smc(df):
    """
    Detects Fair Value Gaps (FVG):
      - BISI (Buyside Imbalance / Sellside Inefficiency) -> Bullish FVG: low[i] > high[i-2]
      - SIBI (Sellside Imbalance / Buyside Inefficiency) -> Bearish FVG: high[i] < low[i-2]
      - Consequent Encroachment (CE): 50% level of the gap.
    Also calculates Premium/Discount arrays & ATR volatility.
    """
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['Swing_High'] = df['high'].rolling(window=5).max()
    df['Swing_Low'] = df['low'].rolling(window=5).min()
    
    # ATR Volatility
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(window=14).mean()
    
    # FVG Detection Lists & Flags
    bisi_list = [] # Bullish FVG
    sibi_list = [] # Bearish FVG
    
    for i in range(2, len(df)):
        # Bullish FVG (BISI): Candle 1 high is lower than Candle 3 low
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            fvg_bottom = df['high'].iloc[i-2]
            fvg_top = df['low'].iloc[i]
            ce = (fvg_top + fvg_bottom) / 2.0
            bisi_list.append({'index': i, 'bottom': fvg_bottom, 'top': fvg_top, 'ce': ce})
            
        # Bearish FVG (SIBI): Candle 1 low is higher than Candle 3 high
        if df['high'].iloc[i] < df['low'].iloc[i-2]:
            fvg_top = df['low'].iloc[i-2]
            fvg_bottom = df['high'].iloc[i]
            ce = (fvg_top + fvg_bottom) / 2.0
            sibi_list.append({'index': i, 'bottom': fvg_bottom, 'top': fvg_top, 'ce': ce})

    # Advanced theory: Premium vs Discount Array (equilibrium of recent range)
    recent_high = df['high'].rolling(window=50).max().iloc[-1]
    recent_low = df['low'].rolling(window=50).min().iloc[-1]
    equilibrium = (recent_high + recent_low) / 2.0
    current_close = df['close'].iloc[-1]
    
    is_in_discount = current_close < equilibrium # Optimal for buying long setups (bullish FVG context)

    # Risk-Reward calculations
    atr = df['ATR'].iloc[-1]
    swing_low = df['Swing_Low'].iloc[-1]
    risk = current_close - swing_low if current_close > swing_low else atr
    risk = max(risk, atr * 0.5)
    reward = df['Swing_High'].iloc[-1] - current_close
    rr_ratio = reward / risk if risk > 0 else 0.0

    df['Calculated_RR'] = rr_ratio
    
    # Store latest FVG state for evaluation
    latest_bisi = bisi_list[-1] if bisi_list else None
    latest_sibi = sibi_list[-1] if sibi_list else None
    
    return df, latest_bisi, latest_sibi, is_in_discount

def deterministic_pre_screen(symbols):
    candidates = []
    for symbol in symbols:
        try:
            df = get_market_data(symbol)
            if df is not None and len(df) > 50:
                df, bisi, sibi, is_discount = detect_fair_value_gaps_and_smc(df)
                current_price = df['close'].iloc[-1]
                sma_20 = df['SMA_20'].iloc[-1]
                sma_50 = df['SMA_50'].iloc[-1]
                is_bullish_trend = (current_price > sma_20) and (sma_20 > sma_50)
                
                # Check if price respects Bullish FVG (BISI) Consequent Encroachment or Discount array
                fvg_respected = False
                if bisi and current_price >= bisi['ce']:
                    fvg_respected = True

                candidates.append({
                    "symbol": symbol,
                    "price": round(current_price, 2),
                    "risk_reward_ratio": round(df['Calculated_RR'].iloc[-1], 2),
                    "bullish_trend": is_bullish_trend,
                    "is_discount": is_discount,
                    "fvg_respected": fvg_respected,
                    "has_bisi": bisi is not None
                })
        except Exception as e:
            print(f"[WARNING] Skipping symbol {symbol} due to calculation error: {e}")
            
    candidates.sort(key=lambda x: (x['risk_reward_ratio'], x['fvg_respected']), reverse=True)
    return candidates

# ==========================================
# 4. EXECUTION ENGINE
# ==========================================
def execute_trades(candidates, macro_score):
    if macro_score < -0.2:
        print(f"[EXECUTION] Macro sentiment score ({macro_score}) is too bearish. Skipping trade execution.")
        return

    try:
        positions = trading_client.get_all_positions()
        print(f"[EXECUTION] Current open positions: {len(positions)}")
    except Exception as e:
        print(f"[WARNING] Could not fetch open positions: {e}")
        positions = []

    if len(positions) >= 3:
        print("[EXECUTION] Max position limit reached. Skipping new orders.")
        return

    for candidate in candidates:
        # Require bullish trend, healthy risk-reward, and alignment with discount pricing / FVG mitigation
        if candidate['bullish_trend'] and candidate['risk_reward_ratio'] > 1.5 and candidate['is_discount']:
            symbol = candidate['symbol']
            print(f"[EXECUTION] SMC Confluence met for {symbol}. Submitting market order (Notional: ${NOTIONAL_PER_TRADE})")
            try:
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    notional=NOTIONAL_PER_TRADE,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                order = trading_client.submit_order(order_data=order_data)
                print(f"[SUCCESS] Order placed for {symbol}: {order.id}")
                break 
            except Exception as e:
                print(f"[ERROR] Failed to execute order for {symbol}: {e}")

# ==========================================
# 5. MAIN CONTROLLER LOOP
# ==========================================
if __name__ == "__main__":
    print("Starting autonomous ICT/SMC trading agent cycle...")
    
    try:
        macro_data = fetch_and_verify_macro_news()
        macro_score = macro_data.get("macro_sentiment_score", 0.0)
    except Exception:
        macro_score = 0.0
    
    try:
        ranked_watchlist = deterministic_pre_screen(SYMBOLS)
        print(f"[INFO] Watchlist pre-screened FVG candidates: {ranked_watchlist}")
        execute_trades(ranked_watchlist, macro_score)
    except Exception as e:
        print(f"[WARNING] Cycle completed with warnings: {e}")
        
    print("Agent cycle completed successfully.")
    sys.exit(0)
