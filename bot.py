import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
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
    print(f"[CRITICAL ERROR] Alpaca client initialization failed: {e}")
    sys.exit(1)

try:
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )
except Exception as e:
    print(f"[CRITICAL ERROR] Groq client initialization failed: {e}")
    sys.exit(1)

try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"[CRITICAL ERROR] Gemini client initialization failed: {e}")
    sys.exit(1)

SYMBOLS = ["QQQ", "AAPL", "NVDA", "MSFT", "META", "GOOGL"]
NOTIONAL_PER_TRADE = 1000.0

# ==========================================
# 2. AUTOMATED REGIME & VOLATILITY FILTER
# ==========================================
def market_regime_is_safe():
    print("[REGIME FILTER] Evaluating market structure and trend volatility on QQQ...")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        request_params = StockBarsRequest(
            symbol_or_symbols="QQQ",
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        bars = data_client.get_stock_bars(request_params)
        df = bars.df
        if df is None or df.empty:
            print("[WARNING] Could not fetch QQQ bars. Triggering emergency RED LIGHT.")
            return False
            
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
            
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        current_price = df['close'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        
        spread = abs(sma_20 - sma_50) / current_price
        
        # Hard check for compressed, choppy sideways chop
        if spread < 0.005:
            print(f"[REGIME FILTER] 🔴 RED LIGHT: Market is compressed (Spread: {spread:.4f}). Halting execution.")
            return False
            
        print(f"[REGIME FILTER] 🟢 GREEN LIGHT: Trend structure is stable (Spread: {spread:.4f}).")
        return True
    except Exception as e:
        print(f"[WARNING] Regime filter exception: {e}. Defaulting to RED LIGHT.")
        return False

# ==========================================
# 3. LIVE WEB-SEARCH MACRO & YIELD AUDITOR
# ==========================================
def fetch_and_verify_macro_news():
    print("[MACRO AGENT] Querying live financial news, Treasury yields, and geopolitical risk via Gemini Web Search...")
    
    prompt = (
        "Search the live web right now for current U.S. Treasury 10-year yields, Federal Reserve statements, "
        "crude oil price movements, and major market risk factors over the last 24 hours.\n"
        "Evaluate whether macro market conditions are safe for long equity momentum strategies or if there is a severe risk-off panic.\n"
        "Calculate a macro sentiment score from -1.0 (extreme risk-off crash/high yield panic) to +1.0 (strong risk-on/bullish).\n"
        "Respond strictly in JSON format with keys: 'macro_sentiment_score' (float), "
        "'verified_headlines' (list of strings), and 'summary_reasoning' (string)."
    )
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        data = json.loads(response.text)
        score = float(data.get('macro_sentiment_score', -1.0)) # Default to hostile if unreadable
        print(f"[MACRO AGENT] Live Web Macro Sentiment Score: {score}")
        print(f"[MACRO AGENT] Reasoning: {data.get('summary_reasoning')}")
        return data
    except Exception as e:
        print(f"[CRITICAL SAFETY WARNING] Live macro web search failed: {e}. Defaulting to HOSTILE macro score (-1.0) to prevent losses.")
        return {
            "macro_sentiment_score": -1.0, 
            "verified_headlines": ["API Error / Search Failure"], 
            "summary_reasoning": "Forced fail-safe due to search failure."
        }

# ==========================================
# 4. ICT / ORDER BLOCK & SMC ENGINE
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

def detect_order_blocks_and_smc(df):
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['Swing_High'] = df['high'].rolling(window=5).max()
    df['Swing_Low'] = df['low'].rolling(window=5).min()
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(window=14).mean()
    
    bisi_list = [] 
    for i in range(2, len(df)):
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            fvg_bottom = df['high'].iloc[i-2]
            fvg_top = df['low'].iloc[i]
            ce = (fvg_top + fvg_bottom) / 2.0
            bisi_list.append({'index': i, 'bottom': fvg_bottom, 'top': fvg_top, 'ce': ce})

    valid_ob = None
    for i in range(5, len(df) - 1):
        is_down_candle = df['close'].iloc[i] < df['open'].iloc[i]
        next_is_strong_up = (df['close'].iloc[i+1] > df['open'].iloc[i+1]) and ((df['close'].iloc[i+1] - df['open'].iloc[i+1]) > df['ATR'].iloc[i+1])
        swept_liquidity = df['low'].iloc[i] <= df['Swing_Low'].iloc[i]
        has_imbalance = any(b['index'] >= i for b in bisi_list)

        if is_down_candle and next_is_strong_up and swept_liquidity and has_imbalance:
            ob_open = df['open'].iloc[i]
            ob_close = df['close'].iloc[i]
            body_low = min(ob_open, ob_close)
            body_high = max(ob_open, ob_close)
            mean_threshold = (body_low + body_high) / 2.0
            
            valid_ob = {
                "index": i,
                "entry_price": ob_open,
                "mean_threshold": mean_threshold,
                "stop_loss": body_low - (df['ATR'].iloc[i] * 0.5)
            }

    recent_high = df['high'].rolling(window=50).max().iloc[-1]
    recent_low = df['low'].rolling(window=50).min().iloc[-1]
    equilibrium = (recent_high + recent_low) / 2.0
    current_close = df['close'].iloc[-1]
    is_in_discount = current_close < equilibrium

    atr = df['ATR'].iloc[-1]
    swing_low = df['Swing_Low'].iloc[-1]
    risk = current_close - swing_low if current_close > swing_low else atr
    risk = max(risk, atr * 0.5)
    reward = df['Swing_High'].iloc[-1] - current_close
    rr_ratio = reward / risk if risk > 0 else 0.0

    df['Calculated_RR'] = rr_ratio
    latest_bisi = bisi_list[-1] if bisi_list else None
    
    return df, latest_bisi, valid_ob, is_in_discount, atr, swing_low

def deterministic_pre_screen(symbols):
    candidates = []
    for symbol in symbols:
        try:
            df = get_market_data(symbol)
            if df is not None and len(df) > 50:
                df, bisi, valid_ob, is_discount, atr, swing_low = detect_order_blocks_and_smc(df)
                current_price = df['close'].iloc[-1]
                current_open = df['open'].iloc[-1]
                
                intraday_momentum_positive = current_price >= current_open
                
                sma_20 = df['SMA_20'].iloc[-1]
                sma_50 = df['SMA_50'].iloc[-1]
                is_bullish_trend = (current_price > sma_20) and (sma_20 > sma_50)
                
                fvg_respected = False
                if bisi and current_price >= bisi['ce']:
                    fvg_respected = True

                ob_respected = False
                if valid_ob:
                    if current_price >= valid_ob['mean_threshold']:
                        ob_respected = True

                candidates.append({
                    "symbol": symbol,
                    "price": round(current_price, 2),
                    "risk_reward_ratio": round(df['Calculated_RR'].iloc[-1], 2),
                    "bullish_trend": is_bullish_trend,
                    "intraday_momentum_positive": intraday_momentum_positive,
                    "is_discount": is_discount,
                    "fvg_respected": fvg_respected,
                    "order_block_detected": valid_ob is not None,
                    "order_block_respected": ob_respected,
                    "atr": round(atr, 2),
                    "suggested_stop_loss": round(valid_ob['stop_loss'] if valid_ob else current_price - (atr * 1.5), 2),
                    "suggested_take_profit": round(current_price + (atr * 3.0), 2)
                })
        except Exception as e:
            print(f"[WARNING] Skipping symbol {symbol}: {e}")
            
    candidates.sort(key=lambda x: (x['risk_reward_ratio'], x['order_block_respected'], x['fvg_respected']), reverse=True)
    return candidates

# ==========================================
# 5. ACTIVE POSITION RISK MANAGER
# ==========================================
def manage_open_positions():
    print("[POSITION MANAGER] Evaluating active positions for early risk exits...")
    try:
        positions = trading_client.get_all_positions()
    except Exception as e:
        print(f"[WARNING] Could not fetch positions: {e}")
        return

    for p in positions:
        symbol = p.symbol
        current_price = float(p.current_price)
        unrealized_plpc = float(p.unrealized_plpc)
        
        df = get_market_data(symbol)
        if df is not None and len(df) > 50:
            df, bisi, valid_ob, is_discount, atr, swing_low = detect_order_blocks_and_smc(df)
            trend_lost = current_price < df['SMA_20'].iloc[-1]
            
            if trend_lost or unrealized_plpc < -0.01:
                print(f"[EMERGENCY EXIT] Cutting losses on {symbol} due to trend failure or drawdown limit.")
                try:
                    trading_client.close_position(symbol_or_asset_id=symbol)
                except Exception as e:
                    print(f"[ERROR] Failed to close position {symbol}: {e}")

# ==========================================
# 6. GROQ AI RISK AUDITOR
# ==========================================
def groq_ai_risk_veto(candidate, macro_data):
    print(f"[GROQ RISK AGENT] Auditing {candidate['symbol']} setup...")
    
    macro_score = macro_data.get("macro_sentiment_score", -1.0)
    headlines = macro_data.get("verified_headlines", [])
    
    prompt = (
        f"You are an ultra-conservative risk management AI controller. Capital preservation is your only directive.\n"
        f"Evaluate this trade setup against live web-scraped macro data:\n\n"
        f"- Symbol: {candidate['symbol']}\n"
        f"- Bullish Trend: {candidate['bullish_trend']}\n"
        f"- Intraday Momentum Positive: {candidate['intraday_momentum_positive']}\n"
        f"- Risk/Reward Ratio: {candidate['risk_reward_ratio']}\n"
        f"- Order Block Respected: {candidate['order_block_respected']}\n"
        f"- Live Macro Sentiment Score (-1 to 1): {macro_score}\n"
        f"- Recent Headlines: {json.dumps(headlines)}\n\n"
        "If macro conditions show high yields, panic, or negative sentiment, VETO immediately.\n"
        "Respond strictly in JSON format with keys: 'approve' (boolean), and 'reasoning' (string)."
    )
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a skeptical risk auditor. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result = json.loads(completion.choices[0].message.content)
        return result.get('approve', False)
    except Exception as e:
        print(f"[WARNING] Groq audit failed, defaulting to veto: {e}")
        return False

# ==========================================
# 7. EXECUTION ENGINE
# ==========================================
def execute_trades(candidates, macro_data):
    macro_score = macro_data.get("macro_sentiment_score", -1.0)
    
    # HARD MACRO CIRCUIT BREAKER: Refuse all trades if macro score is below 0.0
    if macro_score < 0.0:
        print(f"[CIRCUIT BREAKER] 🔴 MACRO LOCKOUT ACTIVE: Sentiment score ({macro_score}) indicates hostile conditions. Zero orders allowed.")
        return

    try:
        positions = trading_client.get_all_positions()
    except Exception:
        positions = []

    if len(positions) >= 2:
        print("[EXECUTION] Position limit reached. Skipping.")
        return

    for candidate in candidates:
        if candidate['bullish_trend'] and candidate['intraday_momentum_positive'] and candidate['risk_reward_ratio'] > 1.3 and candidate['order_block_detected']:
            
            if not groq_ai_risk_veto(candidate, macro_data):
                print(f"[EXECUTION] Trade VETOED for {candidate['symbol']}.")
                continue

            symbol = candidate['symbol']
            try:
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    notional=NOTIONAL_PER_TRADE,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    take_profit=TakeProfitRequest(limit_price=candidate['suggested_take_profit']),
                    stop_loss=StopLossRequest(stop_press=candidate['suggested_stop_loss']) if 'suggested_stop_loss' in candidate else None
                )
                order = trading_client.submit_order(order_data=order_data)
                print(f"[SUCCESS] Protected order submitted for {symbol}: {order.id}")
                break 
            except Exception as e:
                print(f"[ERROR] Order execution failed for {symbol}: {e}")

# ==========================================
# 8. MAIN CONTROLLER LOOP
# ==========================================
if __name__ == "__main__":
    print("Starting bulletproof trading execution cycle...")
    
    if not market_regime_is_safe():
        print("[TERMINATION] Red light active. Exiting safely.")
        sys.exit(0)
        
    try:
        manage_open_positions()
    except Exception as e:
        print(f"[WARNING] Position manager warning: {e}")
    
    # Force live web check; defaults to safe hostile block if any error occurs
    macro_data = fetch_and_verify_macro_news()
    
    try:
        ranked_watchlist = deterministic_pre_screen(SYMBOLS)
        execute_trades(ranked_watchlist, macro_data)
    except Exception as e:
        print(f"[WARNING] Cycle warning: {e}")
        
    print("Cycle completed safely.")
    sys.exit(0)
