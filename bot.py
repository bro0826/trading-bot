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
# 2. FACT-CHECKED POLITICAL & MACRO ENGINE
# ==========================================
def fetch_and_verify_macro_news():
    print("[MACRO AGENT] Querying and fact-checking live political/White House policies via Gemini...")
    
    prompt = (
        "Search the live web for breaking political news, White House announcements, executive orders, "
        "key political speeches, regulatory decisions, and government policy shifts over the last 24-48 hours.\n"
        "CRITICAL INSTRUCTION: Fact-check every claim across multiple searches. Do not trust single-source rumors "
        "or unverified social media chatter. Ensure the policy, speech, or directive is actively underway and verified.\n"
        "Summarize the overall political/macro sentiment score from -1.0 (extremely hostile/bearish policy) "
        "to +1.0 (extremely supportive/bullish policy), list the verified sources/headlines, and provide a strict reasoning synthesis.\n\n"
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
                temperature=0.0
            )
        )
        data = json.loads(response.text)
        print(f"[MACRO AGENT] Fact-Checked Political/Macro Sentiment Score: {data.get('macro_sentiment_score', 0.0)}")
        return data
    except Exception as e:
        print(f"[WARNING] Failed to fetch or fact-check political news with Gemini (using neutral fallback): {e}")
        return {"macro_sentiment_score": 0.0, "verified_headlines": [], "summary_reasoning": "Fallback neutral sentiment."}

# ==========================================
# 3. ICT / ADVANCED ORDER BLOCK, FVG & SMC ENGINE
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
    
    # Fair Value Gap (Imbalance) Detection
    bisi_list = [] 
    for i in range(2, len(df)):
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            fvg_bottom = df['high'].iloc[i-2]
            fvg_top = df['low'].iloc[i]
            ce = (fvg_top + fvg_bottom) / 2.0
            bisi_list.append({'index': i, 'bottom': fvg_bottom, 'top': fvg_top, 'ce': ce})

    # Advanced Bullish Order Block Detection
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
                
                # FIX: Require intraday bullish momentum (price > open) to avoid buying a fading candle
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
            print(f"[WARNING] Skipping symbol {symbol} due to calculation error: {e}")
            
    candidates.sort(key=lambda x: (x['risk_reward_ratio'], x['order_block_respected'], x['fvg_respected']), reverse=True)
    return candidates

# ==========================================
# 4. ACTIVE POSITION MANAGER & EARLY EXIT ENGINE
# ==========================================
def manage_open_positions():
    print("[POSITION MANAGER] Checking active positions for proactive risk management & early exits...")
    try:
        positions = trading_client.get_all_positions()
    except Exception as e:
        print(f"[WARNING] Could not fetch positions for management: {e}")
        return

    for p in positions:
        symbol = p.symbol
        current_price = float(p.current_price)
        avg_entry = float(p.avg_entry_price)
        unrealized_plpc = float(p.unrealized_plpc)
        
        print(f"[POSITION MANAGER] Evaluating open position {symbol} | Entry: {avg_entry} | Current: {current_price} | P&L: {unrealized_plpc*100:.2f}%")
        
        df = get_market_data(symbol)
        if df is not None and len(df) > 50:
            df, bisi, valid_ob, is_discount, atr, swing_low = detect_order_blocks_and_smc(df)
            
            mean_threshold_broken = valid_ob and (current_price < valid_ob['mean_threshold'])
            trend_lost = current_price < df['SMA_20'].iloc[-1]
            
            if unrealized_plpc > 0.005 and (mean_threshold_broken or trend_lost):
                print(f"[EARLY EXIT] Securing profits on {symbol} due to structural breakdown. Closing position.")
                try:
                    trading_client.close_position(symbol_or_asset_id=symbol)
                    print(f"[SUCCESS] Successfully closed {symbol} early to lock in gains.")
                except Exception as e:
                    print(f"[ERROR] Failed to close position for {symbol}: {e}")
            
            elif mean_threshold_broken and unrealized_plpc < -0.005:
                print(f"[EARLY EXIT] Cutting losses early on {symbol} as order block mean threshold failed.")
                try:
                    trading_client.close_position(symbol_or_asset_id=symbol)
                    print(f"[SUCCESS] Successfully cut {symbol} early.")
                except Exception as e:
                    print(f"[ERROR] Failed to close position for {symbol}: {e}")

# ==========================================
# 5. GROQ AI VETO & VALIDATION AGENT
# ==========================================
def groq_ai_risk_veto(candidate, macro_data):
    print(f"[GROQ RISK AGENT] Cross-auditing trade validity, order block structure, and macro claims for {candidate['symbol']}...")
    
    macro_score = macro_data.get("macro_sentiment_score", 0.0)
    headlines = macro_data.get("verified_headlines", [])
    
    prompt = (
        f"You are a strict, skeptical risk management AI controller. Your goal is capital preservation.\n"
        f"Review the trade setup against the advanced Order Block & Fair Value Gap parameters and fact-checked macro data below:\n\n"
        f"--- TRADE CANDIDATE ---\n"
        f"- Symbol: {candidate['symbol']}\n"
        f"- Current Price: {candidate['price']}\n"
        f"- Bullish Trend Confirmed: {candidate['bullish_trend']}\n"
        f"- Intraday Momentum Positive (Price >= Open): {candidate['intraday_momentum_positive']}\n"
        f"- Risk/Reward Ratio: {candidate['risk_reward_ratio']}\n"
        f"- Optimal Discount Zone: {candidate['is_discount']}\n"
        f"- Fair Value Gap Imbalance Present: {candidate['fvg_respected']}\n"
        f"- Valid Order Block Detected & Respected (Mean Threshold Held): {candidate['order_block_respected']}\n\n"
        f"--- FACT-CHECKED POLITICAL CONTEXT ---\n"
        f"- Macro Sentiment Score (-1 to 1): {macro_score}\n"
        f"- Verified Headlines/Policies: {json.dumps(headlines)}\n\n"
        "Strictly evaluate if the order block structure, institutional delivery shift, and political context support this entry. If intraday momentum is negative or the order block is violated, VETO it.\n"
        "Respond strictly in JSON format with keys: 'approve' (boolean: true or false), and 'reasoning' (string)."
    )
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a skeptical, strict risk management auditor bot. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result = json.loads(completion.choices[0].message.content)
        print(f"[GROQ RISK AGENT] Decision for {candidate['symbol']} -> Approve: {result.get('approve')} | Reason: {result.get('reasoning')}")
        return result.get('approve', False)
    except Exception as e:
        print(f"[WARNING] Groq API call failed, defaulting to veto for safety: {e}")
        return False

# ==========================================
# 6. EXECUTION & NEW TRADES ENGINE
# ==========================================
def execute_trades(candidates, macro_data):
    macro_score = macro_data.get("macro_sentiment_score", 0.0)
    if macro_score < -0.2:
        print(f"[EXECUTION] Fact-checked political/macro sentiment score ({macro_score}) is too hostile/bearish. Skipping trade execution.")
        return

    try:
        positions = trading_client.get_all_positions()
        print(f"[EXECUTION] Current open positions count: {len(positions)}")
    except Exception as e:
        print(f"[WARNING] Could not fetch open positions: {e}")
        positions = []

    if len(positions) >= 3:
        print("[EXECUTION] Max position limit reached. Skipping new orders.")
        return

    for candidate in candidates:
        # FIX: Added strict check for positive intraday momentum before execution
        if candidate['bullish_trend'] and candidate['intraday_momentum_positive'] and candidate['risk_reward_ratio'] > 1.2 and candidate['order_block_detected']:
            
            is_approved_by_ai = groq_ai_risk_veto(candidate, macro_data)
            
            if not is_approved_by_ai:
                print(f"[EXECUTION] Groq VETOED trade for {candidate['symbol']}. Skipping.")
                continue

            symbol = candidate['symbol']
            sl_price = candidate['suggested_stop_loss']
            tp_price = candidate['suggested_take_profit']
            
            print(f"[EXECUTION] Submitting protected order block bracket for {symbol} (Notional: ${NOTIONAL_PER_TRADE}) | TP: {tp_price} | SL: {sl_price}")
            
            try:
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    notional=NOTIONAL_PER_TRADE,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    take_profit=TakeProfitRequest(limit_price=tp_price),
                    stop_loss=StopLossRequest(stop_price=sl_price)
                )
                order = trading_client.submit_order(order_data=order_data)
                print(f"[SUCCESS] Protected order placed for {symbol}: {order.id}")
                break 
            except Exception as e:
                print(f"[ERROR] Failed to execute order for {symbol}: {e}")

# ==========================================
# 7. MAIN CONTROLLER LOOP
# ==========================================
if __name__ == "__main__":
    print("Starting active-management Order Block & Macro trading cycle...")
    
    try:
        manage_open_positions()
    except Exception as e:
        print(f"[WARNING] Position management encountered an error: {e}")
    
    try:
        macro_data = fetch_and_verify_macro_news()
    except Exception:
        macro_data = {"macro_sentiment_score": 0.0, "verified_headlines": [], "summary_reasoning": "Error fallback."}
    
    try:
        ranked_watchlist = deterministic_pre_screen(SYMBOLS)
        print(f"[INFO] Watchlist pre-screened candidates: {ranked_watchlist}")
        execute_trades(ranked_watchlist, macro_data)
    except Exception as e:
        print(f"[WARNING] Cycle completed with warnings: {e}")
        
    print("Agent cycle completed successfully.")
    sys.exit(0)
