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

LOCK_FILE = "daily_circuit_breaker.lock"

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

SYMBOLS = ["SPY", "QQQ", "AAPL", "NVDA", "XLE", "XOM", "COP"]
NOTIONAL_PER_TRADE = 1000.0
MAX_DAILY_DRAWDOWN_PCT = 0.015  # 1.5% max account loss threshold per day

# ==========================================
# 2. DAILY ACCOUNT-LEVEL CIRCUIT BREAKER
# ==========================================
def check_daily_account_drawdown():
    print("[CIRCUIT BREAKER] Auditing total account equity and daily loss limit...")
    
    # Check if lock file exists from earlier today
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE, "r") as f:
            lock_date = f.read().strip()
        if lock_date == datetime.now().strftime("%Y-%m-%d"):
            print(f"[CIRCUIT BREAKER] 🔴 HARD LOCKOUT ACTIVE: Account hit daily loss limit on {lock_date}. Execution stopped.")
            return False

    try:
        account = trading_client.get_account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        
        daily_pnl_pct = (equity - last_equity) / last_equity if last_equity > 0 else 0.0
        print(f"[CIRCUIT BREAKER] Today's Account Equity Change: {daily_pnl_pct * 100:.2f}%")
        
        if daily_pnl_pct <= -MAX_DAILY_DRAWDOWN_PCT:
            print(f"[CIRCUIT BREAKER] 🔴 MAX DAILY DRAWDOWN HIT ({daily_pnl_pct * 100:.2f}%). Writing lock file and terminating.")
            with open(LOCK_FILE, "w") as f:
                f.write(datetime.now().strftime("%Y-%m-%d"))
            return False
            
        print("[CIRCUIT BREAKER] 🟢 Account drawdown is within safe limits.")
        return True
    except Exception as e:
        print(f"[WARNING] Account check failed: {e}. Defaulting to safe continuation.")
        return True

# ==========================================
# 3. REGIME FILTER (MARKET VOLATILITY)
# ==========================================
def market_regime_is_safe():
    print("[REGIME FILTER] Evaluating broader market structure and volatility on SPY...")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        request_params = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        bars = data_client.get_stock_bars(request_params)
        df = bars.df
        if df is None or df.empty:
            print("[WARNING] Could not fetch SPY bars. Triggering emergency RED LIGHT.")
            return False
            
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
            
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        current_price = df['close'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        
        spread = abs(sma_20 - sma_50) / current_price
        
        if spread < 0.004:
            print(f"[REGIME FILTER] 🔴 RED LIGHT: Market is compressed (Spread: {spread:.4f}). Choppy regime detected.")
            return False
            
        print(f"[REGIME FILTER] 🟢 GREEN LIGHT: Trend structure is stable (Spread: {spread:.4f}).")
        return True
    except Exception as e:
        print(f"[WARNING] Regime filter exception: {e}. Defaulting to RED LIGHT.")
        return False

# ==========================================
# 4. GEMINI LIVE INTERNET NEWS & MACRO CHECK
# ==========================================
def fetch_and_verify_macro_news():
    print("[MACRO AGENT] Querying live financial news, Treasury yields, and macro catalysts via Gemini Web Search...")
    
    prompt = (
        "Search the live web right now for current U.S. Treasury yields, Federal Reserve statements, "
        "oil price shocks, sector rotations, and major market risk factors over the last 24 hours.\n"
        "Determine if macro market conditions present a severe risk-off panic or high-yield crash environment.\n"
        "Calculate a macro sentiment score from -1.0 (extreme risk-off panic / yield spike) to +1.0 (strong risk-on / stable).\n"
        "Respond strictly in JSON format with keys: 'macro_sentiment_score' (float), "
        "'verified_headlines' (list of strings), and 'summary_reasoning' (string)."
    )
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        data = json.loads(response.text)
        score = float(data.get('macro_sentiment_score', -1.0))
        print(f"[MACRO AGENT] Live Web Macro Sentiment Score: {score}")
        print(f"[MACRO AGENT] Reasoning: {data.get('summary_reasoning')}")
        return data
    except Exception as e:
        print(f"[CRITICAL SAFETY WARNING] Live macro web search failed: {e}. Defaulting to HOSTILE score (-1.0).")
        return {
            "macro_sentiment_score": -1.0, 
            "verified_headlines": ["API Error / Search Failure"], 
            "summary_reasoning": "Forced fail-safe due to search failure."
        }

# ==========================================
# 5. ICT / SMC ORDER BLOCK ENGINE
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
# 6. GROQ ACTIVE POSITION EMERGENCY AUDITOR
# ==========================================
def groq_audit_open_positions(macro_data):
    print("[POSITION AUDITOR] Consulting Groq API for emergency pull checks on active positions...")
    try:
        positions = trading_client.get_all_positions()
    except Exception as e:
        print(f"[WARNING] Could not fetch open positions: {e}")
        return

    if not positions:
        print("[POSITION AUDITOR] No active positions to monitor.")
        return

    macro_score = macro_data.get("macro_sentiment_score", -1.0)

    for p in positions:
        symbol = p.symbol
        current_price = float(p.current_price)
        avg_entry = float(p.avg_entry_price)
        unrealized_plpc = float(p.unrealized_plpc)
        
        df = get_market_data(symbol)
        if df is None or len(df) < 50:
            continue

        df, bisi, valid_ob, is_discount, atr, swing_low = detect_order_blocks_and_smc(df)
        sma_20 = df['SMA_20'].iloc[-1]
        
        prompt = (
            f"You are a hyper-vigilant risk auditor managing open live trades. Capital preservation is paramount.\n"
            f"Audit open position for emergency pull out:\n"
            f"- Asset: {symbol}\n"
            f"- Entry Price: {avg_entry}\n"
            f"- Current Price: {current_price}\n"
            f"- Unrealized P&L (%): {round(unrealized_plpc * 100, 2)}%\n"
            f"- 20 SMA Support Level: {round(sma_20, 2)}\n"
            f"- Macro Sentiment Score (-1 to 1): {macro_score}\n\n"
            f"Instruct an EMERGENCY PULL (emergency_pull: true) if:\n"
            f"1. Unrealized P&L is negative and price breaks below 20 SMA support.\n"
            f"2. Macro sentiment score dropped below 0.0 while in this trade.\n"
            f"3. Drawdown exceeds -1.0%.\n\n"
            f"Respond strictly in JSON format with keys: 'emergency_pull' (boolean), and 'reasoning' (string)."
        )

        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a ruthless risk auditor. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            result = json.loads(completion.choices[0].message.content)
            should_pull = result.get('emergency_pull', False)
            reason = result.get('reasoning', 'No reason provided')
            
            print(f"[GROQ POSITION AUDIT] {symbol} -> Emergency Pull: {should_pull} | Reason: {reason}")
            
            if should_pull:
                print(f"[EMERGENCY PULL EXECUTION] Liquidating {symbol} immediately...")
                trading_client.close_position(symbol_or_asset_id=symbol)
                print(f"[SUCCESS] Position {symbol} closed safely.")
                
        except Exception as e:
            print(f"[WARNING] Groq position audit failed for {symbol}: {e}. Executing fallback safety check...")
            # Fallback mathematical circuit breaker if Groq API errors
            if current_price < sma_20 or unrealized_plpc < -0.01:
                print(f"[FALLBACK EMERGENCY EXIT] Closing {symbol} due to hard technical failure.")
                try:
                    trading_client.close_position(symbol_or_asset_id=symbol)
                except Exception as ex:
                    print(f"[ERROR] Emergency close failed: {ex}")

# ==========================================
# 7. GROQ AI AUDITOR (ENTRY RULE CHECKER)
# ==========================================
def groq_ai_risk_veto(candidate, macro_data):
    print(f"[GROQ RISK AUDITOR] Running high-conviction entry audit for {candidate['symbol']}...")
    
    macro_score = macro_data.get("macro_sentiment_score", -1.0)
    headlines = macro_data.get("verified_headlines", [])
    
    prompt = (
        f"You are an ultra-conservative risk management compliance auditor. Capital preservation is your sole priority.\n"
        f"Audit this trade setup against macro conditions and strict technical requirements:\n\n"
        f"- Symbol: {candidate['symbol']}\n"
        f"- Bullish Trend Active: {candidate['bullish_trend']}\n"
        f"- Intraday Momentum Positive: {candidate['intraday_momentum_positive']}\n"
        f"- Risk/Reward Ratio (Must be >= 1.8): {candidate['risk_reward_ratio']}\n"
        f"- Order Block Respected: {candidate['order_block_respected']}\n"
        f"- Live Macro Sentiment Score (-1 to 1): {macro_score}\n"
        f"- Recent Headlines: {json.dumps(headlines)}\n\n"
        "VETO the trade (approve: false) if:\n"
        "1. Macro sentiment score is below 0.1.\n"
        "2. Risk/Reward ratio is below 1.8.\n"
        "3. Order block mean threshold is compromised or momentum is weak.\n\n"
        "Respond strictly in JSON format with keys: 'approve' (boolean), and 'reasoning' (string)."
    )
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a rigid risk compliance auditor. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result = json.loads(completion.choices[0].message.content)
        print(f"[GROQ AUDIT] Decision for {candidate['symbol']} -> Approve: {result.get('approve')} | Reason: {result.get('reasoning')}")
        return result.get('approve', False)
    except Exception as e:
        print(f"[WARNING] Groq entry audit failed, defaulting to veto: {e}")
        return False

# ==========================================
# 8. EXECUTION ENGINE
# ==========================================
def execute_trades(candidates, macro_data):
    macro_score = macro_data.get("macro_sentiment_score", -1.0)
    
    if macro_score < 0.0:
        print(f"[CIRCUIT BREAKER] 🔴 MACRO LOCKOUT ACTIVE: Sentiment score ({macro_score}) indicates hostile conditions. Zero orders allowed.")
        return

    try:
        positions = trading_client.get_all_positions()
    except Exception:
        positions = []

    if len(positions) >= 2:
        print("[EXECUTION] Position limit reached. Skipping new trades.")
        return

    for candidate in candidates:
        if candidate['bullish_trend'] and candidate['intraday_momentum_positive'] and candidate['risk_reward_ratio'] >= 1.8 and candidate['order_block_detected']:
            
            if not groq_ai_risk_veto(candidate, macro_data):
                print(f"[EXECUTION] Trade VETOED by Groq Auditor for {candidate['symbol']}.")
                continue

            symbol = candidate['symbol']
            try:
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    notional=NOTIONAL_PER_TRADE,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    take_profit=TakeProfitRequest(limit_price=candidate['suggested_take_profit']),
                    stop_loss=StopLossRequest(stop_price=candidate['suggested_stop_loss'])
                )
                order = trading_client.submit_order(order_data=order_data)
                print(f"[SUCCESS] High-conviction protected order submitted for {symbol}: {order.id}")
                break 
            except Exception as e:
                print(f"[ERROR] Order execution failed for {symbol}: {e}")

# ==========================================
# 9. MAIN CONTROLLER LOOP
# ==========================================
if __name__ == "__main__":
    print("Starting hardened multi-sector audited execution cycle...")
    
    # Check account drawdown lock first
    if not check_daily_account_drawdown():
        sys.exit(0)
        
    if not market_regime_is_safe():
        print("[TERMINATION] Red light active. Exiting safely with zero orders placed.")
        sys.exit(0)
        
    macro_data = fetch_and_verify_macro_news()
    
    # 1. Audit active open positions via Groq API for emergency pull outs
    try:
        groq_audit_open_positions(macro_data)
    except Exception as e:
        print(f"[WARNING] Position audit warning: {e}")
        
    # 2. Screen watchlist and execute new trades if green-lit
    try:
        ranked_watchlist = deterministic_pre_screen(SYMBOLS)
        execute_trades(ranked_watchlist, macro_data)
    except Exception as e:
        print(f"[WARNING] Cycle warning: {e}")
        
    print("Cycle completed safely.")
    sys.exit(0)
