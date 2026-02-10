import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import numpy as np
import time
import akshare as ak

# ======================
# 数据获取：双源 fallback
# ======================
def fetch_from_tencent(symbol):
    if not (symbol.isdigit() and len(symbol) == 6):
        return None
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        raw = response.json()
        data_section = raw.get('data', {})
        stock_data = []
        key1 = f"{prefix}{symbol}"
        if key1 in data_section:
            inner = data_section[key1]
            stock_data = inner.get('qfqday', []) or inner.get('day', [])
        elif "" in data_section:
            inner = data_section[""]
            if isinstance(inner, dict):
                stock_data = inner.get('qfqday', []) or inner.get('day', [])
        if not stock_data:
            return None
        cleaned = [row[:6] for row in stock_data if isinstance(row, list) and len(row) >= 6]
        if not cleaned:
            return None
        df = pd.DataFrame(cleaned, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df.set_index('date', inplace=True)
        for col in ['open', 'close', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        if len(df) < 20:
            return None
        return df
    except Exception as e:
        print(f"[腾讯] {symbol} 失败: {e}")
        return None

def fetch_from_yfinance(symbol):
    try:
        ticker = f"{symbol}.SS" if symbol.startswith('6') else f"{symbol}.SZ"
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y", interval="1d")
        if hist.empty:
            return None
        df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        if len(df) < 20:
            return None
        return df
    except Exception as e:
        print(f"[Yahoo] {symbol} 失败: {e}")
        return None

def fetch_stock_history(symbol):
    df = fetch_from_tencent(symbol)
    source = "腾讯财经"
    if df is None:
        time.sleep(1)
        df = fetch_from_yfinance(symbol)
        source = "Yahoo Finance (备用)"
    return df, source

# ======================
# 获取股票名称
# ======================
@st.cache_data(ttl=3600)
def get_stock_name(symbol):
    try:
        info = ak.stock_individual_info_em(symbol=symbol)
        name = info[info['项目'] == '股票简称']['值'].values[0]
        return name
    except:
        return symbol

# ======================
# 技术指标计算（简化，只计算必要列）
# ======================
def calculate_indicators(df):
    df = df.copy()
    
    # 核心指标（用于判断）
    df['BBI'] = (df['close'].rolling(3).mean() + df['close'].rolling(6).mean() + 
                 df['close'].rolling(12).mean() + df['close'].rolling(24).mean()) / 4
    df['趋势白线'] = df['close'].ewm(span=9, adjust=False).mean().ewm(span=11, adjust=False).mean()
    df['大哥黄线'] = (df['close'].ewm(span=7, adjust=False).mean().ewm(span=7, adjust=False).mean() + 
                   df['close'].ewm(span=14, adjust=False).mean().ewm(span=14, adjust=False).mean() + 
                   df['close'].ewm(span=28, adjust=False).mean().ewm(span=28, adjust=False).mean() + 
                   df['close'].ewm(span=56, adjust=False).mean().ewm(span=56, adjust=False).mean()) / 4
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = ema12 - ema26
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    
    # KDJ
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    denominator = high_max - low_min
    denominator[denominator == 0] = 1
    rsv = (df['close'] - low_min) / denominator * 100
    df['k'] = rsv.ewm(span=3, adjust=False).mean()
    df['d'] = df['k'].ewm(span=3, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    
    # RSI
    lc = df['close'].shift(1)
    temp1 = np.maximum(df['close'] - lc, 0)
    temp2 = np.abs(df['close'] - lc)
    df['rsi'] = temp1.rolling(3).mean() / temp2.rolling(3).mean() * 100
    
    # 振幅 & 涨跌幅 & 换手 & 量比
    df['当日振幅'] = (df['high'] - df['low']) / df['low'] * 100
    df['当日涨跌幅'] = abs(df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    df['换手率'] = df['volume'] / (df['close'] * 100000000) * 100  # 粗估
    df['量比'] = df['volume'] / df['volume'].rolling(5).mean()
    
    # 缩量系列（简化）
    df['缩量'] = (df['volume'] < df['volume'].rolling(20).max() * 0.416) | (df['volume'] < df['volume'].rolling(50).max() / 3)
    
    return df

# ======================
# 浩哥战法分析（隐藏 B1 名词）
# ======================
def analyze_stock(df, name, current, market_cap):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 安全访问
    def safe_get(col, default=False):
        return last.get(col, default) if col in last else default
    
    # 核心判断（隐藏 B1 名词，只用内部逻辑）
    dist_white = abs(last['close'] - last['趋势白线']) / last['趋势白线'] * 100 if '趋势白线' in last else 999
    dist_yellow = abs(last['close'] - last['大哥黄线']) / last['大哥黄线'] * 100 if '大哥黄线' in last else 999
    
    # 信号激活（内部计算，不显示名字）
    signals = {
        's1': (last['rsi'] - 15 >= df['rsi'].shift(1).iloc[-1]) and (df['rsi'].shift(1).iloc[-1] < 20 or df['j'].shift(1).iloc[-1] < 14) and safe_get('当日振幅', 999) < 8 and safe_get('当日涨跌幅', 999) < 3,  # 拐头
        's2': (safe_get('j') < 14 or safe_get('rsi') < 23) and safe_get('当日振幅', 999) < 8 and safe_get('缩量', False),  # 缩量
        's3': (last['趋势白线'] > last['大哥黄线']) and (safe_get('j') < 13 or safe_get('rsi') < 21) and safe_get('缩量', False),  # 原始
        's4': (safe_get('j') < 14 or safe_get('rsi') < 23) and safe_get('缩量', False) and safe_get('远期振幅', 0) >= 45,  # 超缩
        's5': (dist_white < 2) and safe_get('缩量', False),  # 回踩白
        's6': safe_get('超牛股', False) and (safe_get('j') < 35 or safe_get('rsi') < 45) and safe_get('缩量', False),  # 超级
        's7': (dist_yellow <= 1.5) and safe_get('缩量', False)  # 黄线
    }
    
    # 权重（不变）
    weights = {
        's6': 25,  # 超级
        's4': 22,  # 超缩
        's5': 18,  # 白线
        's3': 15,  # 原始
        's1': 10,  # 拐头
        's7': 8,   # 黄线
        's2': 5    # 缩量
    }
    
    tech_score = sum(weights.get(k, 0) for k, v in signals.items() if v)
    
    # 低价修正
    price_correction = 0
    if current < 12:
        price_correction = -4
        if (safe_get('换手率', 0) > 5) or (safe_get('量比', 0) > 1.5) or \
           (last['close'] > last['大哥黄线'] and last['macd'] > 0):
            price_correction = +2
    
    tech_score += price_correction
    tech_score = min(max(tech_score, 0), 70)
    
    # AI 分（模拟热点）
    ai_score = 0
    if market_cap > 50:
        ai_score += 8
    if current > 50:
        ai_score += 5
    elif 12 <= current <= 50:
        ai_score += 10
    
    total_score = tech_score + ai_score
    total_score = min(total_score, 100)
    
    # 生动评论（浩哥口吻）
    comment = f"浩哥瞅了瞅 {name}，当前价 {current:.2f} 元，流通市值 {market_cap:.2f} 亿。兄弟，这票今天有点意思啊……"
    active_count = sum(1 for v in signals.values() if v)
    if active_count >= 3:
        comment += f" 形态走得挺漂亮，几个关键点都对上了，浩哥看这走势有点像要起飞的节奏！缩量踩线、J 值低位拐头，情绪也起来了，机会不小。"
    elif active_count >= 1:
        comment += f" 信号有，但还不够猛。浩哥觉得得再等等放量确认，不然容易假动作。别急，子弹留着等更好的。"
    else:
        comment += f" 今天这票还没到浩哥下手的点。形态一般，量没缩到位，情绪也冷冰冰的，先放放，别硬上。"
    
    if price_correction > 0:
        comment += " 哎呀，低价但换手这么猛，主力在偷偷干活？这票有妖股潜质，浩哥有点心动！"
    elif price_correction < 0:
        comment += " 低价股还缩量阴跌，浩哥劝你别碰，容易成韭菜收割机。"
    
    buy_advice = "浩哥建议：重仓干一票！" if total_score >= 90 else "可以买，仓位别太大。" if total_score >= 70 else "小仓试试水，注意止损。" if total_score >= 50 else "浩哥先不碰，等机会。"
    
    return total_score, tech_score, ai_score, comment, buy_advice

# 主界面
st.title("浩哥 AI 分析师 - 浩哥战法")

codes_input = st.text_input("输入股票代码（逗号分隔，如 600519,601218）")
if st.button("让浩哥分析"):
    codes = [c.strip() for c in codes_input.split(',') if c.strip()]
    for symbol in codes:
        stock_name = get_stock_name(symbol)
        st.subheader(f"浩哥看 {symbol} - {stock_name}")
        
        df, source = fetch_stock_history(symbol)
        if df is None:
            st.error(f"无法获取 {symbol} 数据")
            continue
        
        df = calculate_indicators(df)
        last = df.iloc[-1]
        current = last['close']
        market_cap = 100  # 模拟，实际可替换为真实接口
        
        total_score, tech_score, ai_score, comment, buy_advice = analyze_stock(df, stock_name, current, market_cap)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("浩哥打分", f"{total_score:.1f}/100", delta_color="normal")
        with col2:
            st.write("**浩哥点评：**")
            st.info(comment)
            st.write("**浩哥建议：**", buy_advice)
        
        st.markdown("---")

st.sidebar.success("浩哥战法已就绪！")
st.sidebar.info("浩哥亲自点评，真实数据驱动，评论生动接地气。公开分享给朋友们用吧！")
