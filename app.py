import streamlit as st
import pandas as pd
import requests
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# ==========================================
# 数据获取
# ==========================================
@st.cache_data(ttl=300)
def get_real_time_price(symbol, df=None):
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers, timeout=3)
        parts = r.text.split('"')[1].split(',')
        price = float(parts[3])
        if price > 0:
            return price, "实时价"
    except:
        pass
    if df is not None and not df.empty:
        return df['close'].iloc[-1], "(非交易时间/最近收盘价)"
    return 0.0, "无数据"

@st.cache_data(ttl=3600)
def fetch_history_data(symbol):
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            key = f"{prefix}{symbol}"
            qt_data = data.get('data', {}).get(key, {})
            day_data = qt_data.get('qfqday', qt_data.get('day', []))
            if day_data:
                df = pd.DataFrame([row[:6] for row in day_data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.apply(pd.to_numeric, errors='coerce')
                return calculate_indicators(df)
    except:
        pass

    # AkShare 兜底
    try:
        end_date = datetime.datetime.now().strftime("%Y%m%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        if not df.empty:
            df = df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'})
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return calculate_indicators(df)
    except:
        pass

    return None

def get_stock_name(symbol):
    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        return df[df['项目'] == '股票简称']['值'].values[0]
    except:
        return symbol

@st.cache_data(ttl=1800)
def get_money_flow(symbol):
    try:
        market = "sh" if symbol.startswith('6') else "sz"
        flow = ak.stock_individual_fund_flow(stock=symbol, market=market)
        if not flow.empty:
            val = flow.iloc[0]['主力净流入-净额']
            return val / 100000000
    except:
        pass
    return 0.0

# ==========================================
# 技术指标计算
# ==========================================
def calculate_indicators(df):
    if df is None or len(df) < 5:
        return df
   
    df = df.copy()
   
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
   
    df['趋势白线'] = df['close'].ewm(span=9, adjust=False).mean().ewm(span=11, adjust=False).mean()
    df['大哥黄线'] = (df['close'].ewm(span=7, adjust=False).mean().ewm(span=7, adjust=False).mean() +
                       df['close'].ewm(span=14, adjust=False).mean().ewm(span=14, adjust=False).mean() +
                       df['close'].ewm(span=28, adjust=False).mean().ewm(span=28, adjust=False).mean() +
                       df['close'].ewm(span=56, adjust=False).mean().ewm(span=56, adjust=False).mean()) / 4
   
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = exp1 - exp2
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = 2 * (df['dif'] - df['dea'])
   
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
   
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['rsi'] = 100 - (100 / (1 + rs))
   
    df['vol_max5'] = df['volume'].rolling(5).mean()
    df['vol_max10'] = df['volume'].rolling(10).mean()
    df['vol_max20'] = df['volume'].rolling(20).max()
    df['vol_max50'] = df['volume'].rolling(50).max()
   
    df.fillna(method='bfill', inplace=True)
    return df

# ==========================================
# K线图
# ==========================================
def plot_kline(df, symbol, name):
    df = df.iloc[-120:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='white', width=1), name='白线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='yellow', width=1.5), name='大哥线'), row=1, col=1)
    colors = ['red' if row['open'] < row['close'] else 'green' for i, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='成交量'), row=2, col=1)
    fig.update_layout(title=f"{name} ({symbol})", height=500, xaxis_rangeslider_visible=False, plot_bgcolor='#1e1e1e', paper_bgcolor='#0e1117', font=dict(color='white'))
    return fig

# ==========================================
# 核心分析逻辑（方案A：B1精髓优先 + 共振乘数）
# ==========================================
def analyze_stock(df, name, current, symbol, money_flow):
    if df is None or len(df) < 20:
        return 0.0, f"数据不足", "观望"
   
    last = df.iloc[-1]
   
    def safe_get(col, default=0.0):
        return last[col] if col in last else default
   
    # 变量准备
    trend_white = safe_get('趋势白线', last['close'])
    brother_yellow = safe_get('大哥黄线', last['close'])
    vol = safe_get('volume')
    vol5_mean = safe_get('vol_max5')
    vol10_mean = safe_get('vol_max10')
    vol20_max = safe_get('vol_max20')
    vol50_max = safe_get('vol_max50')
   
    # 缩量系列
    shrink = (vol < vol20_max * 0.416) or (vol < vol50_max / 3)
    proper_shrink = (vol < vol20_max * 0.618)
    super_shrink = (vol < vol20_max * 0.25)
   
    # 趋势判定
    do_up_trend = (last['close'] > brother_yellow)
   
    # 回踩判定
    dist_white = abs(last['close'] - trend_white) / last['close'] * 100
    back_white = dist_white < 2.0
   
    j_val = safe_get('J')
   
    # --- B1 精髓专用加成 ---
    b1_bonus = 0.0
    # 放量回调阴线（跌幅>3% + 放量）
    if safe_get('当日涨跌幅', 0) < -3 and vol > vol5_mean * 1.5:
        b1_bonus += 15
    # J超卖
    if j_val < -5:
        b1_bonus += 12
    # 趋势向上回调
    if do_up_trend and back_white:
        b1_bonus += 10
   
    # --- 基础信号判定 ---
    signals = {}
   
    if do_up_trend and j_val < 14 and shrink:
        signals['浩哥缩量战法'] = True
    if do_up_trend and j_val < 14 and super_shrink:
        signals['浩哥极缩战法'] = True
    if do_up_trend and back_white and shrink and j_val < 30:
        signals['浩哥白线战法'] = True
    if last['close'] > safe_get('MA20') * 1.05 and proper_shrink and j_val < 35 and back_white:
        signals['浩哥超级战法'] = True
    if trend_white > brother_yellow and j_val < 13 and proper_shrink:
        signals['浩哥1.0战法'] = True
   
    # 权重
    weights = {
        '浩哥超级战法': 25.0,
        '浩哥极缩战法': 22.0,
        '浩哥白线战法': 18.0,
        '浩哥1.0战法': 15.0,
        '浩哥缩量战法': 5.0
    }
   
    base_tech = 0.0
    triggered = []
    for sig, active in signals.items():
        if active:
            base_tech += weights.get(sig, 0)
            triggered.append(sig)
   
    # 共振乘数（核心加成）
    num_signals = len(triggered)
    if num_signals >= 4:
        multiplier = 2.0
    elif num_signals == 3:
        multiplier = 1.6
    elif num_signals == 2:
        multiplier = 1.3
    else:
        multiplier = 1.0
   
    tech_score = base_tech * multiplier + b1_bonus
    tech_score = min(80, tech_score)
   
    # 浩哥评分（资金 + 低价复活）
    hao_score = 0.0
    if money_flow > 0.5: hao_score += 20
    elif money_flow > 0: hao_score += 10
    elif money_flow < -0.5: hao_score -= 10
   
    if current < 12:
        if money_flow > 0 or j_val < -5:
            hao_score += 10
        else:
            hao_score -= 2
   
    hao_score = min(40, max(hao_score, -10))
   
    total_score = tech_score + hao_score
    total_score = min(120, total_score)
   
    # 技术指标话术
    obs = []
    macd = safe_get('macd')
    if macd > 0: obs.append("MACD柱线翻红，动能修复")
    rsi = safe_get('rsi')
    if rsi > 60: obs.append("RSI超买，注意回调压力")
    if shrink: obs.append("量能萎缩，缩量调整形态")
    if last['MA5'] > last['MA20'] > last['MA60']:
        obs.append("均线多头排列，趋势完整")
   
    obs_text = "；".join(obs) + "。" if obs else "量价关系中性。"
   
    # 评论
    comment = f"浩哥对 {name} 的综合判断：当前价 {current:.2f} 元。\n\n"
    if triggered:
        comment += f"浩哥检测到关键信号：**{', '.join(triggered)}**（共{num_signals}个，共振乘数 {multiplier:.1f}）\n\n"
    else:
        comment += "浩哥今天未检测到核心信号。\n\n"
       
    comment += f"【技术面评分】 {tech_score:.1f}/80 【浩哥评分】 {hao_score:.1f}/40 【浩哥综合打分】 {total_score:.1f}/120\n\n"
    comment += f"技术面观察：{obs_text}\n\n"
    comment += f"💰 资金面：主力净流入 {money_flow:.2f} 亿。\n"
   
    if total_score >= 90:
        advice = "浩哥建议：重仓出击，多信号共振 + 资金强势！"
    elif total_score >= 70:
        advice = "浩哥建议：中仓建仓，值得博弈。"
    elif total_score >= 50:
        advice = "浩哥建议：轻仓观察，等待确认。"
    else:
        advice = "浩哥建议：暂时回避，保护本金。"
   
    return total_score, comment, advice

# ==========================================
# 主界面
# ==========================================
st.set_page_config(page_title="浩哥战法", layout="wide")
st.title("🚀 浩哥战法量化终端 v4.0 (方案A - 共振加成版)")

codes_input = st.text_input("输入股票代码 (逗号分隔)", "600519,000001")
if st.button("开始挖掘"):
    codes = [c.strip() for c in codes_input.replace('，', ',').split(',') if c.strip()]
    for symbol in codes:
        with st.spinner(f"浩哥正在深度分析 {symbol}..."):
            df = fetch_history_data(symbol)
            name = get_stock_name(symbol)
            current_price, price_source = get_real_time_price(symbol, df)
            money_flow = get_money_flow(symbol)
           
            if df is not None:
                score, comment, advice = analyze_stock(df, name, current_price, symbol, money_flow)
               
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.markdown("### 浩哥打分")
                    st.markdown(f"<h1 style='font-size: 60px;'>{score:.1f}/120</h1>", unsafe_allow_html=True)
                with c2:
                    st.info(comment + f"\n\n价格来源: {price_source}")
                    if score >= 70:
                        st.success(advice)
                    else:
                        st.warning(advice)
               
                with st.expander("查看 K线图"):
                    fig = plot_kline(df, symbol, name)
                    st.plotly_chart(fig, use_container_width=True)
               
                st.markdown("---")
            else:
                st.error(f"{symbol} 数据拉取失败")
