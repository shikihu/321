import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import numpy as np
import akshare as ak
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. 技术指标计算（保留 Gemini 版，完整）
# ==========================================
def calculate_indicators(df):
    if df is None or len(df) < 20:
        return df
   
    df = df.copy()
   
    # MA
    df['MA5'] = df['close'].rolling(window=5).mean() # 白线
    df['MA20'] = df['close'].rolling(window=20).mean() # 黄线
    df['MA60'] = df['close'].rolling(window=60).mean() # 生命线
   
    # BBI
    ma3 = df['close'].rolling(window=3).mean()
    ma6 = df['close'].rolling(window=6).mean()
    ma12 = df['close'].rolling(window=12).mean()
    ma24 = df['close'].rolling(window=24).mean()
    df['BBI'] = (ma3 + ma6 + ma12 + ma24) / 4
   
    # VOL5
    df['VOL5'] = df['volume'].rolling(window=5).mean()
   
    # KDJ
    low_list = df['low'].rolling(window=9, min_periods=9).min()
    high_list = df['high'].rolling(window=9, min_periods=9).max()
    rsv = (df['close'] - low_list) / (high_list - low_list).replace(0, 1) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
   
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = ema12 - ema26
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
   
    return df

# ==========================================
# 2. 数据获取（保留 Gemini 版 + 修复）
# ==========================================
def get_real_time_price(symbol):
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
    try:
        r = requests.get(url, timeout=5)
        parts = r.text.split('"')[1].split(',')
        if len(parts) >= 4:
            return float(parts[3])
    except:
        pass
    return 0.0

def fetch_history_data(symbol):
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
    try:
        r = requests.get(url, timeout=5).json()
        data = r.get('data', {}).get(f"{prefix}{symbol}", {}).get('qfqday', [])
        if not data:
            return None
       
        df = pd.DataFrame([row[:6] for row in data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.apply(pd.to_numeric)
        df = calculate_indicators(df)
        return df
    except:
        return None

@st.cache_data(ttl=1800)
def get_market_info(symbol):
    name = symbol
    news = []
    flow = 0.0
   
    try:
        stock_info = ak.stock_individual_info_em(symbol=symbol)
        name = stock_info[stock_info['项目'] == '股票简称']['值'].values[0]
       
        news_df = ak.stock_news_em(symbol=symbol)
        news = news_df.head(3)[['标题', '发布时间']].to_dict('records')
       
        flow_df = ak.stock_individual_fund_flow(stock=symbol, market="sh" if symbol.startswith('6') else "sz")
        if not flow_df.empty:
            flow = flow_df.iloc[0]['主力净流入-净额'] / 100000000
    except:
        pass
       
    return name, news, flow

# ==========================================
# 3. 浩哥战法评分逻辑（精细化 + 小数 + 低价复活）
# ==========================================
def analyze_logic(df, current_price, symbol, name, money_flow):
    if df is None or len(df) < 20:
        return 0.0, "数据不足，浩哥没法算。", "观望"
   
    last = df.iloc[-1]
   
    # 核心指标
    close = last['close']
    vol = last['volume']
    ma5 = last['MA5']
    ma20 = last['MA20']
    ma60 = last['MA60']
    j_val = last['J']
    vol5 = last['VOL5']
    vol_ratio = vol / vol5 if vol5 > 0 else 0
   
    # 信号判定（基于你的权重）
    signals = []
    score = 0.0
   
    # 回踩超级B (25分)
    if vol_ratio < 0.6 and abs(close - ma20)/ma20 < 0.03 and j_val < -5:
        score += 25.0
        signals.append("回踩超级B")
   
    # 超卖超缩量B (22分)
    if vol_ratio < 0.5 and j_val < 0:
        score += 22.0
        signals.append("超卖超缩量B")
   
    # 回踩白线B (18分)
    if abs(close - ma5)/ma5 < 0.02 and ma5 > ma20:
        score += 18.0
        signals.append("回踩白线B")
   
    # 原始B1 (15分)
    if j_val < 10 and close > ma20:
        score += 15.0
        signals.append("原始B1")
   
    # 精细加分
    if j_val < 0:
        j_bonus = min(abs(j_val) * 0.3, 4.0)  # 每低1点 +0.3，上限4
        score += j_bonus
   
    # 低价股复活机制
    price_bonus = 0.0
    if close < 12:
        price_bonus = -5.0
        is_active = (vol_ratio > 1.2) or (close > ma20 * 1.02)  # 简化条件
        if is_active:
            price_bonus = 3.0
        score += price_bonus
   
    # 资金流加分
    if money_flow > 1.0:
        score += 15.0
    elif money_flow > 0.1:
        score += 8.0
    elif money_flow < -0.5:
        score -= 10.0
   
    total_score = min(max(score, 0), 100.0)
   
    # 浩哥生动评论
    comment = f"浩哥瞅了瞅 {name}，现价 {close:.2f}。量比{vol_ratio:.2f}，J值{j_val:.1f}。"
    if price_bonus > 0:
        comment += " 低价但主力点火，浩哥觉得这妖股有戏！"
    elif price_bonus < 0:
        comment += " 低价还缩量阴跌，浩哥劝你别碰。"
   
    if total_score >= 85:
        comment += " 卧槽！这票数据炸裂，形态完美，资金抢筹，浩哥看这节奏要起飞了！"
        advice = "重仓干！Stop Loss -5%"
    elif total_score >= 70:
        comment += " 哎哟不错，触发优质信号，浩哥觉得可以搞点仓位试试。"
        advice = "中仓买入 3-5成"
    else:
        comment += " 形态一般，浩哥看不上眼，先放放。"
        advice = "空仓观望"
   
    if money_flow > 0:
        comment += f" 主力净流入 {money_flow:.2f} 亿，真金白银在挺！"
    else:
        comment += f" 主力净流出 {abs(money_flow):.2f} 亿，小心庄家跑路。"
   
    return total_score, comment, advice

# ==========================================
# K线绘图（保留 Gemini 版）
# ==========================================
def plot_kline(df, symbol, name):
    df = df.iloc[-120:]
   
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])
   
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='K线', increasing_line_color='red', decreasing_line_color='green'
    ), row=1, col=1)
   
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='white', width=1), name='白线(MA5)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='yellow', width=1.5), name='黄线(大哥线)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue', width=1), name='生命线(MA60)'), row=1, col=1)
   
    colors = ['red' if row['open'] < row['close'] else 'green' for i, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='成交量'), row=2, col=1)
   
    fig.update_layout(
        title=f"{name} ({symbol}) - 浩哥专用图表",
        yaxis_title='价格',
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#0e1117',
        font=dict(color='white')
    )
    return fig

# ==========================================
# Streamlit 主界面
# ==========================================
st.set_page_config(page_title="浩哥战法", layout="wide", page_icon="🎯")
st.title("🚀 浩哥战法量化终端")

with st.sidebar:
    st.title("📌 Z哥六步法（背熟！）")
    st.markdown(""" 
1️⃣ 择时：周日看大盘温度  
2️⃣ 选股：强势基因+题材热  
3️⃣ 买点：B1首踩 或 B2主升  
4️⃣ 持仓：等利润垫，不折腾  
5️⃣ 卖点：破位/高潮/情绪退潮  
6️⃣ 复盘：每笔交易必复盘  
""")
    st.markdown("**💡 心态**：沉没成本不决策，戒骄戒躁，珍惜子弹！")

codes_input = st.text_input("🔍 输入股票代码（逗号分隔）", "600519,000001")
if st.button("开始挖掘"):
    codes = [c.strip() for c in codes_input.split(',') if c.strip()]
    for symbol in codes:
        with st.spinner(f"浩哥正在分析 {symbol}..."):
            df = fetch_history_data(symbol)
            name = get_stock_name(symbol)
            current_price = get_real_time_price(symbol)
            news, money_flow = get_market_info(symbol)[1], get_market_info(symbol)[2]
           
            if df is not None:
                score, comment, advice = analyze_logic(df, current_price, symbol, name, money_flow)
               
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.metric("浩哥打分", f"{score:.1f}/100")
                with c2:
                    st.info(comment)
                    st.success(f"**建议：** {advice}")
               
                with st.expander("K线图"):
                    fig = plot_kline(df, symbol, name)
                    st.plotly_chart(fig, use_container_width=True)
               
                st.markdown("---")
            else:
                st.error(f"{symbol} 数据拉取失败")
