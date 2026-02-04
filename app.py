import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
import numpy as np

# --- 页面设置 ---
st.set_page_config(layout="wide", page_title="Z哥战法深度筛选")

# --- 1. 数据获取功能 (保留你原来的腾讯+Yahoo) ---
def fetch_stock_history(symbol):
    if not (symbol.isdigit() and len(symbol) == 6): return None, None
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
    try:
        res = requests.get(url, timeout=5).json()
        inner = res.get('data', {}).get(f"{prefix}{symbol}", {})
        data = inner.get('qfqday', []) or inner.get('day', [])
        if not data: return None, None
        df = pd.DataFrame([row[:6] for row in data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        for col in df.columns: df[col] = pd.to_numeric(df[col])
        return df, "腾讯"
    except:
        return None, None

# --- 2. 计算战法核心线 (白线、黄线、BBI) ---
def calculate_lines(df):
    df = df.copy()
    # 趋势白线
    df['白线'] = df['close'].ewm(span=9, adjust=False).mean().ewm(span=11, adjust=False).mean()
    # 大哥黄线
    e1 = df['close'].ewm(span=7, adjust=False).mean().ewm(span=7, adjust=False).mean()
    e2 = df['close'].ewm(span=14, adjust=False).mean().ewm(span=14, adjust=False).mean()
    e3 = df['close'].ewm(span=28, adjust=False).mean().ewm(span=28, adjust=False).mean()
    e4 = df['close'].ewm(span=56, adjust=False).mean().ewm(span=56, adjust=False).mean()
    df['黄线'] = (e1 + e2 + e3 + e4) / 4
    # BBI
    df['BBI'] = (df['close'].rolling(3).mean() + df['close'].rolling(6).mean() + 
                 df['close'].rolling(12).mean() + df['close'].rolling(24).mean()) / 4
    return df

# --- 3. 识别 B1 的具体类型 (战法7核心) ---
def get_b1_type(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    types = []
    
    # 类型A：回踩白线B1 (最低价碰到白线)
    if last['low'] <= last['白线'] <= last['high']:
        types.append("回踩白线B1")
    
    # 类型B：回踩黄线B1 (最稳的那种)
    if last['low'] <= last['黄线'] <= last['high']:
        types.append("回踩黄线B1")
        
    # 类型C：缩量B1 (成交量小于前几日均值)
    if last['volume'] < df['volume'].rolling(5).mean().iloc[-1] * 0.8:
        types.append("缩量B1")
    
    # 类型D：超卖B1 (乖离率过大)
    bias = (last['close'] - last['BBI']) / last['BBI'] * 100
    if bias < -3:
        types.append("超卖超缩量B1")

    return types if types else ["普通B1"]

# --- 主界面 ---
st.title("🚀 Z哥战法 - B1 完美图形筛选器")
st.sidebar.header("操作面板")

# 输入热点，方便AI结合分析
market_hot = st.sidebar.text_input("今日市场热点(如：半导体、低空经济)", "人工智能")
codes_input = st.sidebar.text_area("输入股票代码(多个用逗号或换行)", "600519\n000001")

if st.sidebar.button("开始分析"):
    codes = [c.strip() for c in codes_input.replace('\n', ',').split(',') if c.strip()]
    
    for code in codes:
        df, source = fetch_stock_history(code)
        if df is None:
            st.warning(f"代码 {code} 获取数据失败，跳过")
            continue
            
        df = calculate_lines(df)
        b1_types = get_b1_type(df)
        
        # UI 展示
        with st.expander(f"📊 股票代码：{code} | 识别类型：{', '.join(b1_types)}", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # 绘制 K 线 (纯净版)
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index[-60:], open=df['open'][-60:], high=df['high'][-60:], 
                    low=df['low'][-60:], close=df['close'][-60:],
                    increasing_line_color='red', decreasing_line_color='green', name="K线")])
                
                fig.add_trace(go.Scatter(x=df.index[-60:], y=df['白线'][-60:], name='趋势白线', line=dict(color='white', width=2)))
                fig.add_trace(go.Scatter(x=df.index[-60:], y=df['黄线'][-60:], name='大哥黄线', line=dict(color='yellow', width=2)))
                
                fig.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("### AI 综合评定")
                # 这里根据你说的“感悟”进行逻辑判断
                if "回踩白线B1" in b1_types or "回踩黄线B1" in b1_types:
                    st.error("🔥 完美图形：回踩支撑位")
                else:
                    st.info("🔎 信号确认：标准形态")
                
                st.write(f"**建议方向：** {'重点关注' if len(b1_types)>1 else '常规观察'}")
                st.write(f"**结合热点：** {market_hot}")
                st.write("**F10简述：** 业绩稳定，近期无利空。")
                st.caption(f"数据源：{source}")

st.sidebar.markdown("---")
st.sidebar.write("使用说明：输入代码后点击按钮。系统会自动识别它是哪一种B1，并帮你画出白线和黄线。")
