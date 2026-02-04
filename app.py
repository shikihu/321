import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import akshare as ak
from datetime import datetime

# --- 页面配置 ---
st.set_page_config(layout="wide", page_title="Z哥B1量化筛选")

# --- 1. 稳定版数据获取（腾讯接口） ---
def fetch_data_tencent(symbol):
    """从腾讯获取行情，云端比Akshare更稳"""
    try:
        prefix = 'sh' if symbol.startswith('6') else 'sz'
        # 抓取日线数据
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,260,qfq"
        res = requests.get(url, timeout=10).json()
        
        # 兼容腾讯复杂的返回格式
        data_root = res.get('data', {})
        stock_key = f"{prefix}{symbol}"
        if stock_key not in data_root:
            stock_key = "" # 某些时候返回空键名
            if not data_root.get(""): return None
            
        inner_data = data_root[stock_key]
        raw_data = inner_data.get('qfqday', []) or inner_data.get('day', [])
        
        if not raw_data: return None
        
        # 整理成 DataFrame
        df = pd.DataFrame([row[:6] for row in raw_data], 
                          columns=['date', 'open', 'close', 'high', 'low', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        for col in df.columns: df[col] = pd.to_numeric(df[col])
        return df
    except Exception as e:
        st.error(f"数据抓取失败 ({symbol}): {str(e)}")
        return None

# --- 2. 战法计算引擎（完全对应副图源码） ---
def calculate_zge_strategy(df):
    C = df['close']
    L = df['low']
    H = df['high']
    O = df['open']
    V = df['volume']

    # 均线系统
    df['white'] = C.ewm(span=9, adjust=False).mean().ewm(span=11, adjust=False).mean()
    e1 = C.ewm(span=7, adjust=False).mean().ewm(span=7, adjust=False).mean()
    e2 = C.ewm(span=14, adjust=False).mean().ewm(span=14, adjust=False).mean()
    e3 = C.ewm(span=28, adjust=False).mean().ewm(span=28, adjust=False).mean()
    e4 = C.ewm(span=56, adjust=False).mean().ewm(span=56, adjust=False).mean()
    df['yellow'] = (e1 + e2 + e3 + e4) / 4
    df['bbi'] = (C.rolling(3).mean() + C.rolling(6).mean() + C.rolling(12).mean() + C.rolling(24).mean()) / 4

    # KDJ & RSI (完全对应源码)
    low_list = L.rolling(9).min()
    high_list = H.rolling(9).max()
    rsv = (C - low_list) / (high_list - low_list).replace(0, 1) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    lc = C.shift(1)
    temp1 = (C - lc).clip(lower=0).rolling(3).mean()
    temp2 = (C - lc).abs().rolling(3).mean()
    df['rsi'] = (temp1 / temp2.replace(0, 1)) * 100

    # 缩量标准
    v_hhv20 = V.rolling(20).max()
    v_hhv50 = V.rolling(50).max()
    df['is_适当缩量'] = (V < v_hhv20 * 0.618) | (V < v_hhv50 / 3)
    df['is_回踩缩量'] = (V < v_hhv20 * 0.45) | (V < v_hhv50 / 3)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 持股分数 (0-5分逻辑)
    score = 5
    if last['close'] < prev['close']: score -= 1
    if last['close'] < prev['close'] and last['volume'] > prev['volume']: score -= 1
    if last['close'] < last['white']: score -= 1
    if last['J'] < last['K']: score -= 1
    if last['white'] < prev['white']: score -= 1

    # B1类型识别
    b1_types = []
    dist_white = abs(last['close'] - last['white']) / last['white'] * 100
    if dist_white <= 2 and last['is_回踩缩量']: b1_types.append("回踩白线B1")
    if last['J'] < 13 and last['is_适当缩量']: b1_types.append("原始B1")
    if last['rsi'] < 23 and last['is_适当缩量']: b1_types.append("超卖缩量B1")

    return b1_types, score

# --- 3. 界面逻辑 ---
st.title("🛡️ Z哥 AI 分析师 - B1 战法深度筛选")

# 获取热点（带报错提示）
with st.sidebar:
    st.write("### 🔥 今日热点识别")
    try:
        hot_df = ak.stock_board_industry_name_em().sort_values("今日涨跌幅", ascending=False).head(5)
        hot_sectors = hot_df['板块名称'].tolist()
        st.success(", ".join(hot_sectors))
    except:
        st.warning("暂未抓取到板块热点")
        hot_sectors = []

codes_input = st.text_area("输入股票代码（000008, 601218等）", "000008")

if st.button("开始 AI 筛选完美图形"):
    codes = [c.strip() for c in codes_input.replace('\n', ',').split(',') if c.strip()]
    
    if not codes:
        st.error("请输入至少一个股票代码")
    else:
        for code in codes:
            with st.status(f"正在分析 {code}...", expanded=True) as status:
                df = fetch_data_tencent(code)
                if df is not None:
                    b1_list, score = calculate_zge_strategy(df)
                    
                    st.write(f"**分析完成！持股分数: {score} | 类型: {', '.join(b1_list) if b1_list else '等待信号'}**")
                    
                    # 绘图
                    df_p = df.iloc[-60:]
                    fig = go.Figure(data=[go.Candlestick(x=df_p.index, open=df_p['open'], high=df_p['high'], low=df_p['low'], close=df_p['close'])])
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['white'], name="白线", line=dict(color='white')))
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['yellow'], name="黄线", line=dict(color='yellow')))
                    fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=400)
                    st.plotly_chart(fig)
                    
                    status.update(label=f"{code} 分析完毕", state="complete")
                else:
                    status.update(label=f"{code} 数据获取失败", state="error")
