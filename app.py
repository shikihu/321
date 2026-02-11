import streamlit as st
import pandas as pd
import requests
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import re

# ==========================================
# 数据服务（稳定版）
# ==========================================
@st.cache_data(ttl=300)
def get_real_time_price(symbol, df=None):
    symbol = str(symbol)
    prefix = 'sh' if symbol.startswith(('6', '9')) else 'sz'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
        r = requests.get(url, headers=headers, timeout=3)
        parts = r.text.split('"')[1].split(',')
        if len(parts) > 3 and float(parts[3]) > 0:
            return float(parts[3]), "实时价"
    except:
        pass
    if df is not None and not df.empty:
        return df['close'].iloc[-1], "(非交易时间/最近收盘价)"
    return 0.0, "无数据"

@st.cache_data(ttl=3600)
def fetch_history_data(symbol):
    symbol = str(symbol)
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
        end = datetime.datetime.now().strftime("%Y%m%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
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
        df = ak.stock_individual_info_em(symbol=str(symbol))
        return df[df['项目'] == '股票简称']['值'].values[0]
    except:
        return symbol

@st.cache_data(ttl=1800)
def get_money_flow(symbol):
    try:
        market = "sh" if str(symbol).startswith('6') else "sz"
        flow = ak.stock_individual_fund_flow(stock=str(symbol), market=market)
        if not flow.empty:
            return flow.iloc[0]['主力净流入-净额'] / 100000000
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
   
    df['vol_max20'] = df['volume'].rolling(20).max()
    df['vol_ma5'] = df['volume'].rolling(5).mean()
   
    df = df.ffill().bfill()
    return df

# ==========================================
# K线图（大块显示）
# ==========================================
def plot_kline(df, symbol, name):
    df = df.iloc[-120:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='white', width=1), name='白线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='yellow', width=1.5), name='大哥线'), row=1, col=1)
    colors = ['red' if row['open'] < row['close'] else 'green' for i, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='成交量'), row=2, col=1)
    fig.update_layout(title=f"{name} ({symbol}) - 浩哥专用图表", height=600, xaxis_rangeslider_visible=True, 
                      plot_bgcolor='#1e1e1e', paper_bgcolor='#0e1117', font=dict(color='white'))
    return fig

# ==========================================
# 评分系统（还原详细技术面观察 + 浩哥风格评论）
# ==========================================
def analyze_stock(df, name, current, symbol, money_flow):
    if df is None or len(df) < 20:
        return 0.0, "数据不足", "观望", "#888"
   
    last = df.iloc[-1]
   
    # 基础信号判定（简化版，保留核心）
    triggered = []
    tech_score = 0.0
   
    j_val = last['J']
    dist_white = abs(last['close'] - last['MA5']) / last['close'] * 100 if last['close'] > 0 else 999
   
    if j_val < 0 and last['volume'] < last['vol_max20'] * 0.5:
        triggered.append("浩哥缩量战法")
        tech_score += 15.0
    if dist_white < 2.0 and last['close'] > last['MA20']:
        triggered.append("浩哥白线战法")
        tech_score += 18.0
   
    # 浩哥评分（资金面）
    hao_score = 0.0
    if money_flow > 0.5: hao_score = 15.0
    elif money_flow > 0: hao_score = 5.0
   
    total_score = min(100, tech_score + hao_score)
   
    # 技术面详细观察（还原你喜欢的详细风格）
    obs_lines = []
    macd = last['macd'] if 'macd' in last else 0
    if macd > 0:
        obs_lines.append("MACD柱线翻红，短期动能有修复迹象")
    else:
        obs_lines.append("MACD绿柱状态，动能偏弱")
   
    if last['volume'] < last['vol_max20'] * 0.6:
        obs_lines.append("量能持续萎缩，属于典型缩量调整形态")
    else:
        obs_lines.append("成交量温和或放大，资金分歧较大")
   
    if last['MA5'] > last['MA20'] > last['MA60']:
        obs_lines.append("短期均线多头排列，趋势结构仍保持完整")
    elif last['close'] < last['MA20']:
        obs_lines.append("股价跌破大哥黄线，注意风险")
   
    obs_text = "；".join(obs_lines) + "。" if obs_lines else "量价关系中性。"
   
    # 浩哥风格评论（还原血性、狠劲儿）
    comment = f"浩哥对{name}的综合判断：当前价{current:.2f}元。\n\n"
   
    if triggered:
        comment += f"浩哥检测到关键信号：{' + '.join(triggered)}\n\n"
    else:
        comment += "浩哥今天未检测到关键信号，形态未到最佳点。\n\n"
   
    comment += f"【技术面评分】{tech_score:.1f}/70 【浩哥评分】{hao_score:.1f}/30 【浩哥综合打分】{total_score:.1f}/100\n\n"
   
    comment += f"技术面观察：{obs_text}\n\n"
   
    comment += f"资金面：主力净流入{money_flow:.2f}亿。浩哥认为当前风险大于机会，形态和情绪均未到位，短期不宜重仓。"
   
    if total_score >= 80:
        advice = "浩哥喊单：机会显著，重仓干！"
    elif total_score >= 60:
        advice = "浩哥建议：形态还行，可以轻仓试试。"
    else:
        advice = "浩哥建议：暂时回避，保护本金，等更清晰信号。"
   
    color = "#d32f2f" if total_score >= 80 else "#ff5722" if total_score >= 60 else "#757575"
   
    return total_score, comment, advice, color

# ==========================================
# 主界面（简洁版）
# ==========================================
st.set_page_config(page_title="浩哥战法", layout="wide")
st.title("浩哥战法量化终端 v3.0")

codes_input = st.text_input("输入股票代码（逗号分隔）", "600519,000001")
if st.button("开始分析"):
    codes = [c.strip() for c in codes_input.split(',') if c.strip()]
    for symbol in codes:
        with st.spinner(f"浩哥正在分析 {symbol}..."):
            df = fetch_history_data(symbol)
            name = get_stock_name(symbol)
            current, price_source = get_real_time_price(symbol, df)
            money = get_money_flow(symbol)
           
            if df is not None:
                score, comment, advice, color = analyze_stock(df, name, current, symbol, money)
               
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.markdown(f"<h2 style='color: {color}'>{score:.1f}/100</h2>", unsafe_allow_html=True)
                    st.caption(f"{name} ({symbol})")
                with c2:
                    st.markdown(comment)
                    st.markdown(f"**浩哥建议：** {advice}")
                    st.caption(f"价格来源：{price_source}")
               
                with st.expander("查看 K线图"):
                    fig = plot_kline(df, symbol, name)
                    st.plotly_chart(fig, use_container_width=True)
               
                st.markdown("---")
            else:
                st.error(f"{symbol} 数据拉取失败")
