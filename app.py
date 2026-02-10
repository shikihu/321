import streamlit as st
import pandas as pd
import requests
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# ==========================================
# 数据获取（加强版 + 兼容腾讯接口变异格式）
# ==========================================
@st.cache_data(ttl=300)
def get_real_time_price(symbol, df=None):
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
    try:
        r = requests.get(url, timeout=3)
        text = r.text.strip()
        if text.startswith('var hq_str_'):
            parts = text.split('"')[1].split(',')
            price = float(parts[3])
            if price > 0:
                return price, "实时价"
    except:
        pass
    # 兜底用历史收盘价
    if df is not None and len(df) > 0:
        return df['close'].iloc[-1], "(非交易时间/最近收盘价)"
    return 0.0, "暂无数据"

@st.cache_data(ttl=600)
def fetch_history_data(symbol):
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()  # 先检查 HTTP 状态码
        resp_json = r.json()
        
        # 腾讯接口返回结构可能变异，兼容处理
        data_key = f"{prefix}{symbol}"
        if data_key in resp_json.get('data', {}):
            qfqday = resp_json['data'][data_key].get('qfqday', [])
        elif 'data' in resp_json and isinstance(resp_json['data'], list):
            # 某些股票返回 list 格式
            qfqday = resp_json['data']
        else:
            qfqday = []
        
        if not qfqday:
            st.warning(f"{symbol} 返回空数据")
            return None
        
        df = pd.DataFrame(qfqday, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(how='all', inplace=True)  # 清理全空行
        return calculate_indicators(df)
    except Exception as e:
        st.error(f"{symbol} 数据拉取失败: {str(e)}")
        return None

def get_stock_name(symbol):
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
    try:
        r = requests.get(url, timeout=3)
        text = r.text.strip()
        if text.startswith('var hq_str_'):
            parts = text.split('"')[1].split(',')
            if len(parts) >= 2:
                return parts[0].strip()
    except:
        pass
    return symbol

@st.cache_data(ttl=1800)
def get_stock_news(symbol):
    try:
        news = ak.stock_news_em(symbol=symbol)
        return news.head(3)[['标题', '发布时间', '来源']].to_dict('records')
    except:
        return []

@st.cache_data(ttl=1800)
def get_money_flow(symbol):
    try:
        flow = ak.stock_individual_fund_flow(stock=symbol, market="sh" if symbol.startswith('6') else "sz")
        if not flow.empty:
            return flow.iloc[0]['主力净流入-净额'] / 100000000
        return 0.0
    except:
        return 0.0

# ==========================================
# 技术指标计算（安全版）
# ==========================================
def calculate_indicators(df):
    if df is None or len(df) < 5:
        return df
    
    df = df.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    
    df['MA5']  = df['close'].rolling(5, min_periods=1).mean()
    df['MA20'] = df['close'].rolling(20, min_periods=1).mean()
    df['MA60'] = df['close'].rolling(60, min_periods=1).mean()
    
    df['MA5']  = df['MA5'].fillna(df['close'])
    df['MA20'] = df['MA20'].fillna(df['close'])
    df['MA60'] = df['MA60'].fillna(df['close'])
    
    return df

# ==========================================
# K线图（安全版）
# ==========================================
def plot_kline(df, symbol, name):
    df = df.iloc[-120:].copy()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='K线', increasing_line_color='red', decreasing_line_color='green'
    ), row=1, col=1)
    
    for ma, color, name in [('MA5', 'white', '白线(MA5)'), ('MA20', 'yellow', '黄线(大哥线)'), ('MA60', 'blue', '生命线(MA60)')]:
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ma], line=dict(color=color, width=1 if ma == 'MA5' else 1.5 if ma == 'MA20' else 1), name=name), row=1, col=1)
    
    colors = ['red' if row['open'] < row['close'] else 'green' for i, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='成交量'), row=2, col=1)
    
    fig.update_layout(
        title=f"{name} ({symbol}) - 浩哥专用图表",
        yaxis_title='价格',
        xaxis_rangeslider_visible=True,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#0e1117',
        font=dict(color='white')
    )
    return fig

# ==========================================
# 浩哥战法评分（简化版，方便调试）
# ==========================================
def analyze_stock(df, name, current, symbol, money_flow):
    if df is None or len(df) < 20:
        return 0.0, f"浩哥看 {name} 数据不足，无法分析。", "浩哥建议：暂缓操作。"
    
    last = df.iloc[-1]
    
    # 简单模拟信号（实际替换成你的公式）
    triggered_signals = []
    tech_score = 0.0
    if safe_get('close', 0) > safe_get('MA20', 0):
        tech_score += 10.0
        triggered_signals.append("浩哥趋势战法")
    
    hao_score = 0.0
    if money_flow > 0.5:
        hao_score += 15.0
    
    total_score = tech_score + hao_score
    
    comment = f"浩哥对 {name} 的综合判断：当前价 {current:.2f} 元。\n\n"
    if triggered_signals:
        comment += f"浩哥检测到关键信号：{ ' + '.join(triggered_signals) }\n\n"
    else:
        comment += "浩哥今天未检测到关键信号。\n\n"
    
    comment += f"【技术面评分】{tech_score:.1f}/70\n"
    comment += f"【浩哥评分】{hao_score:.1f}/30\n"
    comment += f"【浩哥综合打分】{total_score:.1f}/100\n\n"
    
    comment += f"💰 资金面：主力净流入 {money_flow:.2f} 亿。\n"
    
    advice = "浩哥建议观望。"
    
    return total_score, comment, advice

# ==========================================
# 主界面
# ==========================================
st.set_page_config(page_title="浩哥战法", layout="wide")
st.title("🚀 浩哥战法量化终端 v3.0 (修复版)")

codes_input = st.text_input("输入股票代码（逗号分隔）", "600519,002235,002501,002425,600545")
if st.button("开始挖掘"):
    codes = [c.strip() for c in codes_input.split(',') if c.strip()]
    for symbol in codes:
        with st.spinner(f"正在分析 {symbol}..."):
            df = fetch_history_data(symbol)
            name = get_stock_name(symbol)
            current_price, price_msg = get_real_time_price(symbol, df)
            money_flow = get_money_flow(symbol)
           
            if df is not None:
                score, comment, advice = analyze_stock(df, name, current_price, symbol, money_flow)
               
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.metric("浩哥打分", f"{score:.1f}/100")
                with c2:
                    st.info(f"{comment}\n\n价格来源：{price_msg}")
                    st.success(f"**浩哥建议：** {advice}")
               
                with st.expander("查看 K线图"):
                    fig = plot_kline(df, symbol, name)
                    st.plotly_chart(fig, use_container_width=True)
               
                st.markdown("---")
            else:
                st.error(f"{symbol} 数据拉取失败")
