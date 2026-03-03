import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import time
import socket
import numpy as np

# ==========================================
# 1. 基础配置
# ==========================================
socket.setdefaulttimeout(15)
st.set_page_config(
    page_title="浩哥战法量化终端 v11.0 ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 核心数据引擎
# ==========================================
def get_realtime_data(symbol):
    symbol = str(symbol).strip()
    prefix = 'sh' if symbol.startswith(('6', '9')) else 'sz'
    code = f"{prefix}{symbol}"
    try:
        url = f"http://qt.gtimg.cn/q={code}"
        r = requests.get(url, timeout=3)
        r.encoding = 'gbk'
        text = r.text
        if f'v_{code}="' in text:
            data_str = text.split('"')[1]
            parts = data_str.split('~')
            if len(parts) > 45:
                return {
                    'name': parts[1],
                    'price': float(parts[3]),
                    'turnover': float(parts[38]) if parts[38] else 0,
                    'pe': float(parts[39]) if parts[39] else 0,
                    'pb': float(parts[46]) if parts[46] else 0,
                    'change': float(parts[32]) if parts[32] else 0
                }
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def fetch_kline_data(symbol):
    symbol = str(symbol).strip()
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    try:
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
        r = requests.get(url, headers={'Connection': 'close'}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            key = f"{prefix}{symbol}"
            day_data = data.get('data', {}).get(key, {}).get('qfqday', [])
            if not day_data:
                day_data = data.get('data', {}).get(key, {}).get('day', [])
            if day_data:
                df = pd.DataFrame([row[:6] for row in day_data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.apply(pd.to_numeric, errors='coerce')
                return calculate_indicators(df)
    except:
        pass
    return None

# ==========================================
# 3. 核心算法 (全战法融合)
# ==========================================
def calculate_sma(series, period, weight=1):
    return series.ewm(alpha=weight/period, adjust=False).mean()

def calculate_indicators(df):
    if df is None or len(df) < 20: return df
    df = df.copy()
    
    # --- 基础均线 ---
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    # --- 浩哥趋势线 ---
    ema9 = df['close'].ewm(span=9, adjust=False).mean()
    df['趋势白线'] = ema9.ewm(span=11, adjust=False).mean()
    ema_vals = [df['close'].ewm(span=x, adjust=False).mean().ewm(span=x, adjust=False).mean() for x in [7,14,28,56]]
    df['大哥黄线'] = sum(ema_vals) / 4
    
    # --- KDJ ---
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # --- 浩哥砖型图 (核心) ---
    hhv4 = df['high'].rolling(4).max()
    llv4 = df['low'].rolling(4).min()
    range4 = (hhv4 - llv4).replace(0, 0.01)
    
    uar1a = (hhv4 - df['close']) / range4 * 100 - 90
    uar2a = calculate_sma(uar1a, 4, 1) + 100
    uar3a = (df['close'] - llv4) / range4 * 100
    uar4a = calculate_sma(uar3a, 6, 1)
    uar5a = calculate_sma(uar4a, 6, 1) + 100
    uar6a = uar5a - uar2a
    
    df['砖型图'] = np.where(uar6a > 4, uar6a - 4, 0)
    
    # 砖型信号
    df['砖型翻红'] = (df['砖型图'] > 0) & (df['砖型图'].shift(1) == 0) # XG起爆点
    df['砖型持续'] = (df['砖型图'] > df['砖型图'].shift(1)) & (df['砖型图'] > 0)

    # --- 浩哥全家桶战法 ---
    
    # 1. 量能
    df['vol_max20'] = df['volume'].rolling(20).max()
    df['极缩'] = (df['volume'] < df['vol_max20'] * 0.25)
    df['普通缩量'] = (df['volume'] < df['vol_max20'] * 0.45)
    
    # 2. 形态回踩
    dist_white = abs(df['close'] - df['趋势白线']) / df['close'] * 100
    dist_yellow = abs(df['close'] - df['大哥黄线']) / df['大哥黄线'] * 100
    
    df['回踩白线'] = (dist_white < 2.5) & (df['close'] > df['MA60'])
    df['回踩黄线'] = (dist_yellow < 2.5) & (df['close'] > df['MA60'])
    
    # 3. 战法信号
    # 白线战法: 回踩白线 + 缩量
    df['浩哥白线'] = df['回踩白线'] & df['普通缩量']
    
    # 黄线战法: 回踩黄线 + 缩量 (强支撑)
    df['浩哥黄线'] = df['回踩黄线'] & df['普通缩量']
    
    # 超级战法: 站稳MA20 + 缩量 + J值低
    df['浩哥超级'] = (df['close'] > df['MA20']) & df['普通缩量'] & (df['J'] < 35)
    
    # 极缩战法: 极缩 + 回踩 (左侧埋伏)
    df['浩哥极缩'] = df['极缩'] & (df['回踩白线'] | df['回踩黄线'])

    # --- 止损与目标位 ---
    # 止损: 取 MA20 和 黄线 的最大值作为支撑，下浮 3%
    df['技术支撑'] = df[['MA20', '大哥黄线']].max(axis=1)
    df['止损价'] = df['技术支撑'] * 0.97
    
    # 目标: 近期高点
    df['近期高点'] = df['high'].rolling(20).max()
    df['目标价'] = df['近期高点']

    df = df.ffill().bfill()
    return df

# ==========================================
# 4. 分析逻辑 (Grok建议融合)
# ==========================================
def analyze_stock_logic(code, info, df):
    if not info or df is None: return None
    
    last = df.iloc[-1]
    name = info['name']
    price = info['price']
    
    # --- 评分系统 (激进优化) ---
    score = 0
    signals = []
    
    # 1. 砖型图权重 (趋势确认)
    if last['砖型翻红']:
        score = 88 # 起爆点直接给高分
        signals.append("🧱 **砖型起爆** (趋势转强)")
    elif last['砖型持续']:
        score = 75
        signals.append("📈 **砖型持股** (动能增强)")
        
    # 2. 浩哥战法共振 (加分项)
    # 如果砖型起爆 + 白线战法 = 完美共振
    
    if last['浩哥白线']:
        score += 10
        signals.append("⚪ **白线回踩** (缩量确认)")
    
    if last['浩哥黄线']:
        score += 10
        signals.append("🟡 **黄线回踩** (强支撑)")
        
    if last['浩哥超级']:
        score += 10
        signals.append("⚡ **超级战法** (位置极佳)")
        
    if last['浩哥极缩']:
        # 极缩如果单独出现，给85分(左侧)；如果共振出现，再加分
        if score == 0: score = 85 
        else: score += 10
        signals.append("💎 **极致缩量** (洗盘结束)")
        
    # 3. 完美共振判断
    # 过去3天有极缩，今天砖型翻红 = 98分
    has_extreme_vol = df['极缩'].iloc[-4:-1].any() # 检查前几天
    if has_extreme_vol and last['砖型翻红']:
        score = 98
        signals.insert(0, "🚀 **完美共振** (极缩后起爆)")
        
    # 4. 减分项
    if last['close'] < last['MA60']:
        score -= 10
        signals.append("⚠️ **趋势偏弱** (均线下方)")

    score = min(99, max(0, score))
    
    # 建议
    advice = "观望"
    if score >= 90: advice = "S级买点 (全战法共振)"
    elif score >= 80: advice = "A级买点 (右侧起爆)"
    elif score >= 70: advice = "B级买点 (回踩低吸)"
    
    # 止损止盈文案
    stop_loss = last['止损价']
    target_price = last['目标价']
    risk_reward = (target_price - price) / (price - stop_loss + 0.01) # 盈亏比
    
    comment = f"**{name}** ({code}) 现价: {price}\n\n"
    comment += f"📡 **核心信号**: {' + '.join(signals) if signals else '无明显机会'}\n"
    comment += f"🛡️ **交易计划**: \n"
    comment += f"- 🛑 **止损位**: **{stop_loss:.2f}** (跌破离场)\n"
    comment += f"- 🎯 **目标位**: **{target_price:.2f}** (近期压力)\n"
    if risk_reward > 2:
        comment += f"- ⚖️ **盈亏比**: **{risk_reward:.1f}** (划算)\n"
    
    return {
        'code': code, 'name': name, 'score': score, 
        'comment': comment, 'advice': advice, 'df': df,
        'is_king': score >= 88
    }

# ==========================================
# 5. 主程序
# ==========================================
st.title("浩哥战法量化终端 v11.0 (全战法共振版)")
st.caption("🚀 核心升级：融合砖型图 + 白线/黄线/超级战法，实现多维共振打分。增加止损/目标位/盈亏比计算。")

codes_input = st.text_area("请输入股票代码", height=100)

if st.button("🚀 扫描共振买点"):
    codes = re.findall(r'\d{6}', codes_input)
    codes = list(set(codes))[:50]
    
    if not codes:
        st.warning("请输入代码")
    else:
        results = []
        bar = st.progress(0)
        
        for i, code in enumerate(codes):
            info = get_realtime_data(code)
            if info:
                df = fetch_kline_data(code)
                res = analyze_stock_logic(code, info, df)
                if res: results.append(res)
            bar.progress((i+1)/len(codes))
            time.sleep(0.05)
            
        results.sort(key=lambda x: x['score'], reverse=True)
        st.success(f"扫描完成！")
        
        for res in results:
            prefix = "🚀 [完美共振] " if res['score'] >= 95 else "👑 " if res['score'] >= 88 else ""
            with st.expander(f"{prefix}{res['name']} ({res['code']}) - {res['score']:.0f}分", expanded=res['is_king']):
                c1, c2 = st.columns([3, 1])
                with c1: st.markdown(res['comment'])
                with c2: 
                    if res['score'] >= 88: st.error(res['advice'])
                    elif res['score'] >= 70: st.success(res['advice'])
                    else: st.info(res['advice'])
                    
                if res['df'] is not None:
                    df_p = res['df'].iloc[-100:]
                    
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25])
                    
                    # 1. K线 + 浩哥双线 + B1箭头
                    fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['open'], high=df_p['high'], low=df_p['low'], close=df_p['close'], name='K线'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['趋势白线'], line=dict(color='white', width=1), name='白线'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['大哥黄线'], line=dict(color='yellow', width=1), name='黄线'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['止损价'], line=dict(color='red', width=1, dash='dot'), name='止损线'), row=1, col=1)
                    
                    # 标记买点 (只在分数高的地方画箭头)
                    buy_signals = df_p[df_p['砖型翻红'] | df_p['浩哥极缩']]
                    if not buy_signals.empty:
                        fig.add_trace(go.Scatter(
                            x=buy_signals.index, y=buy_signals['low']*0.98,
                            mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00e676'),
                            name='B1买点'
                        ), row=1, col=1)

                    # 2. 砖型图
                    brick_colors = ['red' if r['砖型图'] > 0 else 'green' for _, r in df_p.iterrows()]
                    fig.add_trace(go.Bar(x=df_p.index, y=df_p['砖型图'], marker_color=brick_colors, name='砖型图'), row=2, col=1)
                    
                    # 3. 成交量
                    vol_colors = ['blue' if r['极缩'] else '#ef5350' if r['close']>=r['open'] else '#26a69a' for _, r in df_p.iterrows()]
                    fig.add_trace(go.Bar(x=df_p.index, y=df_p['volume'], marker_color=vol_colors, name='成交量'), row=3, col=1)
                    
                    fig.update_layout(height=650, margin=dict(l=0,r=0,t=0,b=0), plot_bgcolor='#131722', paper_bgcolor='#131722', font=dict(color='#d1d4dc'), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
