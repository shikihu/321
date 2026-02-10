import streamlit as st
import pandas as pd
import requests
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# 数据获取（保持不变，省略以节省空间）

# 技术指标计算（完整复现你的公式）
def calculate_indicators(df):
    if df is None or len(df) < 20:
        return df
    
    df = df.copy()
    
    # 趋势白线 & 大哥黄线
    df['趋势白线'] = df['close'].ewm(span=9, adjust=False).mean().ewm(span=11, adjust=False).mean()
    df['大哥黄线'] = (df['close'].ewm(span=7, adjust=False).mean().ewm(span=7, adjust=False).mean() +
                       df['close'].ewm(span=14, adjust=False).mean().ewm(span=14, adjust=False).mean() +
                       df['close'].ewm(span=28, adjust=False).mean().ewm(span=28, adjust=False).mean() +
                       df['close'].ewm(span=56, adjust=False).mean().ewm(span=56, adjust=False).mean()) / 4
    
    # BBI
    ma3 = df['close'].rolling(3).mean()
    ma6 = df['close'].rolling(6).mean()
    ma12 = df['close'].rolling(12).mean()
    ma24 = df['close'].rolling(24).mean()
    df['BBI'] = (ma3 + ma6 + ma12 + ma24) / 4
    
    # VOL5
    df['VOL5'] = df['volume'].rolling(5).mean()
    
    # KDJ
    low_list = df['low'].rolling(9, min_periods=9).min()
    high_list = df['high'].rolling(9, min_periods=9).max()
    rsv = (df['close'] - low_list) / (high_list - low_list).replace(0, 1) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # RSI3日
    lc = df['close'].shift(1)
    temp1 = np.maximum(df['close'] - lc, 0)
    temp2 = np.abs(df['close'] - lc)
    df['rsi'] = temp1.rolling(3).mean() / temp2.rolling(3).mean() * 100
    
    # 振幅 & 涨跌幅
    df['当日振幅'] = (df['high'] - df['low']) / df['low'] * 100
    df['当日涨跌幅'] = abs(df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    
    # 缩量系列
    df['缩量'] = (df['volume'] < df['volume'].rolling(20).max() * 0.416) | (df['volume'] < df['volume'].rolling(50).max() / 3)
    df['回踩缩量'] = (df['volume'] < df['volume'].rolling(20).max() * 0.45) | (df['volume'] < df['volume'].rolling(50).max() / 3)
    df['适当缩量'] = (df['volume'] < df['volume'].rolling(20).max() * 0.618) | (df['volume'] < df['volume'].rolling(50).max() / 3)
    df['超缩量'] = (df['volume'] < df['volume'].rolling(30).max() / 4) | (df['volume'] < df['volume'].rolling(50).max() / 6)
    
    # 大绿棒
    vday = df['volume'].rolling(40).apply(lambda x: x.argmax(), raw=True).astype(int)
    df['大绿棒'] = (df['close'].shift(vday) < df['close'].shift(vday + 1)) & (df['close'].shift(vday) < df['open'].shift(vday))
    df['大绿棒离得远'] = vday >= 15 & df['大绿棒']
    
    # 近期/远期振幅
    df['近期振幅'] = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['low'].rolling(20).min() * 100
    df['远期振幅'] = (df['high'].rolling(50).max() - df['low'].rolling(50).min()) / df['low'].rolling(50).min() * 100
    
    # MA60
    df['MA60'] = df['close'].rolling(60).mean()
    
    return df

# K线图（安全版）
def plot_kline(df, symbol, name):
    df = df.iloc[-120:].copy()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='K线', increasing_line_color='red', decreasing_line_color='green'
    ), row=1, col=1)
    
    # 安全添加均线
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

# 浩哥战法评分（100%对齐你的公式）
def analyze_stock(df, name, current, symbol, money_flow):
    if df is None or len(df) < 20:
        return 0.0, f"浩哥看 {name} 数据不足，无法分析。", "浩哥建议：暂缓操作。"
    
    last = df.iloc[-1]
    
    def safe_get(col, default=0.0):
        return last.get(col, default) if col in last else default
    
    # 核心指标
    trend_white = safe_get('趋势白线', last['close'])
    brother_yellow = safe_get('大哥黄线', last['close'])
    
    # 做上涨趋势
    do_up_trend = trend_white >= brother_yellow * 0.999 and (last['close'] >= brother_yellow or (last['close'] > brother_yellow * 0.975 and last['close'] > last['open']))
    
    # 缩量系列
    shrink = safe_get('缩量', False)
    back_shrink = safe_get('回踩缩量', False)
    proper_shrink = safe_get('适当缩量', False)
    super_shrink = safe_get('超缩量', False)
    
    # 回踩白线 & 黄线
    dist_white = abs(last['close'] - trend_white) / last['close'] * 100
    back_white = (last['close'] >= trend_white and dist_white <= 2) or (last['close'] < trend_white and dist_white < 0.8)
    
    dist_yellow = abs(last['close'] - brother_yellow) / brother_yellow * 100
    back_yellow = (last['close'] >= brother_yellow and dist_yellow <= 1.5) or (last['close'] < brother_yellow and dist_yellow <= 0.8)
    
    # 7种浩哥战法（完整对齐你的公式）
    signals = {}
    
    # 浩哥缩量战法（红色缩量B1）
    if do_up_trend and safe_get('J', 0) < 14 and shrink and safe_get('当日振幅', 999) < 8 and (safe_get('当日涨跌幅', 999) < 2.5 or (last['close'] > last['open'] and safe_get('当日涨跌幅', 999) < 4)):
        signals['浩哥缩量战法'] = True
    
    # 浩哥极缩战法（青色超级缩量B1）
    if do_up_trend and safe_get('J', 0) < 14 and super_shrink and safe_get('当日振幅', 999) < 8:
        signals['浩哥极缩战法'] = True
    
    # 浩哥拐头战法（黄色缩量拐头B1）
    if 'rsi' in df:
        rsi_prev = df['rsi'].shift(1).iloc[-1] if len(df) > 1 else 50
        if do_up_trend and (safe_get('rsi', 50) - 15 >= rsi_prev) and (rsi_prev < 20) and safe_get('当日振幅', 999) < 8 and shrink:
            signals['浩哥拐头战法'] = True
    
    # 浩哥白线战法（紫色回踩白线B1）
    if do_up_trend and back_white and back_shrink and safe_get('J', 0) < 30 and safe_get('当日振幅', 999) < 8.5:
        signals['浩哥白线战法'] = True
    
    # 浩哥超级战法（绿色超牛股回踩白线B1）
    if safe_get('close', 0) > safe_get('MA20', 0) * 1.05 and proper_shrink and safe_get('J', 0) < 35 and back_white:
        signals['浩哥超级战法'] = True
    
    # 浩哥黄线战法（短黄色回踩黄线B1） - 完整条件
    yellow_prev = df['大哥黄线'].shift(1).iloc[-1] if '大哥黄线' in df else 0
    ma60_prev = df['MA60'].shift(1).iloc[-1] if 'MA60' in df else 0
    if back_yellow and shrink and (brother_yellow >= yellow_prev * 0.997) and (safe_get('MA60', 0) >= ma60_prev) and safe_get('近期振幅', 0) >= 11.9 and safe_get('远期振幅', 0) >= 19.5 and (safe_get('J', 0) < 13 or safe_get('rsi', 0) < 18):
        signals['浩哥黄线战法'] = True
    
    # 浩哥1.0战法（白色原始B1）
    if trend_white > brother_yellow and safe_get('J', 0) < 13 and proper_shrink and safe_get('当日振幅', 999) < 8:
        signals['浩哥1.0战法'] = True
    
    # 权重
    weights = {
        '浩哥超级战法': 25.0,
        '浩哥极缩战法': 22.0,
        '浩哥白线战法': 18.0,
        '浩哥1.0战法': 15.0,
        '浩哥拐头战法': 10.0,
        '浩哥黄线战法': 8.0,
        '浩哥缩量战法': 5.0
    }
    
    # 技术分
    tech_score = 0.0
    triggered_signals = []
    for sig, active in signals.items():
        if active:
            tech_score += weights[sig]
            triggered_signals.append(sig)
    
    # 低价股复活
    price_correction = 0.0
    if current < 12:
        price_correction = -5.0
        is_active = (safe_get('volume', 0) / safe_get('VOL5', 1) > 1.5) or \
                    (last['close'] > brother_yellow * 1.03)
        if is_active:
            price_correction = +3.0
    tech_score += price_correction
    
    tech_score = min(max(tech_score, 0), 70.0)
    
    # AI 面
    ai_score = 0.0
    lhb_net = get_lhb_data(symbol)
    real_flow = money_flow if abs(money_flow) > 0 else lhb_net
    if real_flow > 0.5:
        ai_score += min(real_flow * 5, 15.0)
    elif real_flow > 0.1:
        ai_score += 5.0
    elif real_flow < -0.5:
        ai_score -= min(abs(real_flow) * 5, 10.0)
    
    total_score = tech_score + ai_score
    total_score = min(max(total_score, 0), 100.0)
    
    # 专业评论
    comment = f"浩哥对 {name} 的综合判断：当前价 {current:.2f} 元。\n\n"
    
    if triggered_signals:
        comment += f"浩哥检测到关键信号：{ ' + '.join(triggered_signals) }\n\n"
    else:
        comment += "浩哥今天未检测到关键信号，形态未到最佳点。\n\n"
    
    comment += f"【技术面评分】{tech_score:.1f}/70\n"
    comment += f"【AI 面评分】{ai_score:.1f}/30\n"
    comment += f"【浩哥综合打分】{total_score:.1f}/100\n\n"
    
    comment += f"💰 资金面：主力净流入 {real_flow:.2f} 亿。\n"
    
    if total_score >= 85:
        comment += "浩哥认为当前形态与资金情绪高度共振，机会显著大于风险，属于较优的低吸/加仓窗口。"
        advice = "建议积极布局，仓位可适当加重，关注放量突破确认。"
    elif total_score >= 70:
        comment += "浩哥判断技术结构稳定，资金面有支撑，但还需更多确认信号，避免追高。"
        advice = "可分批试仓，控制仓位在30-50%，等待更明确的信号再加仓。"
    elif total_score >= 50:
        comment += "浩哥目前持中性偏谨慎态度，形态尚未完全走好，风险与机会并存。"
        advice = "建议轻仓或观望，耐心等待更好的入场点。"
    else:
        comment += "浩哥认为当前风险大于机会，形态和情绪均未到位，短期不宜重仓。"
        advice = "浩哥建议暂时回避，保护本金，等待更清晰的信号。"
    
    return total_score, comment, advice

# 主界面（保持不变，省略以节省空间）
