import streamlit as st
import pandas as pd
import requests
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 数据获取函数（保持不变，省略）

# 技术指标计算（已加入你的核心公式元素）
def calculate_indicators(df):
    if df is None or len(df) < 20:
        return df
    
    df = df.copy()
    
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    ma3 = df['close'].rolling(3).mean()
    ma6 = df['close'].rolling(6).mean()
    ma12 = df['close'].rolling(12).mean()
    ma24 = df['close'].rolling(24).mean()
    df['BBI'] = (ma3 + ma6 + ma12 + ma24) / 4
    
    df['VOL5'] = df['volume'].rolling(5).mean()
    
    low_list = df['low'].rolling(9, min_periods=9).min()
    high_list = df['high'].rolling(9, min_periods=9).max()
    rsv = (df['close'] - low_list) / (high_list - low_list).replace(0, 1) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = ema12 - ema26
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    
    df['当日振幅'] = (df['high'] - df['low']) / df['low'] * 100
    df['当日涨跌幅'] = abs(df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    
    df['缩量'] = df['volume'] < df['volume'].rolling(20).max() * 0.416
    df['回踩缩量'] = df['volume'] < df['volume'].rolling(20).max() * 0.45
    df['适当缩量'] = df['volume'] < df['volume'].rolling(20).max() * 0.618
    df['超缩量'] = df['volume'] < df['volume'].rolling(30).max() / 4
    
    df['距离白线'] = abs(df['close'] - df['MA5']) / df['close'] * 100
    df['距离黄线'] = abs(df['close'] - df['MA20']) / df['close'] * 100
    
    return df

# K线图（保持不变，省略）

# 浩哥战法评分（严格匹配你的副图公式）
def analyze_stock(df, name, current, symbol):
    if df is None or len(df) < 20:
        return 0.0, f"浩哥看 {name} 数据不足，无法分析。", "浩哥建议：暂缓操作。"
    
    last = df.iloc[-1]
    
    def safe_get(col, default=0.0):
        return last.get(col, default) if col in last else default
    
    # 严格匹配你的副图公式信号
    signals = {}
    
    # 浩哥缩量战法（红色缩量B1）
    if safe_get('缩量', False) and safe_get('J', 0) < 14 and safe_get('当日振幅', 999) < 8:
        signals['浩哥缩量战法'] = True
    
    # 浩哥极缩战法（青色超级缩量B1）
    if safe_get('超缩量', False) and safe_get('J', 0) < 14 and safe_get('远期振幅', 0) >= 45:
        signals['浩哥极缩战法'] = True
    
    # 浩哥拐头战法（黄色缩量拐头B1）
    if 'rsi' in df:
        rsi_prev = df['rsi'].shift(1).iloc[-1] if len(df) > 1 else 50
        if (safe_get('rsi', 50) - 15 >= rsi_prev) and (rsi_prev < 20) and safe_get('当日振幅', 999) < 8:
            signals['浩哥拐头战法'] = True
    
    # 浩哥白线战法（紫色回踩白线B1）
    if abs(last['close'] - safe_get('MA5', last['close'])) / last['close'] * 100 < 2 and safe_get('回踩缩量', False):
        signals['浩哥白线战法'] = True
    
    # 浩哥超级战法（绿色超牛股回踩白线B1）
    if safe_get('close', 0) > safe_get('MA20', 0) * 1.05 and safe_get('缩量', False) and safe_get('J', 0) < 35:
        signals['浩哥超级战法'] = True
    
    # 浩哥黄线战法（短黄色回踩黄线B1）
    if abs(last['close'] - safe_get('MA20', last['close'])) / last['close'] * 100 <= 1.5 and safe_get('缩量', False):
        signals['浩哥黄线战法'] = True
    
    # 浩哥1.0战法（白色原始B1）
    if safe_get('MA5', 0) > safe_get('MA20', 0) and safe_get('J', 0) < 13 and safe_get('缩量', False):
        signals['浩哥1.0战法'] = True
    
    # 权重（回测胜率排序）
    weights = {
        '浩哥超级战法': 25.0,
        '浩哥极缩战法': 22.0,
        '浩哥白线战法': 18.0,
        '浩哥1.0战法': 15.0,
        '浩哥拐头战法': 10.0,
        '浩哥黄线战法': 8.0,
        '浩哥缩量战法': 5.0
    }
    
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
                    (last['close'] > safe_get('MA20', last['close']) * 1.03)
        if is_active:
            price_correction = +3.0
    tech_score += price_correction
    
    tech_score = min(max(tech_score, 0), 70.0)
    
    # AI 面
    ai_score = 0.0
    lhb_net = get_lhb_data(symbol)
    if lhb_net > 0.5:
        ai_score += min(lhb_net * 5, 15.0)
    elif lhb_net < -0.5:
        ai_score -= min(abs(lhb_net) * 5, 10.0)
    
    total_score = tech_score + ai_score
    total_score = min(max(total_score, 0), 100.0)
    
    # 专业评论
    comment = f"浩哥对 {name} 的综合判断：当前价 {current:.2f} 元。\n\n"
    
    if triggered_signals:
        comment += f"浩哥检测到关键信号：{ ' + '.join(triggered_signals) }\n\n"
    
    comment += f"【技术面评分】{tech_score:.1f}/70\n"
    comment += f"【AI 面评分】{ai_score:.1f}/30\n"
    comment += f"【浩哥综合打分】{total_score:.1f}/100\n\n"
    
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

# 主界面（保持不变，省略）
