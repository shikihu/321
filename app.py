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
    page_title="浩哥战法量化终端 v9.0 (数据回测版)",
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
# 3. 指标与全策略信号生成
# ==========================================
def calculate_indicators(df):
    if df is None or len(df) < 20: return df
    df = df.copy()
    
    # 基础均线
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    # 浩哥双线
    ema9 = df['close'].ewm(span=9, adjust=False).mean()
    df['趋势白线'] = ema9.ewm(span=11, adjust=False).mean()
    ema_vals = [df['close'].ewm(span=x, adjust=False).mean().ewm(span=x, adjust=False).mean() for x in [7,14,28,56]]
    df['大哥黄线'] = sum(ema_vals) / 4
    
    # KDJ & RSI
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(com=13).mean() / down.ewm(com=13).mean()
    df['RSI'] = 100 - (100 / (1 + rs))

    # 量能
    df['vol_max20'] = df['volume'].rolling(20).max()
    df['极缩'] = (df['volume'] < df['vol_max20'] * 0.25)
    df['普通缩量'] = (df['volume'] < df['vol_max20'] * 0.45)
    
    # 形态特征
    dist_white = abs(df['close'] - df['趋势白线']) / df['close'] * 100
    dist_yellow = abs(df['close'] - df['大哥黄线']) / df['大哥黄线'] * 100
    
    df['回踩白线'] = (dist_white < 2.0)
    df['回踩黄线'] = (dist_yellow < 2.0)
    df['拐头'] = ((df['J'] < 25) & (df['J'] > df.shift(1)['J']))
    df['趋势向上'] = (df['close'] > df['MA60'])
    
    # ==========================
    # 定义 5 大核心策略信号
    # ==========================
    
    # 1. 浩哥极缩 (重点: 地量 + 趋势)
    df['SIG_极缩'] = df['趋势向上'] & df['极缩']
    
    # 2. 浩哥拐头 (重点: 缩量 + J值低位拐头)
    df['SIG_拐头'] = df['趋势向上'] & df['普通缩量'] & df['拐头']
    
    # 3. 浩哥白线 (重点: 回踩白线 + 缩量)
    df['SIG_白线'] = df['趋势向上'] & df['回踩白线'] & df['普通缩量']
    
    # 4. 浩哥黄线 (重点: 回踩黄线支撑)
    df['SIG_黄线'] = df['回踩黄线'] & df['普通缩量']
    
    # 5. 浩哥超级 (重点: 站稳MA20 + 缩量 + J低)
    df['SIG_超级'] = (df['close'] > df['MA20']) & df['普通缩量'] & (df['J'] < 35) & df['回踩白线']

    # 叠加王炸 (不作为独立策略，而是作为加分项)
    df['SIG_王炸'] = df['SIG_极缩'] & df['回踩白线'] & df['拐头']
    
    # 计算未来3日收益率 (用于回测)
    # 逻辑: 如果今天出现信号，未来3天最大涨幅超过2%，算胜
    df['未来3日最高'] = df['high'].shift(-3).rolling(3).max()
    df['未来收益达标'] = (df['未来3日最高'] - df['close']) / df['close'] > 0.02
    
    df = df.ffill().bfill()
    return df

# ==========================================
# 4. 回测引擎 (Real-time Backtest Logic)
# ==========================================
def perform_backtest(df):
    """
    对当前股票的历史数据进行回测，计算各大战法的真实胜率
    """
    # 统计最近 120 天的数据
    df_test = df.iloc[-120:-3] # 去掉最近3天，因为最近3天还没走完
    
    strategies = ['SIG_极缩', 'SIG_拐头', 'SIG_白线', 'SIG_黄线', 'SIG_超级']
    stats = {}
    
    for sig in strategies:
        # 找出触发信号的天数
        triggered_days = df_test[df_test[sig] == True]
        count = len(triggered_days)
        
        if count > 0:
            # 统计触发后，未来收益达标的次数
            wins = triggered_days['未来收益达标'].sum()
            win_rate = (wins / count) * 100
        else:
            win_rate = 0 # 没出现过
            
        stats[sig] = {
            'count': count,
            'win_rate': win_rate
        }
    return stats

# ==========================================
# 5. 数据驱动的评分系统
# ==========================================
def analyze_stock_logic(code, info, df):
    if not info or df is None: return None
    
    last = df.iloc[-1]
    name = info['name']
    price = info['price']
    
    # 1. 获取回测数据
    backtest_stats = perform_backtest(df)
    
    # 2. 定义价位段的基础胜率/权重 (这是大数据经验值)
    # 格式: {策略: {低价, 中价, 高价}}
    base_weights = {
        'SIG_极缩': {'low': 40, 'mid': 70, 'high': 50}, # 极缩在中价股最有效
        'SIG_拐头': {'low': 50, 'mid': 60, 'high': 60}, # 拐头比较通用
        'SIG_白线': {'low': 45, 'mid': 65, 'high': 70}, # 白线适合趋势股(高价)
        'SIG_黄线': {'low': 55, 'mid': 60, 'high': 50}, # 黄线适合低吸
        'SIG_超级': {'low': 50, 'mid': 65, 'high': 60},
    }
    
    # 确定当前价位段
    tier = 'mid'
    if price < 10: tier = 'low'
    elif price > 50: tier = 'high'
    
    # 3. 计算得分
    active_signals = []
    final_score = 0
    max_weight_score = 0
    
    # 遍历所有策略，看今天触发了没
    for sig_code, sig_name in [
        ('SIG_极缩', '极缩战法'), ('SIG_拐头', '拐头战法'), 
        ('SIG_白线', '白线战法'), ('SIG_黄线', '黄线战法'), ('SIG_超级', '超级战法')
    ]:
        if last[sig_code]:
            # A. 基础分 (根据价位段)
            base_score = base_weights[sig_code][tier]
            
            # B. 回测修正 (数据驱动核心)
            # 如果这只票历史上这个战法胜率高(>60%)，大幅加分；如果胜率低(<40%)，扣分
            hist_win_rate = backtest_stats[sig_code]['win_rate']
            hist_count = backtest_stats[sig_code]['count']
            
            adjust = 0
            win_msg = ""
            
            if hist_count >= 2: # 样本太少不参考
                if hist_win_rate > 70: 
                    adjust = 15
                    win_msg = f"(历史胜率{hist_win_rate:.0f}%🔥)"
                elif hist_win_rate > 50:
                    adjust = 5
                elif hist_win_rate < 30:
                    adjust = -15
                    win_msg = f"(历史胜率仅{hist_win_rate:.0f}%⚠️)"
            
            # 策略得分
            score = base_score + adjust
            active_signals.append(f"{sig_name}{win_msg}")
            
            # 取最高的一个策略分作为主技术分 (避免重复叠加爆炸)
            if score > max_weight_score:
                max_weight_score = score
    
    # 叠加王炸加成 (王炸是形态共振，额外加分)
    if last['SIG_王炸']:
        max_weight_score += 15
        active_signals.insert(0, "👑 王炸形态 (形态共振)")
        
    tech_score = max_weight_score
    
    # 4. 其他加分项 (热点/基本面)
    hot_score = 0
    if 5 <= info['turnover'] <= 15: hot_score = 10
    
    basic_score = 0
    if info['pb'] < 2.0: basic_score += 5 # 股权财政
    if 0 < info['pe'] < 40: basic_score += 5
    
    total_score = tech_score + hot_score + basic_score
    total_score = min(99, total_score)
    
    # 建议
    advice = "观望"
    if total_score >= 80: advice = "B1买点 (高胜率)"
    elif total_score >= 65: advice = "适当关注"
    
    # 构造评论
    sig_str = " | ".join(active_signals) if active_signals else "无有效战法信号"
    
    # 生成回测报告文案
    backtest_report = []
    for sig, data in backtest_stats.items():
        if data['count'] > 0 and last[sig]: # 只显示触发了的
            backtest_report.append(f"- **{sig.replace('SIG_', '')}**: 过去120天出现 {data['count']} 次，胜率 **{data['win_rate']:.0f}%**")
            
    bt_str = "\n".join(backtest_report) if backtest_report else "该战法近期未出现过，无历史参考。"
    
    comment = f"**{name}** ({code}) 现价: {price}\n\n"
    comment += f"📡 **触发信号**: {sig_str}\n"
    comment += f"⏳ **回测验证** (数据驱动):\n{bt_str}\n\n"
    comment += f"📊 **评分构成**: 技术(含回测){tech_score:.0f} + 热点{hot_score} + 基本面{basic_score}"
    
    return {
        'code': code, 'name': name, 'score': total_score, 'comment': comment,
        'advice': advice, 'df': df, 'is_king': last['SIG_王炸']
    }

# ==========================================
# 6. 主程序
# ==========================================
st.title("浩哥战法量化终端 v9.0 ")

codes_input = st.text_area("请输入股票代码", height=100)

if st.button("🚀 开始回测与分析"):
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
        
        st.success(f"分析完成！共 {len(results)} 只标的")
        
        for res in results:
            prefix = "👑 " if res['is_king'] else ""
            with st.expander(f"{prefix}{res['name']} ({res['code']}) - {res['score']:.0f}分", expanded=res['is_king']):
                c1, c2 = st.columns([3, 1])
                with c1: st.markdown(res['comment'])
                with c2: 
                    if res['score'] >= 80: st.error(res['advice'])
                    elif res['score'] >= 65: st.success(res['advice'])
                    else: st.info(res['advice'])
                    
                if res['df'] is not None:
                    df_p = res['df'].iloc[-100:]
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['open'], high=df_p['high'],
                                               low=df_p['low'], close=df_p['close'], name='K线'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['趋势白线'], line=dict(color='white', width=1), name='白线'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['大哥黄线'], line=dict(color='yellow', width=1), name='黄线'), row=1, col=1)
                    
                    colors = ['#9c27b0' if r['SIG_王炸'] else '#2196f3' if r['SIG_极缩'] else '#ef5350' if r['close']>=r['open'] else '#26a69a' for _,r in df_p.iterrows()]
                    fig.add_trace(go.Bar(x=df_p.index, y=df_p['volume'], marker_color=colors, name='成交量'), row=2, col=1)
                    fig.update_layout(height=450, margin=dict(l=0,r=0,t=0,b=0), plot_bgcolor='#131722', paper_bgcolor='#131722', font=dict(color='#d1d4dc'), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
