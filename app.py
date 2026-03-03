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
socket.setdefaulttimeout(20)
st.set_page_config(
    page_title="浩哥战法量化终端 v14.1 (稳定性修复版)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 伪装浏览器头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Connection': 'close'
}

# ==========================================
# 2. 核心数据引擎
# ==========================================
def get_realtime_data(symbol):
    symbol = str(symbol).strip()
    prefix = 'sh' if symbol.startswith(('6', '9')) else 'sz'
    code = f"{prefix}{symbol}"
    try:
        url = f"http://qt.gtimg.cn/q={code}"
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.encoding = 'gbk'
        text = r.text
        if not text or f'v_{code}="' not in text: return None
        data_str = text.split('"')[1]
        parts = data_str.split('~')
        if len(parts) > 45:
            return {
                'name': parts[1],
                'code': code,
                'price': float(parts[3]),
                'turnover': float(parts[38]) if parts[38] else 0,
                'pe': float(parts[39]) if parts[39] else 0,
                'pb': float(parts[46]) if parts[46] else 0,
                'mkt_cap': float(parts[45]) if parts[45] else 0,
                'change': float(parts[32]) if parts[32] else 0
            }
    except: pass
    return None

@st.cache_data(ttl=3600)
def fetch_kline_data(symbol):
    symbol = str(symbol).strip()
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    try:
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            key = f"{prefix}{symbol}"
            day_data = data.get('data', {}).get(key, {}).get('qfqday', [])
            if not day_data: day_data = data.get('data', {}).get(key, {}).get('day', [])
            if day_data:
                df = pd.DataFrame([row[:6] for row in day_data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.apply(pd.to_numeric, errors='coerce')
                # 只要有数据就尝试计算，内部做长度保护
                return calculate_indicators(df)
    except: pass
    return None

# ==========================================
# 3. 核心算法 (带安全检查)
# ==========================================
def sma(series, n, m=1): return series.ewm(alpha=m/n, adjust=False).mean()
def hhv(series, n): return series.rolling(n).max()
def llv(series, n): return series.rolling(n).min()

def calculate_indicators(df):
    # 如果数据太短(新股)，直接返回，不计算复杂指标
    if df is None or len(df) < 60: 
        # 标记为数据不足，避免后续报错
        df['数据不足'] = True
        return df
        
    df = df.copy()
    df['数据不足'] = False # 标记数据正常
    
    C, O, H, L, V = df['close'], df['open'], df['high'], df['low'], df['volume']
    RC = C.shift(1)
    
    # --- 浩哥双线 ---
    df['趋势白线'] = C.ewm(span=9, adjust=False).mean().ewm(span=11, adjust=False).mean()
    ema_vals = [C.ewm(span=x, adjust=False).mean().ewm(span=x, adjust=False).mean() for x in [7,14,28,56]]
    df['大哥黄线'] = sum(ema_vals) / 4
    df['MA60'] = C.rolling(60).mean()
    
    # --- 基础形态 ---
    df['当日振幅'] = (H - L) / L * 100
    df['当日涨跌幅'] = abs(C - RC) / RC * 100
    df['上涨十字星'] = (C > RC) & (abs(C - O) / O * 100 < 1.8)
    df['振幅区间'] = 8.5 
    
    # --- 缩量系列 ---
    hhv20, hhv30, hhv50 = hhv(V, 20), hhv(V, 30), hhv(V, 50)
    df['缩量'] = (V < hhv20 * 0.45) | (V < hhv50 / 3) 
    df['回踩缩量'] = (V < hhv20 * 0.5) | (V < hhv50 / 3)
    df['适当缩量'] = (V < hhv20 * 0.65) | (V < hhv50 / 3)
    df['超缩量'] = (V < hhv30 / 3.5) | (V < hhv50 / 5)
    
    # --- KDJ & RSI ---
    rsv = (C - llv(L, 9)) / (hhv(H, 9) - llv(L, 9)) * 100
    df['K'] = sma(rsv, 3, 1)
    df['D'] = sma(df['K'], 3, 1)
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    temp1 = (C - RC).clip(lower=0)
    temp2 = abs(C - RC)
    df['RSI'] = sma(temp1, 3, 1) / sma(temp2, 3, 1) * 100
    
    # --- 异动与辅助 ---
    df['近期振幅'] = (hhv(H, 20) - llv(L, 20)) / llv(L, 20) * 100
    df['远期振幅'] = (hhv(H, 50) - llv(L, 50)) / llv(L, 50) * 100
    df['不是大绿棒'] = ~((C < O) & (C < RC * 0.96) & (V > RC['volume'] * 1.1))
    
    # --- 核心信号逻辑 ---
    df['做上涨趋势'] = (df['趋势白线'] >= df['大哥黄线'] * 0.99) & \
                      ((C >= df['大哥黄线']) | ((C > df['大哥黄线'] * 0.97) & (C>O)))
    
    dist_white = abs(C - df['趋势白线']) / C * 100
    dist_yellow = abs(C - df['大哥黄线']) / df['大哥黄线'] * 100
    
    df['回踩白线'] = ((C >= df['趋势白线']) & (dist_white <= 2.5)) | ((C < df['趋势白线']) & (dist_white < 1.0))
    df['回踩黄线'] = ((C >= df['大哥黄线']) & (dist_yellow <= 2.5)) | ((C < df['大哥黄线']) & (dist_yellow <= 1.0))
    
    # 1. 超卖缩量拐头B
    df['拐头B'] = df['做上涨趋势'] & (df['RSI']-15 >= df['RSI'].shift(1)) & \
                 ((df['RSI'].shift(1)<25) | (df['J'].shift(1)<20)) & \
                 (df['当日振幅'] < df['振幅区间']+1) & \
                 df['不是大绿棒'] & (C >= df['大哥黄线'])

    # 2. 超卖缩量B
    df['缩量B'] = df['做上涨趋势'] & ((df['J']<18) | (df['RSI']<25)) & \
                 ((df['RSI']+df['J']<60) | (df['J']==llv(df['J'], 20))) & \
                 (df['当日振幅']<df['振幅区间']) & df['不是大绿棒'] & \
                 (df['缩量'] | (df['适当缩量'] & (df['当日涨跌幅']<1.5)))

    # 3. 原始B1
    df['原始B1'] = (df['趋势白线']>df['大哥黄线']) & (C>=df['大哥黄线']*0.985) & \
                  (df['大哥黄线']>=df['大哥黄线'].shift(1)) & ((df['J']<18)|(df['RSI']<25)) & \
                  df['适当缩量'] & df['不是大绿棒']

    # 4. 超卖超缩量B
    df['超缩量B'] = df['做上涨趋势'] & ((df['J']<18)|(df['RSI']<25)) & \
                   (df['RSI']+df['J']<65) & (df['远期振幅']>=40) & \
                   df['超缩量'] & df['不是大绿棒']

    # 5. 回踩白线B
    df['白线B'] = ((df['J']<35)|(df['RSI']<45)) & \
                 df['回踩白线'] & df['不是大绿棒'] & df['回踩缩量'] & (L<=RC)

    # 6. 回踩黄线B
    df['黄线B'] = (df['趋势白线']>=df['大哥黄线']) & (C>=df['大哥黄线']*0.97) & \
                 ((df['J']<18)|(df['RSI']<23)) & df['回踩黄线'] & df['不是大绿棒'] & \
                 (df['缩量'] | df['适当缩量']) & \
                 (df['大哥黄线']>=df['大哥黄线'].shift(1)*0.995) & \
                 (df['MA60']>=df['MA60'].shift(1)*0.999)

    # 收益达标
    df['未来3日最高'] = H.shift(-3).rolling(3).max()
    df['收益达标'] = (df['未来3日最高'] - C) / C > 0.02
    
    df = df.ffill().bfill()
    return df

# ==========================================
# 4. 矩阵回测引擎 (带KeyError修复)
# ==========================================

TIER_MATRIX = {
    'low':  {'min': 0, 'max': 8,   'base_score': 50, 'name': '低价股'},
    'mid':  {'min': 8, 'max': 50,  'base_score': 70, 'name': '黄金价位'},
    'high': {'min': 50,'max': 9999,'base_score': 60, 'name': '高价股'}
}

def perform_matrix_backtest(df, current_price):
    # 关键修复：如果数据不足，直接返回空结果，防止报错
    if '数据不足' in df.columns and df['数据不足'].iloc[-1]:
        return None, {}, ["⚠️ 数据不足(次新股)，无法回测"]

    df_test = df.iloc[-120:-3] # 过去半年
    
    # 关键修复：检查列是否存在
    all_strategies = ['拐头B', '缩量B', '原始B1', '超缩量B', '白线B', '黄线B']
    strategies = [s for s in all_strategies if s in df.columns]
    
    backtest_result = {}
    
    # 1. 确定价位段
    tier_info = {}
    for t_name, t_data in TIER_MATRIX.items():
        if t_data['min'] <= current_price < t_data['max']:
            tier_info = t_data
            break
            
    # 2. 个股历史回测
    history_report = []
    
    for sig in strategies:
        triggered = df_test[df_test[sig] == True]
        count = len(triggered)
        
        win_rate = 0
        if count > 0:
            wins = triggered['收益达标'].sum()
            win_rate = (wins / count) * 100
            
        backtest_result[sig] = {
            'count': count,
            'win_rate': win_rate
        }
        
        if count >= 1:
            history_report.append(f"{sig}: 出现{count}次, 胜率{win_rate:.0f}%")

    return tier_info, backtest_result, history_report

# ==========================================
# 5. 评分与展示逻辑
# ==========================================
def analyze_stock_logic(code, info, df):
    if not info or df is None: return None
    
    last = df.iloc[-1]
    name = info['name']
    price = info['price']
    
    # 如果数据不足
    if '数据不足' in df.columns and df['数据不足'].iloc[-1]:
        return {
            'code': code, 'name': name, 'score': 0,
            'comment': f"**{name}** ({code})\n⚠️ 上市时间太短或数据不足，无法计算战法指标。",
            'advice': "数据不足", 'df': df, 'has_signal': False
        }

    # 执行矩阵回测
    tier_info, bt_result, hist_report = perform_matrix_backtest(df, price)
    
    score = 0
    signals = []
    active_sigs = [] 
    
    # 遍历所有信号
    for sig, data in bt_result.items():
        if sig in last and last[sig]: # 关键修复：检查sig是否在last里
            active_sigs.append(sig)
            
            current_score = tier_info.get('base_score', 60)
            
            if data['count'] > 0:
                if data['win_rate'] >= 60: 
                    current_score += 15
                    signals.append(f"{sig} (历史胜率{data['win_rate']:.0f}%🔥)")
                elif data['win_rate'] <= 40:
                    current_score -= 15
                    signals.append(f"{sig} (历史胜率{data['win_rate']:.0f}%⚠️)")
                else:
                    signals.append(f"{sig} (历史胜率{data['win_rate']:.0f}%)")
            else:
                current_score += 5
                signals.append(f"{sig} (稀缺信号🆕)")
            
            if current_score > score:
                score = current_score

    # 额外加分项
    if info['pb'] < 2.0: score += 5
    if 5 <= info['turnover'] <= 15: score += 5
    
    score = min(99, max(0, score))
    
    advice = "观望"
    if score >= 85: advice = "S级 (历史验证高胜率)"
    elif score >= 70: advice = "A级 (值得关注)"
    elif score >= 60: advice = "B级 (需谨慎)"
    
    sig_str = " + ".join(signals) if signals else "无 B1 信号"
    hist_str = " | ".join(hist_report) if hist_report else "该股近期无此类信号记录"
    
    comment = f"**{name}** ({code}) 现价: {price}\n\n"
    comment += f"📊 **价位属性**: {tier_info.get('name', '未知')} (基础分 {tier_info.get('base_score', 60)})\n"
    comment += f"📡 **今日信号**: {sig_str}\n"
    comment += f"⏳ **历史回测**: \n> {hist_str}\n"
    
    # 止损位
    try:
        stop_loss = df[['趋势白线', '大哥黄线']].max(axis=1).iloc[-1] * 0.97
        comment += f"\n🛡️ **止损参考**: {stop_loss:.2f}"
    except:
        pass
    
    return {
        'code': code, 'name': name, 'score': score, 
        'comment': comment, 'advice': advice, 'df': df,
        'has_signal': len(active_sigs) > 0
    }

# ==========================================
# 6. 主程序
# ==========================================
st.title("浩哥战法量化终端 v14.1 (稳定性修复版)")
st.caption("🚀 修复说明：彻底解决了 KeyError 报错问题，增加数据长度检查，防止次新股卡死程序。")

codes_input = st.text_area("请输入股票代码 (例如: 002471, 002339)", height=100)

if st.button("🚀 矩阵回测分析"):
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
                if df is not None:
                    res = analyze_stock_logic(code, info, df)
                    if res: results.append(res)
            bar.progress((i+1)/len(codes))
            time.sleep(0.05)
            
        results.sort(key=lambda x: x['score'], reverse=True)
        st.success("分析完成！")
        
        for res in results:
            prefix = "👑 " if res['has_signal'] else ""
            with st.expander(f"{prefix}{res['name']} ({res['code']}) - {res['score']:.0f}分", expanded=res['has_signal']):
                c1, c2 = st.columns([3, 1])
                with c1: st.markdown(res['comment'])
                with c2: 
                    if res['score'] >= 80: st.error(res['advice'])
                    elif res['score'] >= 60: st.success(res['advice'])
                    else: st.info(res['advice'])
                    
                if res['df'] is not None and len(res['df']) > 20:
                    df_p = res['df'].iloc[-100:]
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                    
                    fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['open'], high=df_p['high'], low=df_p['low'], close=df_p['close'], name='K线'), row=1, col=1)
                    if '趋势白线' in df_p.columns:
                        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['趋势白线'], line=dict(color='white', width=1), name='白线'), row=1, col=1)
                    if '大哥黄线' in df_p.columns:
                        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['大哥黄线'], line=dict(color='yellow', width=1), name='黄线'), row=1, col=1)
                    
                    # 标记信号
                    sigs = ['拐头B', '缩量B', '原始B1', '超缩量B', '白线B', '黄线B']
                    colors = ['#FFD700', '#FF0000', '#FFFFFF', '#00FFFF', '#FFC0CB', '#FFA500']
                    
                    for sig, color in zip(sigs, colors):
                        if sig in df_p.columns:
                            win_data = df_p[(df_p[sig]==True) & (df_p['收益达标']==True)]
                            if not win_data.empty:
                                 fig.add_trace(go.Scatter(x=win_data.index, y=win_data['low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=8, color='green'), name=f'{sig}成功'))
                    
                    # 今日信号
                    current_sigs = df_p.iloc[[-1]]
                    for sig in sigs:
                        if sig in current_sigs.columns and current_sigs[sig].values[0]:
                             fig.add_trace(go.Scatter(x=current_sigs.index, y=current_sigs['low']*0.96, mode='markers', marker=dict(symbol='triangle-up', size=12, color='gold'), name=f'今日{sig}'))

                    fig.add_trace(go.Bar(x=df_p.index, y=df_p['volume'], name='成交量'), row=2, col=1)
                    fig.update_layout(height=500, margin=dict(l=0,r=0,t=0,b=0), plot_bgcolor='#131722', paper_bgcolor='#131722', font=dict(color='#d1d4dc'), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
