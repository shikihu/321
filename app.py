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
    page_title="浩哥战法量化终端 v13.0 (全维度智能版)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 伪装浏览器头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Connection': 'close'
}

# ==========================================
# 2. 核心数据引擎 (腾讯直连)
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
                'turnover': float(parts[38]) if parts[38] else 0, # 换手率
                'pe': float(parts[39]) if parts[39] else 0,       # 市盈率(动)
                'pb': float(parts[46]) if parts[46] else 0,       # 市净率
                'mkt_cap': float(parts[45]) if parts[45] else 0,  # 总市值(亿)
                'change': float(parts[32]) if parts[32] else 0    # 涨跌幅
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
                return calculate_indicators(df)
    except: pass
    return None

# ==========================================
# 3. 核心算法 (1:1 复刻浩哥通达信源码)
# ==========================================
def sma(series, n, m=1): return series.ewm(alpha=m/n, adjust=False).mean()
def hhv(series, n): return series.rolling(n).max()
def llv(series, n): return series.rolling(n).min()

def calculate_indicators(df):
    if df is None or len(df) < 60: return df
    df = df.copy()
    
    C, O, H, L, V = df['close'], df['open'], df['high'], df['low'], df['volume']
    RC = C.shift(1)
    
    # --- 浩哥双线 ---
    df['趋势白线'] = C.ewm(span=9, adjust=False).mean().ewm(span=11, adjust=False).mean()
    ema_vals = [C.ewm(span=x, adjust=False).mean().ewm(span=x, adjust=False).mean() for x in [7,14,28,56]]
    df['大哥黄线'] = sum(ema_vals) / 4
    df['BBI'] = (C.rolling(3).mean() + C.rolling(6).mean() + C.rolling(12).mean() + C.rolling(24).mean()) / 4
    
    # --- 基础形态 ---
    df['当日振幅'] = (H - L) / L * 100
    df['当日涨跌幅'] = abs(C - RC) / RC * 100
    df['上涨十字星'] = (C > RC) & (abs(C - O) / O * 100 < 1.8)
    df['振幅区间'] = 8 
    
    # --- 缩量系列 ---
    hhv20, hhv30, hhv50 = hhv(V, 20), hhv(V, 30), hhv(V, 50)
    df['缩量'] = (V < hhv20 * 0.416) | (V < hhv50 / 3)
    df['回踩缩量'] = (V < hhv20 * 0.45) | (V < hhv50 / 3)
    df['适当缩量'] = (V < hhv20 * 0.618) | (V < hhv50 / 3)
    df['超缩量'] = (V < hhv30 / 4) | (V < hhv50 / 6)
    
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
    df['不是大绿棒'] = ~((C < O) & (C < RC * 0.96) & (V > RC['volume'] if 'volume' in RC else V))
    
    # --- 核心信号逻辑 (完全复刻) ---
    df['做上涨趋势'] = (df['趋势白线'] >= df['大哥黄线'] * 0.999) & \
                      ((C >= df['大哥黄线']) | ((C > df['大哥黄线'] * 0.975) & (C>O)))
    
    dist_white = abs(C - df['趋势白线']) / C * 100
    dist_yellow = abs(C - df['大哥黄线']) / df['大哥黄线'] * 100
    
    df['回踩白线'] = ((C >= df['趋势白线']) & (dist_white <= 2)) | ((C < df['趋势白线']) & (dist_white < 0.8))
    df['回踩黄线'] = ((C >= df['大哥黄线']) & ((dist_yellow <= 1.5) | ((dist_yellow <= 2) & (df['当日涨跌幅'] < 1)))) | \
                    ((C < df['大哥黄线']) & (dist_yellow <= 0.8))
    
    # 1. 超卖缩量拐头B
    df['超卖缩量拐头B'] = df['做上涨趋势'] & (df['RSI']-15 >= df['RSI'].shift(1)) & \
                         ((df['RSI'].shift(1)<20) | (df['J'].shift(1)<14)) & \
                         (df['当日振幅'] < df['振幅区间']+0.5) & \
                         ((df['当日涨跌幅']<2.3) | (df['上涨十字星'] & (df['当日涨跌幅']<4))) & \
                         df['不是大绿棒'] & (C >= df['大哥黄线'])

    # 2. 超卖缩量B
    df['超卖缩量B'] = df['做上涨趋势'] & ((df['J']<14) | (df['RSI']<23)) & \
                     ((df['RSI']+df['J']<55) | (df['J']==llv(df['J'], 20))) & \
                     (df['当日振幅']<df['振幅区间']) & df['不是大绿棒'] & \
                     (df['缩量'] | (df['适当缩量'] & (df['当日涨跌幅']<1)))

    # 3. 原始B1
    df['原始B1'] = (df['趋势白线']>df['大哥黄线']) & (C>=df['大哥黄线']*0.99) & \
                  (df['大哥黄线']>=df['大哥黄线'].shift(1)) & ((df['J']<13)|(df['RSI']<21)) & \
                  (df['RSI']+df['J'] < llv(df['RSI']+df['J'], 15)*1.5) & \
                  df['适当缩量'] & df['不是大绿棒']

    # 4. 超卖超缩量B
    df['超卖超缩量B'] = df['做上涨趋势'] & ((df['J']<14)|(df['RSI']<23)) & \
                       (df['RSI']+df['J']<60) & (df['远期振幅']>=45) & \
                       df['超缩量'] & df['不是大绿棒']

    # 5. 回踩白线B
    df['回踩白线B'] = ((df['J']<30)|(df['RSI']<40)) & (df['RSI']+df['J']<70) & \
                     ((df['当日振幅']<df['振幅区间']+0.5)|(dist_white<1)) & \
                     df['回踩白线'] & df['不是大绿棒'] & df['回踩缩量'] & (L<=RC)

    # 6. 回踩黄线B
    df['回踩黄线B'] = (df['趋势白线']>=df['大哥黄线']) & (C>=df['大哥黄线']*0.975) & \
                     ((df['J']<13)|(df['RSI']<18)) & df['回踩黄线'] & df['不是大绿棒'] & \
                     (df['缩量'] | (df['适当缩量'] & (df['J']==llv(df['J'],20)))) & \
                     (df['大哥黄线']>=df['大哥黄线'].shift(1)*0.997)

    # 辅助计算：止损与收益
    df['技术支撑'] = df[['趋势白线', '大哥黄线']].max(axis=1)
    df['止损价'] = df['技术支撑'] * 0.97
    df['未来3日最高'] = H.shift(-3).rolling(3).max()
    df['未来收益达标'] = (df['未来3日最高'] - C) / C > 0.02
    
    df = df.ffill().bfill()
    return df

# ==========================================
# 4. 智能评价系统 (技术+基本+消息)
# ==========================================
def calculate_scores(info, df):
    last = df.iloc[-1]
    
    # --- 1. 技术面 (60分) ---
    # 回测胜率
    df_test = df.iloc[-120:-3]
    strategies = {
        '超卖缩量拐头B': 1, '超卖缩量B': 1, '原始B1': 1,
        '超卖超缩量B': 1, '回踩白线B': 1, '回踩黄线B': 1
    }
    
    tech_score = 0
    signals = []
    
    # 遍历信号
    for sig in strategies.keys():
        if last[sig]:
            # 基础分
            base = 40
            # 回测修正
            triggered = df_test[df_test[sig] == True]
            if len(triggered) > 0:
                win_rate = (triggered['未来收益达标'].sum() / len(triggered)) * 100
                if win_rate >= 60: base += 20
                elif win_rate >= 50: base += 10
            else:
                base += 10 # 稀缺信号
                
            if base > tech_score: tech_score = base
            signals.append(sig.replace("B", "").replace("1", ""))
    
    # 额外趋势加分
    if last['做上涨趋势']: tech_score += 5
    if last['超缩量']: tech_score += 5
    tech_score = min(60, tech_score)

    # --- 2. 基本面 (20分) ---
    basic_score = 0
    basic_comments = []
    pe = info['pe']
    pb = info['pb']
    mkt = info['mkt_cap']
    
    # 浩哥逻辑：股权财政 = 低PB + 稳健PE
    if pb > 0 and pb < 1.5:
        basic_score += 10
        basic_comments.append("PB极低(股权财政)")
    elif pb < 3:
        basic_score += 5
        
    if pe > 0 and pe < 25:
        basic_score += 10
        basic_comments.append("低估值绩优")
    elif pe > 0 and pe < 40:
        basic_score += 5
    elif pe < 0:
        basic_score = 0
        basic_comments.append("业绩亏损⚠️")
        
    basic_score = min(20, basic_score)

    # --- 3. 消息/资金面 (20分) ---
    msg_score = 0
    msg_comments = []
    turn = info['turnover']
    
    # 逻辑：Z哥喜欢“蹭热点”，热点=换手活跃
    if 5 <= turn <= 15:
        msg_score = 20
        msg_comments.append("资金活跃(热点)")
    elif 2 <= turn < 5:
        msg_score = 15
        msg_comments.append("机构关注")
    elif turn > 15:
        msg_score = 10
        msg_comments.append("过热风险")
    else:
        msg_score = 5
        msg_comments.append("关注度低")
        
    # 量价配合
    if last['当日涨跌幅'] < 2 and last['缩量']:
        msg_score = min(20, msg_score + 5)
        
    return tech_score, basic_score, msg_score, signals, basic_comments, msg_comments

def generate_smart_comment(code, info, tech, basic, msg, signals, b_cmt, m_cmt, last):
    """浩哥AI分析师：生成真人风格的点评"""
    total = tech + basic + msg
    
    # 开头
    text = f"**{info['name']}** ({code}) 现价: **{info['price']}**\n\n"
    
    # 1. 信号定性
    if signals:
        sig_str = " + ".join(signals)
        text += f"📡 **发现信号**: {sig_str}\n"
        if "超卖" in sig_str:
            text += "> 💡 **浩哥点评**: 此时属于超卖反弹区，指标修复需求强烈，是企稳的标志。\n"
        elif "原始" in sig_str or "黄线" in sig_str:
            text += "> 💡 **浩哥点评**: 回踩重要支撑位且缩量，典型的 B1 买点结构。\n"
    else:
        text += f"📡 **发现信号**: 暂无 B1 信号\n"
        text += "> 💡 **浩哥点评**: 形态尚未调整到位，建议耐心等待缩量企稳。\n"
        
    # 2. 资金面分析
    text += f"\n💰 **资金热度**: {' '.join(m_cmt)}\n"
    if info['turnover'] < 2:
        text += "> ⚠️ **注意**: 换手率较低，说明非当前市场主线，潜伏需要耐心。\n"
    elif 5 <= info['turnover'] <= 15:
        text += "> 🔥 **注意**: 换手活跃，大概率蹭上了近期题材热点，股性较活。\n"
        
    # 3. 基本面兜底
    text += f"\n🏛️ **基本面**: {' '.join(b_cmt)}\n"
    if info['pb'] < 1.5:
        text += "> ✅ 符合 **'股权财政'** 选股逻辑，安全垫较厚，适合中线配置。\n"
        
    # 4. 操作计划
    stop_loss = last['止损价']
    text += f"\n🛡️ **交易计划**: \n"
    text += f"- **建议止损**: {stop_loss:.2f} (破位大哥黄线/白线离场)\n"
    
    return text, total

def analyze_stock_wrapper(code, info, df):
    if not info or df is None: return None
    
    tech, basic, msg, sigs, b_cmt, m_cmt = calculate_scores(info, df)
    comment, total = generate_smart_comment(code, info, tech, basic, msg, sigs, b_cmt, m_cmt, df.iloc[-1])
    
    advice = "观望"
    if total >= 80: advice = "B1买点 (重点)"
    elif total >= 65: advice = "适当关注"
    
    return {
        'code': code, 'name': info['name'], 'score': total,
        'tech': tech, 'basic': basic, 'msg': msg,
        'comment': comment, 'advice': advice, 'df': df,
        'has_signal': len(sigs) > 0
    }

# ==========================================
# 5. 主程序
# ==========================================
st.title("浩哥战法量化终端 v13.0 (全维度智能版)")
st.caption("🚀 核心：100%还原通达信B1指标 + 基本面/资金面双重过滤 + AI智能点评。")

codes_input = st.text_area("请输入股票代码 (例如: 601665, 002339)", height=100)

if st.button("🚀 开始全维度分析"):
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
                    res = analyze_stock_wrapper(code, info, df)
                    if res: results.append(res)
            bar.progress((i+1)/len(codes))
            time.sleep(0.05)
            
        results.sort(key=lambda x: x['score'], reverse=True)
        st.success("分析完成！")
        
        for res in results:
            prefix = "👑 " if res['has_signal'] else ""
            with st.expander(f"{prefix}{res['name']} ({res['code']}) - 总分 {res['score']:.0f}", expanded=res['has_signal']):
                c1, c2 = st.columns([3, 1])
                with c1: st.markdown(res['comment'])
                with c2:
                    st.metric("总分", f"{res['score']:.0f}")
                    st.caption(f"技术: {res['tech']}/60")
                    st.caption(f"基本: {res['basic']}/20")
                    st.caption(f"资金: {res['msg']}/20")
                    if res['score'] >= 80: st.error(res['advice'])
                    else: st.info(res['advice'])
                
                if res['df'] is not None:
                    df_p = res['df'].iloc[-100:]
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                    
                    # K线
                    fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['open'], high=df_p['high'], low=df_p['low'], close=df_p['close'], name='K线'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['趋势白线'], line=dict(color='white', width=1), name='白线'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['大哥黄线'], line=dict(color='yellow', width=1), name='黄线'), row=1, col=1)
                    
                    # 标记信号 (严格对应颜色)
                    sigs = [
                        ('超卖缩量拐头B', '#FFD700', '拐头B'), 
                        ('超卖缩量B', '#FF0000', '缩量B'),
                        ('原始B1', '#FFFFFF', '原始B1'),
                        ('超卖超缩量B', '#00FFFF', '超缩量B'),
                        ('回踩白线B', '#FFC0CB', '白线B'),
                        ('回踩黄线B', '#FFA500', '黄线B')
                    ]
                    
                    for col_name, color, label in sigs:
                        sig_data = df_p[df_p[col_name] == True]
                        if not sig_data.empty:
                            fig.add_trace(go.Scatter(
                                x=sig_data.index, y=sig_data['low']*0.98,
                                mode='markers', marker=dict(symbol='triangle-up', size=10, color=color),
                                name=label
                            ), row=1, col=1)
                            
                    # 成交量
                    vol_colors = ['#2962ff' if r['超缩量'] else '#ef5350' if r['close']>=r['open'] else '#26a69a' for _, r in df_p.iterrows()]
                    fig.add_trace(go.Bar(x=df_p.index, y=df_p['volume'], marker_color=vol_colors, name='成交量'), row=2, col=1)
                    
                    fig.update_layout(height=500, margin=dict(l=0,r=0,t=0,b=0), plot_bgcolor='#131722', paper_bgcolor='#131722', font=dict(color='#d1d4dc'), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
