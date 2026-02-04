import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import akshare as ak
from datetime import datetime

# --- 页面配置 ---
st.set_page_config(layout="wide", page_title="Z哥战法 AI 深度筛选")

# --- 数据获取逻辑 ---
def fetch_data(symbol):
    # 自动处理带后缀或不带后缀的代码
    symbol = ''.join(filter(str.isdigit, symbol)) 
    try:
        prefix = 'sh' if symbol.startswith('6') else 'sz'
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,260,qfq"
        res = requests.get(url, timeout=10).json()
        data_root = res.get('data', {})
        stock_key = f"{prefix}{symbol}"
        inner_data = data_root.get(stock_key, data_root.get("", {}))
        raw_data = inner_data.get('qfqday', []) or inner_data.get('day', [])
        if not raw_data: return None
        df = pd.DataFrame([row[:6] for row in raw_data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        for col in df.columns: df[col] = pd.to_numeric(df[col])
        return df
    except:
        return None

@st.cache_data(ttl=3600) # 给基本面数据加缓存，防止多次查询被封IP或变慢
def get_f10(symbol):
    """获取个股基本面"""
    symbol = ''.join(filter(str.isdigit, symbol))
    try:
        info = ak.stock_individual_info_em(symbol=symbol)
        industry = info[info['项目'] == '行业']['值'].values[0]
        return {"行业": industry}
    except:
        return {"行业": "综合行业"}

@st.cache_data(ttl=3600)
def get_hot_sectors():
    """抓取今日热点板块"""
    try:
        df = ak.stock_board_industry_name_em()
        df = df.sort_values("今日涨跌幅", ascending=False).head(10)
        return df['板块名称'].tolist()
    except:
        return []

# --- 战法核心逻辑 ---
def analyze_zge_v2(df, symbol, hot_sectors):
    # 指标计算
    C, L, H, O, V = df['close'], df['low'], df['high'], df['open'], df['volume']
    
    # 趋势线
    white = C.ewm(span=9, adjust=False).mean().ewm(span=11, adjust=False).mean()
    e1 = C.ewm(span=7, adjust=False).mean().ewm(span=7, adjust=False).mean()
    e4 = C.ewm(span=56, adjust=False).mean().ewm(span=56, adjust=False).mean()
    yellow = (e1 + e4) / 2
    
    # KDJ 
    low_9 = L.rolling(9).min()
    high_9 = H.rolling(9).max()
    rsv = (C - low_9) / (high_9 - low_9).replace(0, 1) * 100
    K = rsv.ewm(com=2, adjust=False).mean()
    D = K.ewm(com=2, adjust=False).mean()
    J = 3 * K - 2 * D
    
    # 缩量与异动
    v_hhv20 = V.rolling(20).max()
    v_hhv50 = V.rolling(50).max()
    is_extreme_vol = V.iloc[-1] < v_hhv20.iloc[-1] * 0.4
    vol_ratio = V.iloc[-1] / V.rolling(5).mean().iloc[-1] if V.rolling(5).mean().iloc[-1] != 0 else 1
    
    # 7种B1识别核心
    b1_found = []
    dist_white = abs(C.iloc[-1] - white.iloc[-1]) / white.iloc[-1] * 100
    
    if dist_white < 1.5 and V.iloc[-1] < v_hhv20.iloc[-1] * 0.5: b1_found.append("回踩白线B1")
    if J.iloc[-1] < 15 and is_extreme_vol: b1_found.append("超卖缩量B1")
    if J.iloc[-1] < 10: b1_found.append("原始B1")
    if V.iloc[-1] < v_hhv50.iloc[-1] / 5: b1_found.append("超级缩量B1")

    # --- 深度打分逻辑 ---
    score = 0
    if b1_found: score += 40
    if is_extreme_vol: score += 20
    if dist_white < 2: score += 15
    if white.iloc[-1] > white.iloc[-2]: score += 10
    
    f10 = get_f10(symbol)
    if any(s in f10['行业'] for s in hot_sectors): score += 15

    # AI 点评
    comments = []
    if dist_white < 1.0: comments.append(f"K线目前精准卡位在趋势白线附近。")
    elif C.iloc[-1] > white.iloc[-1]: comments.append(f"股价站稳白线，重心在缓慢抬升。")
    
    if vol_ratio < 0.6: comments.append(f"今日成交量仅为均量的{vol_ratio:.1f}倍，处于窒息缩量状态。")
    else: comments.append(f"量能控制尚可，但尚未到达极度地量。")
        
    if J.iloc[-1] < 0: comments.append(f"J值杀入负值区，短线极度超卖。")
    if any(s in f10['行业'] for s in hot_sectors): comments.append(f"属于热点板块{f10['行业']}，共振性强。")

    return b1_found, score, " ".join(comments), f10, white, yellow

# --- 界面展示 ---
st.markdown("<h1 style='text-align: center;'>🚀 Z哥 AI 战法筛选 - 寻找完美图形</h1>", unsafe_allow_html=True)

hot_sectors = get_hot_sectors()
with st.sidebar:
    st.write("### 🔥 今日热点板块")
    if hot_sectors:
        for s in hot_sectors: st.success(s)
    else:
        st.warning("热点抓取受限。")

code_input = st.text_area("输入股票代码（多个用逗号或换行分隔）：", "000008")

if st.button("让 AI 开始感悟分析"):
    # 增强代码解析能力
    codes = [c.strip() for c in code_input.replace('\n', ',').replace('，', ',').split(',') if c.strip()]
    
    for code in codes:
        with st.spinner(f'正在分析 {code}...'):
            df = fetch_data(code)
            if df is not None:
                b1_types, buy_score, ai_comment, f10, white, yellow = analyze_zge_v2(df, code, hot_sectors)
                score_color = "red" if buy_score >= 80 else "orange" if buy_score >= 60 else "#AAAAAA"
                
                st.markdown(f"---")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    df_p = df.iloc[-60:]
                    fig = go.Figure(data=[go.Candlestick(x=df_p.index, open=df_p['open'], high=df_p['high'], low=df_p['low'], close=df_p['close'], name="K线")])
                    fig.add_trace(go.Scatter(x=df_p.index, y=white.iloc[-60:], name="趋势白线", line=dict(color='white', width=2)))
                    fig.add_trace(go.Scatter(x=df_p.index, y=yellow.iloc[-60:], name="大哥黄线", line=dict(color='yellow', width=2)))
                    fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=400, margin=dict(l=10,r=10,t=10,b=10))
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown(f"### 代码：{code}")
                    st.markdown(f"### <span style='color:{score_color}'>买入评分：{buy_score} 分</span>", unsafe_allow_html=True)
                    st.info(f"**识别类型：** {', '.join(b1_types) if b1_types else '信号潜伏中'}")
                    st.write(f"**所属行业：** {f10['行业']}")
                    st.write(f"**AI 导师点评：**")
                    st.write(ai_comment)
                    
                    if buy_score >= 80:
                        st.error("🏆 结论：图形极度完美，建议在白线支撑位逢低布局！")
                    elif buy_score >= 60:
                        st.warning("💡 结论：符合战法逻辑，但建议小仓位试错。")
                    else:
                        st.info("⌛ 结论：图形尚未走圆润，耐心等待极端缩量或B1信号出现。") # 这里修复了之前的错误
            else:
                st.error(f"代码 {code} 数据抓取失败。")
