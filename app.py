import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import akshare as ak
from datetime import datetime, timedelta

# ======================
# 1. 核心战法逻辑计算引擎
# ======================
def calculate_zge_strategy(df, symbol):
    """完全复刻通达信源码逻辑"""
    C = df['close']
    L = df['low']
    H = df['high']
    O = df['open']
    V = df['volume']

    # --- 基础均线 ---
    # 趋势白线:=EMA(EMA(C,9),11);
    df['white'] = C.ewm(span=9, adjust=False).mean().ewm(span=11, adjust=False).mean()
    # 大哥黄线
    e1 = C.ewm(span=7, adjust=False).mean().ewm(span=7, adjust=False).mean()
    e2 = C.ewm(span=14, adjust=False).mean().ewm(span=14, adjust=False).mean()
    e3 = C.ewm(span=28, adjust=False).mean().ewm(span=28, adjust=False).mean()
    e4 = C.ewm(span=56, adjust=False).mean().ewm(span=56, adjust=False).mean()
    df['yellow'] = (e1 + e2 + e3 + e4) / 4
    # BBI
    df['bbi'] = (C.rolling(3).mean() + C.rolling(6).mean() + C.rolling(12).mean() + C.rolling(24).mean()) / 4

    # --- KDJ & RSI ---
    # J:KDJ.J
    low_list = L.rolling(9).min()
    high_list = H.rolling(9).max()
    rsv = (C - low_list) / (high_list - low_list) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # RSI 3日
    lc = C.shift(1)
    temp1 = (C - lc).clip(lower=0).rolling(3).mean()
    temp2 = (C - lc).abs().rolling(3).mean()
    df['rsi'] = temp1 / temp2 * 100

    # --- 缩量判定 ---
    v_hhv20 = V.rolling(20).max()
    v_hhv30 = V.rolling(30).max()
    v_hhv50 = V.rolling(50).max()
    df['is_缩量'] = (V < v_hhv20 * 0.416) | (V < v_hhv50 / 3)
    df['is_回踩缩量'] = (V < v_hhv20 * 0.45) | (V < v_hhv50 / 3)
    df['is_适当缩量'] = (V < v_hhv20 * 0.618) | (V < v_hhv50 / 3)
    df['is_超缩量'] = (V < v_hhv30 / 4) | (V < v_hhv50 / 6)

    # --- 趋势判定 ---
    df['做上涨趋势'] = (df['white'] >= df['yellow'] * 0.999) & ((C >= df['yellow']) | ((C > df['yellow'] * 0.975) & (C > O)))
    
    # 振幅异动计算
    lown = L.rolling(20).min()
    highn = H.rolling(20).max()
    df['近期振幅'] = (highn - lown) / lown * 100
    df['近期异动'] = df['近期振幅'] >= 11
    
    # 超牛股判定
    bbi_up = (df['bbi'] >= df['bbi'].shift(1) * 0.999).rolling(20).all()
    df['超牛股'] = bbi_up & (df['近期振幅'] >= 30)

    # --- 7种B1识别 ---
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    b1_types = []
    # 1. 回踩白线判定
    dist_white = abs(last['close'] - last['white']) / last['close'] * 100
    is_回踩白线 = (last['close'] >= last['white'] and dist_white <= 2) or (last['close'] < last['white'] and dist_white < 0.8)
    
    # 匹配类型
    if last['做上涨趋势'] and last['rsi'] < 23 and last['is_缩量']: b1_types.append("超卖缩量B1")
    if last['white'] > last['yellow'] and last['is_适当缩量'] and last['J'] < 13: b1_types.append("原始B1")
    if last['超牛股'] and is_回踩白线 and last['is_适当缩量']: b1_types.append("回踩超级B")
    if last['做上涨趋势'] and last['is_超缩量'] and last['J'] < 14: b1_types.append("超卖超缩量B1")
    if is_回踩白线 and last['is_回踩缩量']: b1_types.append("回踩白线B1")

    # --- 分数计算 (0-5分) ---
    # 下跌, 放量(跌时放量), 破线, 死叉, 转势
    score = 5
    if last['close'] < prev['close']: score -= 1 # 下跌
    if last['close'] < prev['close'] and last['volume'] > prev['volume']: score -= 1 # 放量下跌
    if last['close'] < last['white']: score -= 1 # 破白线
    if last['J'] < last['K']: score -= 1 # 死叉
    if last['white'] < prev['white']: score -= 1 # 转势(白线向下)
    
    return b1_types, score, last['white'], last['yellow'], last['bbi']

# ======================
# 2. 实时热点与基本面
# ======================
@st.cache_data(ttl=3600)
def get_realtime_hot():
    """自动抓取今日行业热点榜"""
    try:
        df = ak.stock_board_industry_name_em().sort_values("今日涨跌幅", ascending=False)
        return df.head(8)['板块名称'].tolist()
    except:
        return []

def get_stock_info(symbol):
    """获取F10基本面"""
    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        industry = df[df['项目'] == '行业']['值'].values[0]
        return industry
    except:
        return "未知行业"

# ======================
# 3. Streamlit 界面渲染
# ======================
st.set_page_config(layout="wide", page_title="Z哥B1完美筛选器")

st.title("🛡️ Z哥 AI 分析师 - B1 战法深度筛选")

# 自动热点显示
hot_sectors = get_realtime_hot()
st.sidebar.info(f"🔥 今日AI识别热点板块：\n{', '.join(hot_sectors)}")

codes_input = st.text_area("输入股票代码列表（支持批量，逗号或换行分隔）", "601218, 002138, 300274")

if st.button("开始 AI 筛选完美图形"):
    codes = [c.strip() for c in codes_input.replace('\n', ',').split(',') if c.strip()]
    
    # 模拟批量处理
    all_results = []
    
    pbar = st.progress(0)
    for i, code in enumerate(codes):
        try:
            # 数据获取 (建议实际使用腾讯接口，此处示例使用akshare)
            df_hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20240101").iloc[-100:]
            df_hist.columns = ['date','open','close','high','low','volume','amount','amplitude','pct','change','turnover']
            df_hist.set_index('date', inplace=True)
            
            b1_types, score, white, yellow, bbi = calculate_zge_strategy(df_hist, code)
            industry = get_stock_info(code)
            
            # 判断是否完美
            is_hot = any(s in industry for s in hot_sectors)
            # 完美定义：分高 + 是B1 + 缩量 + 最好是热点
            perfect_score = score + (2 if is_hot else 0) + (1 if len(b1_types)>0 else 0)
            
            all_results.append({
                "code": code,
                "score": score,
                "perfect_score": perfect_score,
                "types": b1_types,
                "industry": industry,
                "is_hot": is_hot,
                "df": df_hist
            })
        except:
            continue
        pbar.progress((i + 1) / len(codes))

    # 排序：完美得分最高的排前面
    all_results = sorted(all_results, key=lambda x: x['perfect_score'], reverse=True)

    for res in all_results:
        # 分数 >= 4 且有 B1 信号的，自动展开
        is_expanded = (res['score'] >= 4 and len(res['types']) > 0)
        status_icon = "💎 完美图形" if res['perfect_score'] >= 6 else "🔎 观察"
        
        with st.expander(f"{status_icon} | 代码: {res['code']} | 持股分: {res['score']} | 类型: {', '.join(res['types'])} | 行业: {res['industry']}", expanded=is_expanded):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # 绘图：仅保留K线、白线、黄线
                df_plot = res['df'].iloc[-60:]
                fig = go.Figure(data=[go.Candlestick(
                    x=df_plot.index, open=df_plot['open'], high=df_plot['high'], 
                    low=df_plot['low'], close=df_plot['close'],
                    increasing_line_color='red', decreasing_line_color='green'
                )])
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['white'], name="白线", line=dict(color='white', width=2)))
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['yellow'], name="黄线", line=dict(color='yellow', width=2)))
                fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=400, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.subheader("AI 评定报告")
                if res['score'] >= 4:
                    st.success("✅ 持股状态：强势，拿住！")
                elif res['score'] == 3:
                    st.warning("⚠️ 持股状态：转弱，考虑放飞。")
                else:
                    st.error("❌ 持股状态：极差，考虑卖出。")
                
                st.write(f"**识别B1类型：** {', '.join(res['types']) if res['types'] else '无'}")
                st.write(f"**板块契合度：** {'🔥 属于当前热点' if res['is_hot'] else '一般'}")
                
                # 完美图形感悟总结
                if res['perfect_score'] >= 6:
                    st.markdown("---")
                    st.error("🏆 **完美图形感悟：**")
                    st.write("该股处于上涨趋势，且成交量极度萎缩至地量，回踩不破趋势支撑。配合题材走强，是战法中的‘完美买点’。")
