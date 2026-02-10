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
# 1. 基础数据服务（修复腾讯接口 + AkShare兜底 + 港股兼容）
# ==========================================
@st.cache_data(ttl=10)
def get_real_time_price(symbol, df=None):
    if symbol.startswith(('60', '68')): prefix = 'sh'
    elif symbol.startswith(('00', '30')): prefix = 'sz'
    else: prefix = 'sz'  # 港股/其他默认sz（腾讯接口兼容）
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
        r = requests.get(url, headers=headers, timeout=3)
        parts = r.text.split('"')[1].split(',')
        if len(parts) > 3:
            return float(parts[3]), "实时接口"
    except:
        pass
    if df is not None and not df.empty:
        return df['close'].iloc[-1], "(非交易时间/最近收盘价)"
    return 0.0, "无数据"

@st.cache_data(ttl=3600)
def fetch_history_data(symbol):
    # 方案A: 腾讯接口（加强兼容）
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            key = f"{prefix}{symbol}"
            # 兼容不同返回格式
            if 'data' in data and isinstance(data['data'], dict) and key in data['data']:
                qt_data = data['data'][key]
                day_data = qt_data.get('qfqday', qt_data.get('day', []))
            elif 'data' in data and isinstance(data['data'], list):
                day_data = data['data']
            else:
                day_data = []
            
            if day_data:
                df = pd.DataFrame([row[:6] for row in day_data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.apply(pd.to_numeric, errors='coerce')
                return calculate_indicators(df)
    except Exception as e:
        st.warning(f"腾讯接口异常 ({symbol}): {str(e)}")

    # 方案B: AkShare 兜底（支持A股 + 港股）
    try:
        if symbol.startswith(('0', '3', '6')):  # A股
            end = datetime.datetime.now().strftime("%Y%m%d")
            start = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y%m%d")
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
        else:  # 港股
            df = ak.stock_hk_daily(symbol=symbol, adjust="qfq")
        
        if not df.empty:
            df = df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'})
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return calculate_indicators(df)
    except Exception as e:
        st.warning(f"AkShare 兜底失败 ({symbol}): {str(e)}")

    return None

def get_stock_name(symbol):
    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        return df[df['项目'] == '股票简称']['值'].values[0]
    except:
        return symbol

@st.cache_data(ttl=1800)
def get_money_flow(symbol):
    try:
        market = "sh" if symbol.startswith('6') else "sz"
        flow = ak.stock_individual_fund_flow(stock=symbol, market=market)
        if not flow.empty:
            return flow.iloc[0]['主力净流入-净额'] / 100000000
    except:
        pass
    return 0.0

# ==========================================
# 2. 核心指标（修复 fillna 弃用警告 + 增加必要列）
# ==========================================
def calculate_indicators(df):
    if df is None or len(df) < 5:
        return df
   
    df = df.copy()
   
    # 均线
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
   
    # 浩哥专用线
    df['趋势白线'] = df['close'].ewm(span=9, adjust=False).mean().ewm(span=11, adjust=False).mean()
    df['大哥黄线'] = (df['close'].ewm(span=7, adjust=False).mean().ewm(span=7, adjust=False).mean() +
                       df['close'].ewm(span=14, adjust=False).mean().ewm(span=14, adjust=False).mean() +
                       df['close'].ewm(span=28, adjust=False).mean().ewm(span=28, adjust=False).mean() +
                       df['close'].ewm(span=56, adjust=False).mean().ewm(span=56, adjust=False).mean()) / 4
   
    # KDJ
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
   
    # 量能
    df['vol_max20'] = df['volume'].rolling(20).max()
    df['vol_ma5'] = df['volume'].rolling(5).mean()
   
    # 修复：fillna 用 ffill + bfill 组合，避免弃用警告
    df = df.ffill().bfill()
    return df

# ==========================================
# 3. 评分系统（浩哥个性化重构版 - 保留你的原汁原味）
# ==========================================
def rank_stock(df, name, current, symbol, money_flow):
    if df is None or len(df) < 20:
        return 0, "数据不足", "跳过", "#888"
   
    last = df.iloc[-1]
   
    # --- 变量准备 ---
    vol_ratio = last['volume'] / last['vol_max20'] if last['vol_max20'] > 0 else 0
    vol_ma5_ratio = last['volume'] / last['vol_ma5'] if last['vol_ma5'] > 0 else 0
    j_val = last['J']
    dist_yellow = abs(last['close'] - last['大哥黄线']) / last['大哥黄线'] * 100 if last['大哥黄线'] > 0 else 999
    dist_white = abs(last['close'] - last['趋势白线']) / last['close'] * 100 if last['close'] > 0 else 999
   
    is_green = last['close'] < last['open']  # 收阴
   
    base_score = 60.0
    signal_name = "浩哥1.0战法"  # 默认名
   
    # --- 判定模式：A.缩量 vs B.洗盘 ---
    is_shrink = vol_ratio < 0.5
    is_super_shrink = vol_ratio < 0.3
   
    # 洗盘逻辑 (蜜雪集团模式：放量+绿柱+不破黄线+超卖)
    is_panic_wash = (vol_ma5_ratio > 1.0) and is_green and (last['close'] > last['大哥黄线'] * 0.98) and (j_val < -5)
   
    # --- 定档与改名 ---
    if is_panic_wash:
        base_score = 85.0
        signal_name = "🩸 浩哥洗盘战法"
        key_feature = "放量恐慌洗盘"
    elif is_super_shrink and j_val < 0:
        base_score = 80.0
        signal_name = "💎 浩哥极缩战法"
        key_feature = "极致窒息缩量"
    elif is_shrink and j_val < 0:
        base_score = 75.0
        signal_name = "🟢 浩哥缩量战法"
        key_feature = "标准缩量回调"
    elif (dist_white < 1.5) and (last['close'] > last['大哥黄线']):
        base_score = 68.0
        signal_name = "🛡️ 浩哥白线战法"
        key_feature = "回踩趋势白线"
   
    # --- 细节加分 ---
    quality_score = 0.0
    if j_val < -10: quality_score += 8.0
    elif j_val < -5: quality_score += 4.0
       
    if dist_yellow < 1.0: quality_score += 5.0  # 精准回踩
   
    # --- 资金面修正 ---
    bonus_score = 0.0
    if money_flow > 0.5: bonus_score += 15.0
    elif money_flow > 0: bonus_score += 5.0
    # 流出不扣分（保持你的风格）
   
    # 低价股活跃
    if current < 12 and vol_ma5_ratio > 1.2: bonus_score += 5.0
   
    # --- 算总分 ---
    total_score = base_score + quality_score + bonus_score
    total_score = min(99.0, total_score)
   
    # --- 生成浩哥个性化评论 (拒绝套话) ---
    comment = f"浩哥瞅了瞅 {name}，现价 {current}。\n"
   
    if is_panic_wash:
        comment += f"🔥 **{signal_name}** 触发！这票主力够狠，放量砸盘想把散户吓出去。但你看，黄线根本没破，J值也打到底了。这是送钱的带血筹码！\n"
    elif "极缩" in signal_name:
        comment += f"💎 **{signal_name}** 触发！量能缩得都快没了（{vol_ratio:.2f}），说明大家都不想卖了。主力锁仓锁得死死的，变盘在即！\n"
    elif "缩量" in signal_name:
        comment += f"🟢 **{signal_name}** 触发！典型的缩量回调，走势很稳，属于标准的上车机会。\n"
    elif "白线" in signal_name:
        comment += f"🛡️ **{signal_name}** 触发！踩着白线往上走，趋势还在，比较稳健。\n"
    else:
        comment += f"🔧 形态勉强符合 **{signal_name}**，但没啥特别亮眼的，凑合看吧。\n"
   
    # 资金点评
    if money_flow > 0.3:
        comment += f"💰 资金面杠杠的！主力净流入 {money_flow:.2f} 亿，这是真金白银在干啊！"
    elif money_flow < -0.1:
        if is_panic_wash:
            comment += f"💡 资金流出 {abs(money_flow):.2f} 亿，别怕，这是主力在制造恐慌，假摔！"
        else:
            comment += f"💸 资金流出 {abs(money_flow):.2f} 亿，稍微有点虚，控制好仓位。"
   
    # 建议与颜色
    if total_score >= 85:
        advice = "浩哥喊单：极品机会，重仓干！"
        color = "#d32f2f"  # 深红
    elif total_score >= 75:
        advice = "浩哥建议：形态不错，可以搞。"
        color = "#ff5722"  # 橙红
    elif total_score >= 60:
        advice = "浩哥建议：轻仓试错，设好止损。"
        color = "#ff9800"  # 橙
    else:
        advice = "浩哥建议：有点鸡肋，换个更好的？"
        color = "#757575"  # 灰
   
    return total_score, comment, advice, color

# ==========================================
# 4. 绘图
# ==========================================
def plot_kline(df, symbol, name):
    df = df.iloc[-120:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='white', width=1), name='白线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='yellow', width=1.5), name='大哥线'), row=1, col=1)
    colors = ['red' if row['open'] < row['close'] else 'green' for i, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='成交量'), row=2, col=1)
    fig.update_layout(title=f"{name} ({symbol})", height=500, xaxis_rangeslider_visible=False, plot_bgcolor='#1e1e1e', paper_bgcolor='#0e1117', font=dict(color='white'))
    return fig

# ==========================================
# 5. 主界面
# ==========================================
st.set_page_config(page_title="浩哥战法 PK", layout="wide")
st.title("🏆 浩哥战法：优中选优 PK 终端 (v6.1 浩哥语音版)")
st.markdown("### 拒绝套话！浩哥帮你实战选股：缩量/洗盘/资金 三维定档！")

with st.sidebar:
    st.header("候选股票池")
    st.caption("粘贴代码 (支持乱序、空格)")
    codes_input = st.text_area("粘贴区域", height=300)
    run_btn = st.button("开始 PK 排名", type="primary")

if run_btn:
    codes = re.findall(r'\d{6}', codes_input)
    if not codes:
        st.error("没找到代码，兄弟你输对了吗？")
    else:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
       
        for i, symbol in enumerate(codes):
            status_text.text(f"浩哥正在分析 {symbol} ({i+1}/{len(codes)})...")
            df = fetch_history_data(symbol)
            if df is not None:
                name = get_stock_name(symbol)
                current = get_real_time_price(symbol, df)[0]
                money = get_money_flow(symbol)
                score, comment, advice, color = rank_stock(df, name, current, symbol, money)
               
                results.append({
                    "code": symbol,
                    "name": name,
                    "score": score,
                    "comment": comment,
                    "advice": advice,
                    "color": color,
                    "df": df
                })
            progress_bar.progress((i + 1) / len(codes))
       
        results.sort(key=lambda x: x['score'], reverse=True)
        st.success(f"PK 完成！浩哥帮你选出了 {len(results)} 只票。")
       
        for rank, res in enumerate(results):
            with st.container():
                c1, c2, c3 = st.columns([1.2, 3.5, 1.5])
               
                with c1:
                    st.markdown(f"### 第 {rank+1} 名")
                    st.markdown(f"<h1 style='color: {res['color']}'>{res['score']:.1f}</h1>", unsafe_allow_html=True)
                    st.caption(f"{res['name']} ({res['code']})")
               
                with c2:
                    st.markdown(res['comment'])
               
                with c3:
                    st.markdown(f"#### {res['advice']}")
                    with st.expander("K线图"):
                        st.plotly_chart(plot_kline(res['df'], res['code'], res['name']), use_container_width=True)
               
            st.divider()
