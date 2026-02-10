import streamlit as st
import pandas as pd
import requests
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import re  # 引入正则处理各种分隔符

# ==========================================
# 1. 基础服务 (增强版：智能识别+防反爬)
# ==========================================
@st.cache_data(ttl=10)
def get_real_time_price(symbol, df=None):
    # 智能判断前缀
    if symbol.startswith(('60', '68')): prefix = 'sh'
    elif symbol.startswith(('00', '30')): prefix = 'sz'
    else: prefix = 'sz' # 默认兜底
    
    headers = {'User-Agent': 'Mozilla/5.0'} 
    try:
        url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
        r = requests.get(url, headers=headers, timeout=3)
        parts = r.text.split('"')[1].split(',')
        if len(parts) > 3:
            return float(parts[3])
    except:
        pass
    if df is not None and not df.empty:
        return df['close'].iloc[-1]
    return 0.0

@st.cache_data(ttl=3600)
def fetch_history_data(symbol):
    if symbol.startswith(('60', '68')): prefix = 'sh'
    elif symbol.startswith(('00', '30')): prefix = 'sz'
    else: prefix = 'sz'

    # 方案A: 腾讯接口 (快)
    try:
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)
        data = r.json()
        key = f"{prefix}{symbol}"
        # 兼容腾讯各种返回层级
        qt_data = data.get('data', {}).get(key, {})
        day_data = qt_data.get('qfqday', qt_data.get('day', []))
        
        if day_data:
            df = pd.DataFrame([row[:6] for row in day_data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.apply(pd.to_numeric, errors='coerce')
            return calculate_indicators(df)
    except:
        pass

    # 方案B: AkShare 兜底 (稳)
    try:
        end = datetime.datetime.now().strftime("%Y%m%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
        if not df.empty:
            df = df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'})
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return calculate_indicators(df)
    except:
        pass
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
# 2. 核心指标 (保留原味公式)
# ==========================================
def calculate_indicators(df):
    if df is None or len(df) < 20: return df
    df = df.copy()
    
    # 均线
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean() # 黄线
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
    
    df.fillna(method='bfill', inplace=True)
    return df

# ==========================================
# 3. 评分系统 (加入“蜜雪集团”洗盘逻辑)
# ==========================================
def rank_stock(df, name, current, symbol, money_flow):
    if df is None or len(df) < 20: return 0, "数据不足", "跳过", "#888"
    
    last = df.iloc[-1]
    
    # --- 变量准备 ---
    vol_ratio = last['volume'] / last['vol_max20'] if last['vol_max20'] > 0 else 0
    vol_ma5_ratio = last['volume'] / last['vol_ma5'] if last['vol_ma5'] > 0 else 0
    j_val = last['J']
    dist_yellow = abs(last['close'] - last['大哥黄线']) / last['大哥黄线'] * 100
    
    is_red = last['close'] >= last['open'] # 收阳
    is_green = last['close'] < last['open'] # 收阴
    
    base_score = 60.0 # 只要入围就是60分
    signal_type = "普通观察"
    details = []

    # --- 判定模式：A.缩量B1 vs B.洗盘B1 ---

    # 模式A：经典缩量 B1
    is_shrink = vol_ratio < 0.5
    is_super_shrink = vol_ratio < 0.3
    
    if is_shrink and j_val < 0:
        base_score = 75.0
        signal_type = "🟢 经典缩量B1"
        if is_super_shrink:
            base_score = 80.0
            signal_type = "💎 极致缩量B1"
            details.append("窒息量(主力锁仓)")
    
    # 模式B：恐慌洗盘 B1 (蜜雪集团模式)
    # 条件：放量(量比>1) + 收阴 + 回踩黄线不破 + J值极低
    is_panic_wash = (vol_ma5_ratio > 1.0) and is_green and (last['close'] > last['大哥黄线']) and (j_val < -5)
    
    if is_panic_wash:
        base_score = 85.0 # 给高分！
        signal_type = "🩸 恐慌洗盘B1 (带血筹码)"
        details.append("放量绿柱未破位")
        details.append("主力借势洗盘")
        
    # --- 2. 细节加分 ---
    quality_score = 0.0
    
    # J值越低反弹越猛
    if j_val < -10:
        quality_score += 8.0
        details.append("J值极致超卖")
    elif j_val < -5:
        quality_score += 4.0
        
    # 精准回踩
    if dist_yellow < 1.0:
        quality_score += 5.0
        details.append("精准踩黄线")

    # --- 3. 资金面修正 (只加不减) ---
    bonus_score = 0.0
    
    if money_flow > 0.5: 
        bonus_score += 15.0
        details.append("主力大举买入")
    elif money_flow > 0:
        bonus_score += 5.0
        details.append("资金翻红")
    # 注意：流出不再扣分！
    
    # 低价股活跃加分
    if current < 12 and vol_ma5_ratio > 1.2:
        bonus_score += 5.0
        details.append("低价活跃")

    # --- 算总分 ---
    total_score = base_score + quality_score + bonus_score
    total_score = min(99.0, total_score)
    
    # --- 生成评论 ---
    comment = f"定性：**{signal_type}**\n"
    if details:
        comment += f"亮点：{', '.join(details)}\n"
    
    comment += f"资金：{'🟥 流入' if money_flow>0 else '🟩 流出'} {abs(money_flow):.2f} 亿"
    if money_flow < 0 and is_panic_wash:
        comment += " (洗盘流出，不扣分)"
        
    if total_score >= 85:
        advice = "极品！洗盘到位或缩量极致。"
        color = "#ff2b2b" # 红
    elif total_score >= 70:
        advice = "优质。形态良好。"
        color = "#ff9800" # 橙
    else:
        advice = "一般。暂无强信号。"
        color = "#888888" # 灰
        
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
st.title("🏆 浩哥战法：优中选优 PK 终端 (v6.0 终极版)")
st.markdown("### 已加入【洗盘B1】逻辑：放量绿柱+回踩黄线不破+超卖 = 高分！")

# 侧边栏
with st.sidebar:
    st.header("候选股票池")
    st.caption("支持空格、换行、逗号分隔")
    # 这里允许用户直接粘贴那堆乱七八糟的数据
    codes_input = st.text_area("粘贴代码", height=300)
    run_btn = st.button("开始 PK 排名", type="primary")

if run_btn:
    # 智能清洗输入数据：用正则匹配所有连续的数字串
    codes = re.findall(r'\d{6}', codes_input)
    
    if not codes:
        st.error("没找到有效的股票代码，请检查输入！")
    else:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(codes):
            status_text.text(f"正在分析 {i+1}/{len(codes)}: {symbol} ...")
            
            # 这里的 fetch 已经包含了防反爬和 AkShare 兜底
            df = fetch_history_data(symbol)
            
            if df is not None:
                name = get_stock_name(symbol)
                current = get_real_time_price(symbol, df)
                money = get_money_flow(symbol)
                
                score, comment, advice, color = rank_stock(df, name, current, symbol, money)
                
                results.append({
                    "code": symbol, "name": name, "score": score, 
                    "comment": comment, "advice": advice, "color": color,
                    "df": df
                })
            else:
                # 拉取失败的默默跳过，不报错
                pass
                
            progress_bar.progress((i + 1) / len(codes))
        
        status_text.text("分析完成！正在排序...")
        
        # 排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        st.success(f"PK 完成！有效分析 {len(results)} 只股票。")
        
        # 展示前 50 名 (防止页面太长卡死)
        for rank, res in enumerate(results):
            with st.container():
                c1, c2, c3 = st.columns([1.2, 3, 1.5])
                
                with c1:
                    st.markdown(f"### 第 {rank+1} 名")
                    st.markdown(f"<h1 style='color: {res['color']}'>{res['score']:.1f}</h1>", unsafe_allow_html=True)
                    st.caption(f"{res['name']} ({res['code']})")
                
                with c2:
                    st.info(res['comment'])
                
                with c3:
                    st.markdown(f"### {res['advice']}")
                    with st.expander("K线图"):
                        st.plotly_chart(plot_kline(res['df'], res['code'], res['name']), use_container_width=True)
                
            st.divider()
