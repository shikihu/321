import streamlit as st
import pandas as pd
import requests
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# ==========================================
# 1. 稳健的数据获取 (保留修复后的防报错逻辑)
# ==========================================
@st.cache_data(ttl=10)
def get_real_time_price(symbol, df=None):
    """获取实时价格，带反爬虫头"""
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
    headers = {'User-Agent': 'Mozilla/5.0'} 
    try:
        r = requests.get(url, headers=headers, timeout=3)
        parts = r.text.split('"')[1].split(',')
        if len(parts) > 3:
            price = float(parts[3])
            if price > 0:
                return price, "实时接口"
    except:
        pass
    
    if df is not None and not df.empty:
        return df['close'].iloc[-1], "(非交易时间/最近收盘价)"
    return 0.0, "无数据"

@st.cache_data(ttl=3600)
def fetch_history_data(symbol):
    """获取历史数据 (腾讯接口 + AkShare兜底)"""
    # 方案A: 腾讯接口
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            key = f"{prefix}{symbol}"
            # 兼容腾讯返回格式
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

    # 方案B: AkShare 兜底
    try:
        end_date = datetime.datetime.now().strftime("%Y%m%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
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
            val = flow.iloc[0]['主力净流入-净额']
            return val / 100000000 
    except:
        pass
    return 0.0

# ==========================================
# 2. 详细技术指标 (恢复以前的逻辑)
# ==========================================
def calculate_indicators(df):
    if df is None or len(df) < 5:
        return df
    
    df = df.copy()
    
    # 均线系统
    df['MA5'] = df['close'].rolling(5).mean()     # 白线
    df['MA20'] = df['close'].rolling(20).mean()   # 黄线(大哥线)
    df['MA60'] = df['close'].rolling(60).mean()   # 生命线
    
    # 趋势白线 & 大哥黄线 (你的特定公式)
    df['趋势白线'] = df['close'].ewm(span=9, adjust=False).mean().ewm(span=11, adjust=False).mean()
    df['大哥黄线'] = (df['close'].ewm(span=7, adjust=False).mean().ewm(span=7, adjust=False).mean() + 
                       df['close'].ewm(span=14, adjust=False).mean().ewm(span=14, adjust=False).mean() + 
                       df['close'].ewm(span=28, adjust=False).mean().ewm(span=28, adjust=False).mean() + 
                       df['close'].ewm(span=56, adjust=False).mean().ewm(span=56, adjust=False).mean()) / 4

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = exp1 - exp2
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = 2 * (df['dif'] - df['dea'])

    # KDJ
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['rsi'] = 100 - (100 / (1 + rs))

    # 缩量计算
    df['vol_max20'] = df['volume'].rolling(20).max()
    df['vol_max50'] = df['volume'].rolling(50).max()
    
    # 填充空值以免计算报错
    df.fillna(method='bfill', inplace=True)
    return df

# ==========================================
# 3. K线图 (保持 Plotly)
# ==========================================
def plot_kline(df, symbol, name):
    df = df.iloc[-120:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='white', width=1), name='白线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='yellow', width=1.5), name='大哥线'), row=1, col=1)
    colors = ['red' if row['open'] < row['close'] else 'green' for i, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='成交量'), row=2, col=1)
    fig.update_layout(title=f"{name} ({symbol})", height=500, xaxis_rangeslider_visible=False, plot_bgcolor='#1e1e1e', paper_bgcolor='#0e1117', font=dict(color='white'))
    return fig

# ==========================================
# 4. 核心分析逻辑 (完全恢复你的 7大B1 战法 + 详细文本)
# ==========================================
def analyze_stock(df, name, current, symbol, money_flow):
    if df is None or len(df) < 20:
        return 0.0, f"数据不足", "观望"
    
    last = df.iloc[-1]
    
    # 安全取值辅助函数
    def safe_get(col, default=0.0):
        return last[col] if col in last else default

    # --- 1. 信号判定 (恢复之前的逻辑) ---
    signals = {}
    
    # 变量准备
    trend_white = safe_get('趋势白线', last['close'])
    brother_yellow = safe_get('大哥黄线', last['close'])
    vol = safe_get('volume')
    vol20_max = safe_get('vol_max20')
    vol50_max = safe_get('vol_max50')
    
    # 缩量判定
    shrink = (vol < vol20_max * 0.416) or (vol < vol50_max / 3)
    proper_shrink = (vol < vol20_max * 0.618)
    super_shrink = (vol < vol20_max * 0.25)
    
    # 趋势判定
    do_up_trend = (last['close'] > brother_yellow)
    
    # 回踩判定
    dist_white = abs(last['close'] - trend_white) / last['close'] * 100
    back_white = dist_white < 2.0
    
    # --- 7种战法判断 ---
    j_val = safe_get('J')
    
    if do_up_trend and j_val < 14 and shrink:
        signals['浩哥缩量战法'] = True
    if do_up_trend and j_val < 14 and super_shrink:
        signals['浩哥极缩战法'] = True
    if do_up_trend and back_white and shrink and j_val < 30:
        signals['浩哥白线战法'] = True
    if last['close'] > safe_get('MA20') * 1.05 and proper_shrink and j_val < 35 and back_white:
        signals['浩哥超级战法'] = True
    if trend_white > brother_yellow and j_val < 13 and proper_shrink:
        signals['浩哥1.0战法'] = True
        
    # 权重
    weights = {
        '浩哥超级战法': 25.0, '浩哥极缩战法': 22.0, '浩哥白线战法': 18.0,
        '浩哥1.0战法': 15.0, '浩哥缩量战法': 5.0
    }
    
    tech_score = 0.0
    triggered = []
    for sig, active in signals.items():
        if active:
            tech_score += weights.get(sig, 0)
            triggered.append(sig)
    
    # --- 2. 浩哥评分 (资金与低价) ---
    hao_score = 0.0
    
    # 资金流加分
    if money_flow > 0.5: hao_score += 15
    elif money_flow > 0: hao_score += 5
    elif money_flow < -0.5: hao_score -= 5
    
    # 低价股复活
    if current < 12 and (money_flow > 0.1 or last['close'] > safe_get('MA20') * 1.02):
        hao_score += 5
        
    # J值超卖加分
    if j_val < 0: hao_score += 5

    # 限制分数
    tech_score = min(70, tech_score)
    hao_score = min(30, hao_score)
    total_score = tech_score + hao_score

    # --- 3. 生成详细文本 (技术面观察) ---
    obs = []
    # MACD 描述
    macd = safe_get('macd')
    if macd > 0: obs.append("MACD 柱线翻红，短期动能有修复迹象")
    else: obs.append("MACD 绿柱状态，动能偏弱")
    
    # 量能描述
    if shrink: obs.append("量能持续萎缩，属于典型缩量调整形态")
    elif vol > safe_get('volume', 0) * 1.5: obs.append("今日明显放量，资金分歧较大")
    else: obs.append("成交量温和")
    
    # 均线描述
    if last['MA5'] > last['MA20'] > last['MA60']:
        obs.append("短期均线多头排列，趋势结构仍保持完整")
    elif last['close'] < last['MA20']:
        obs.append("股价跌破大哥黄线，注意风险")
        
    obs_text = "；".join(obs) + "。"

    # --- 4. 生成评论与建议 ---
    comment = f"浩哥对 {name} 的综合判断：当前价 {current:.2f} 元。\n\n"
    if triggered:
        comment += f"浩哥今天检测到关键信号：**{', '.join(triggered)}**，形态不错！\n\n"
    else:
        comment += f"浩哥今天未检测到关键信号，形态未到最佳点。\n\n"
        
    comment += f"【技术面评分】 {tech_score:.1f}/70  【浩哥评分】 {hao_score:.1f}/30  【浩哥综合打分】 {total_score:.1f}/100\n\n"
    comment += f"技术面观察：{obs_text}\n\n"
    comment += f"💰 资金面：主力净流入 {money_flow:.2f} 亿。浩哥认为{'当前风险大于机会' if total_score < 60 else '当前有机会'}，{'形态和情绪均未到位' if total_score < 60 else '值得关注'}，{'短期不宜重仓' if total_score < 60 else '可尝试建仓'}。"

    if total_score >= 80:
        advice = "浩哥建议：重仓出击，形态资金共振！"
    elif total_score >= 60:
        advice = "浩哥建议：分批低吸，控制仓位。"
    else:
        advice = "浩哥建议：暂时回避，保护本金，等待更清晰的信号。"
        
    return total_score, comment, advice

# ==========================================
# 5. 主界面 (恢复左右布局)
# ==========================================
st.set_page_config(page_title="浩哥战法", layout="wide")
st.title("🚀 浩哥战法量化终端 v4.0 (完整复刻版)")

codes_input = st.text_input("输入股票代码 (逗号分隔)", "600519,000001")

if st.button("开始挖掘"):
    codes = [c.strip() for c in codes_input.replace('，', ',').split(',') if c.strip()]
    for symbol in codes:
        with st.spinner(f"浩哥正在深度分析 {symbol}..."):
            df = fetch_history_data(symbol)
            name = get_stock_name(symbol)
            current_price, price_source = get_real_time_price(symbol, df)
            money_flow = get_money_flow(symbol)
            
            if df is not None:
                score, comment, advice = analyze_stock(df, name, current_price, symbol, money_flow)
                
                # 恢复你喜欢的左右布局
                c1, c2 = st.columns([1, 3])
                
                with c1:
                    # 巨大的分数显示
                    st.markdown("### 浩哥打分")
                    st.markdown(f"<h1 style='font-size: 60px; color: #333;'>{score:.1f}/100</h1>", unsafe_allow_html=True)
                
                with c2:
                    # 蓝色的详细点评框
                    st.info(comment + f"\n\n价格来源: {price_source}")
                    # 绿色的建议框
                    if score >= 60:
                        st.success(advice)
                    else:
                        st.error(advice) # 低分用红色警告，或者你如果喜欢绿色也可以统一用success
                
                # K线图
                with st.expander("查看 K线图"):
                    fig = plot_kline(df, symbol, name)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
            else:
                st.error(f"{symbol} 数据拉取失败，请检查代码。")
