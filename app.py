import streamlit as st
import pandas as pd
import requests
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 数据获取（极速 + 价格兜底）
# ==========================================
@st.cache_data(ttl=300)
def get_real_time_price(symbol, df=None):
    """
    获取实时价格
    返回: (price, source_msg)
    """
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
    try:
        r = requests.get(url, timeout=3)
        text = r.text.strip()
        if text.startswith('var hq_str_'):
            parts = text.split('"')[1].split(',')
            if len(parts) > 3:
                price = float(parts[3])
                if price > 0:
                    return price, "实时价"
    except:
        pass
    
    # 兜底用历史收盘价
    if df is not None and len(df) > 0:
        return df['close'].iloc[-1], "(非交易时间/最近收盘价)"
    return 0.0, "暂无数据"

@st.cache_data(ttl=600)
def fetch_history_data(symbol):
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
    try:
        r = requests.get(url, timeout=6).json()
        data = r.get('data', {}).get(f"{prefix}{symbol}", {}).get('qfqday', [])
        if not data:
            return None
        df = pd.DataFrame([row[:6] for row in data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        # 强制转为数字，处理非数字字符
        df = df.apply(pd.to_numeric, errors='coerce')
        return calculate_indicators(df)
    except:
        return None

def get_stock_name(symbol):
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
    try:
        r = requests.get(url, timeout=3)
        text = r.text.strip()
        if text.startswith('var hq_str_'):
            parts = text.split('"')[1].split(',')
            if len(parts) >= 2:
                return parts[0].strip()
    except:
        pass
    return symbol

@st.cache_data(ttl=1800)
def get_stock_news(symbol):
    try:
        news = ak.stock_news_em(symbol=symbol)
        return news.head(3)[['标题', '发布时间', '来源']].to_dict('records')
    except:
        return []

@st.cache_data(ttl=1800)
def get_money_flow(symbol):
    """
    获取主力资金净流入（单位：亿）
    """
    try:
        # 注意：这里获取的是历史资金流，如果今日未收盘，可能拿到的是昨日的
        # 如需实时资金流，接口会更复杂，这里先用通用接口
        market = "sh" if symbol.startswith('6') else "sz"
        flow = ak.stock_individual_fund_flow(stock=symbol, market=market)
        if not flow.empty:
            # 通常第一行是最新的
            latest_flow = flow.iloc[0]['主力净流入-净额']
            return latest_flow / 100000000 # 转为亿元
        return 0.0
    except:
        return 0.0

@st.cache_data(ttl=1800)
def get_lhb_data(symbol):
    try:
        lhb = ak.stock_lhb_detail_em(symbol=symbol)
        if not lhb.empty:
            latest = lhb.iloc[0]
            net_amount = latest.get('净买入额(万元)', 0) / 10000
            return net_amount
        return 0.0
    except:
        return 0.0

# ==========================================
# 技术指标计算
# ==========================================
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
    
    # MA20 (浩哥超级战法用到)
    df['MA20'] = df['close'].rolling(window=20).mean()
    # MA60 (画图用到)
    df['MA60'] = df['close'].rolling(window=60).mean()

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
    
    return df

# ==========================================
# K线图
# ==========================================
def plot_kline(df, symbol, name):
    df = df.iloc[-120:]
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='K线', increasing_line_color='red', decreasing_line_color='green'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='white', width=1), name='白线(MA5)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='yellow', width=1.5), name='黄线(大哥线)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue', width=1), name='生命线(MA60)'), row=1, col=1)
    
    colors = ['red' if row['open'] < row['close'] else 'green' for i, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='成交量'), row=2, col=1)
    
    fig.update_layout(
        title=f"{name} ({symbol}) - 浩哥专用图表",
        yaxis_title='价格',
        xaxis_rangeslider_visible=False, # 关掉slider更美观
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#0e1117',
        font=dict(color='white')
    )
    return fig

# ==========================================
# 浩哥战法评分 (修复版)
# ==========================================
# 【关键修复】增加了 money_flow 参数
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
    
    # 回踩
    dist_white = abs(last['close'] - trend_white) / last['close'] * 100
    back_white = (last['close'] >= trend_white and dist_white <= 2) or (last['close'] < trend_white and dist_white < 0.8)
    
    dist_yellow = abs(last['close'] - brother_yellow) / brother_yellow * 100
    back_yellow = (last['close'] >= brother_yellow and dist_yellow <= 1.5) or (last['close'] < brother_yellow and dist_yellow <= 0.8)
    
    # 7种浩哥战法信号
    signals = {}
    
    # 1. 浩哥缩量战法
    if do_up_trend and safe_get('J', 0) < 14 and shrink and safe_get('当日振幅', 999) < 8 and safe_get('当日涨跌幅', 999) < 2.5:
        signals['浩哥缩量战法'] = True
    
    # 2. 浩哥极缩战法
    if do_up_trend and safe_get('J', 0) < 14 and super_shrink and safe_get('当日振幅', 999) < 8:
        signals['浩哥极缩战法'] = True
    
    # 3. 浩哥拐头战法
    if 'rsi' in df:
        rsi_prev = df['rsi'].shift(1).iloc[-1] if len(df) > 1 else 50
        if do_up_trend and (safe_get('rsi', 50) - 15 >= rsi_prev) and (rsi_prev < 20) and safe_get('当日振幅', 999) < 8 and shrink:
            signals['浩哥拐头战法'] = True
    
    # 4. 浩哥白线战法
    if do_up_trend and back_white and back_shrink and safe_get('J', 0) < 30 and safe_get('当日振幅', 999) < 8.5:
        signals['浩哥白线战法'] = True
    
    # 5. 浩哥超级战法
    if safe_get('close', 0) > safe_get('MA20', 0) * 1.05 and proper_shrink and safe_get('J', 0) < 35 and back_white:
        signals['浩哥超级战法'] = True
    
    # 6. 浩哥黄线战法 (修复语法错误)
    yellow_prev = df['大哥黄线'].shift(1).iloc[-1] if '大哥黄线' in df else 0
    if back_yellow and shrink and (brother_yellow >= yellow_prev * 0.997):
        signals['浩哥黄线战法'] = True
    
    # 7. 浩哥1.0战法
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
    
    # 技术分计算
    tech_score = 0.0
    triggered_signals = []
    for sig, active in signals.items():
        if active:
            tech_score += weights[sig]
            triggered_signals.append(sig)
    
    # J值额外加分 (J越负越好)
    j_val = safe_get('J', 0)
    if j_val < 0:
        tech_score += min(abs(j_val) * 0.2, 3.0)

    # 低价股复活机制
    price_correction = 0.0
    is_resurrected = False
    
    if current < 12:
        base_penalty = -5.0
        # 复活条件: 量比活跃 或 站稳大哥线
        vol_active = (safe_get('volume', 0) / safe_get('VOL5', 1) > 1.5)
        price_strong = (last['close'] > brother_yellow * 1.02)
        
        if vol_active or price_strong:
            price_correction = +2.0 # 不扣反加
            is_resurrected = True
        else:
            price_correction = base_penalty
    else:
        # 12-50元 甜点区
        if 12 <= current <= 50:
            price_correction = +2.0

    tech_score += price_correction
    tech_score = min(max(tech_score, 0), 70.0)
    
    # AI 资金分 (修复：现在使用传入的 money_flow)
    ai_score = 0.0
    lhb_net = get_lhb_data(symbol) # 龙虎榜
    
    # 优先用主力净流入，如果没有则参考龙虎榜
    real_flow = money_flow if abs(money_flow) > 0 else lhb_net
    
    if real_flow > 0.5: # 流入超0.5亿
        ai_score += min(real_flow * 5, 15.0)
    elif real_flow > 0.1:
        ai_score += 5.0
    elif real_flow < -0.5:
        ai_score -= min(abs(real_flow) * 5, 10.0)
        
    total_score = tech_score + ai_score
    total_score = min(max(total_score, 0), 100.0)
    
    # 生成评论
    comment = f"浩哥对 {name} 的判断：现价 {current:.2f} 元。\n\n"
    
    if triggered_signals:
        comment += f"🔥 **关键信号触发**：{ ' + '.join(triggered_signals) }\n"
    else:
        comment += "👀 暂未触发核心战法信号。\n"
        
    if is_resurrected:
        comment += "🚀 **低价股复活**：股价虽低，但资金/形态活跃，浩哥判定有机会！\n"
    
    comment += f"💰 **资金面**：主力净流入 {real_flow:.2f} 亿。\n"
    
    if total_score >= 85:
        advice = "🚀 **建议**：重仓出击，形态完美+资金流入。"
    elif total_score >= 70:
        advice = "👍 **建议**：中仓买入，信号不错，值得博弈。"
    elif total_score >= 50:
        advice = "🤔 **建议**：轻仓观察，风险收益比一般。"
    else:
        advice = "🛑 **建议**：空仓观望，形态走坏或资金流出。"
    
    return total_score, comment, advice

# ==========================================
# 主界面逻辑
# ==========================================
st.set_page_config(page_title="浩哥战法", layout="wide")
st.title("🚀 浩哥战法量化终端 v3.0 (修复版)")

with st.sidebar:
    st.header("浩哥战法")
    st.info("已修复：资金流显示 0.00 问题\n已修复：低价股复活逻辑")

codes_input = st.text_input("输入股票代码（逗号分隔）", "600519,000001,601138")

if st.button("开始挖掘"):
    codes = [c.strip() for c in codes_input.split(',') if c.strip()]
    for symbol in codes:
        with st.spinner(f"浩哥正在分析 {symbol}..."):
            # 1. 获取数据
            df = fetch_history_data(symbol)
            name = get_stock_name(symbol)
            
            # 【关键修复】这里把元组解包了
            current_price_data = get_real_time_price(symbol, df)
            current_price = current_price_data[0] # 只取价格数字
            
            news = get_stock_news(symbol)
            money_flow = get_money_flow(symbol)
           
            if df is not None:
                # 【关键修复】把 money_flow 传进去了
                score, comment, advice = analyze_stock(df, name, current_price, symbol, money_flow)
               
                # 结果展示
                with st.container():
                    score_color = "red" if score >= 80 else "orange" if score >= 60 else "green"
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.markdown(f"""
                        <div style="border: 2px solid {score_color}; border-radius: 10px; text-align: center; padding: 10px;">
                            <h1 style="color: {score_color}; margin:0">{score:.1f}</h1>
                            <small>浩哥总分</small>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        st.info(comment)
                        st.subheader(advice)
               
                with st.expander("📈 查看 K线图"):
                    fig = plot_kline(df, symbol, name)
                    st.plotly_chart(fig, use_container_width=True)
               
                st.markdown("---")
            else:
                st.error(f"{symbol} 数据拉取失败")
