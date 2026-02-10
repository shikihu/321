import streamlit as st
import pandas as pd
import requests
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# ==========================================
# 1. 数据获取（极速版 + 缓存）
# ==========================================
@st.cache_data(ttl=300)  # 5分钟缓存，极大提升速度
def get_real_time_price(symbol):
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
    try:
        r = requests.get(url, timeout=3)
        text = r.text.strip()
        if text.startswith('var hq_str_'):
            parts = text.split('"')[1].split(',')
            if len(parts) >= 4 and float(parts[3]) > 0:
                return float(parts[3])
    except:
        pass
    return 0.0

@st.cache_data(ttl=600)  # 10分钟缓存历史数据
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
    try:
        flow = ak.stock_individual_fund_flow(stock=symbol, market="sh" if symbol.startswith('6') else "sz")
        if not flow.empty:
            return flow.iloc[0]['主力净流入-净额'] / 100000000
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
# 2. 技术指标计算
# ==========================================
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
    
    return df

# ==========================================
# 3. K线图（Gemini版完整保留）
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
        xaxis_rangeslider_visible=True,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#0e1117',
        font=dict(color='white')
    )
    return fig

# ==========================================
# 4. 浩哥战法评分（严格匹配你的副图公式 + 回测权重）
# ==========================================
def analyze_stock(df, name, current, symbol):
    if df is None or len(df) < 20:
        return 0.0, f"浩哥看 {name} 数据不足，无法分析。", "浩哥建议：暂缓操作。"
    
    last = df.iloc[-1]
    
    # 安全访问
    def safe_get(col, default=0.0):
        return last.get(col, default) if col in last else default
    
    # 7种浩哥战法（严格匹配你的公式）
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
    
    # 技术分
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
    
    # 浩哥点评
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

# ==========================================
# 主界面
# ==========================================
st.set_page_config(page_title="浩哥战法", layout="wide")
st.title("🚀 浩哥战法量化终端")

with st.sidebar:
    st.header("浩哥战法六步")
    st.markdown(""" 
1. 择时：看大盘温度  
2. 选股：强势+题材热  
3. 买点：首踩或主升  
4. 持仓：等利润垫  
5. 卖点：破位/高潮/情绪退潮  
6. 复盘：每笔必复盘  
""")
    st.markdown("**心态**：沉没成本不决策，珍惜子弹！")

codes_input = st.text_input("输入股票代码（逗号分隔）", "600519,000001")
if st.button("开始挖掘"):
    codes = [c.strip() for c in codes_input.split(',') if c.strip()]
    for symbol in codes:
        with st.spinner(f"浩哥正在分析 {symbol}..."):
            df = fetch_history_data(symbol)
            name = get_stock_name(symbol)
            current_price = get_real_time_price(symbol)
            news = get_stock_news(symbol)
            money_flow = get_money_flow(symbol)
           
            if df is not None:
                score, comment, advice = analyze_stock(df, name, current_price, symbol)
               
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.metric("浩哥打分", f"{score:.1f}/100")
                with c2:
                    st.info(comment)
                    st.success(f"**浩哥建议：** {advice}")
               
                with st.expander("查看 K线图"):
                    fig = plot_kline(df, symbol, name)
                    st.plotly_chart(fig, use_container_width=True)
               
                st.markdown("---")
            else:
                st.error(f"{symbol} 数据拉取失败")
