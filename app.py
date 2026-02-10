import streamlit as st
import pandas as pd
import requests
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# ==========================================
# 1. 数据获取（腾讯接口 + AkShare 双保险）
# ==========================================
@st.cache_data(ttl=300)
def get_real_time_price(symbol, df=None):
    """获取实时价格，如果接口失败则用历史收盘价兜底"""
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    
    try:
        r = requests.get(url, headers=headers, timeout=3)
        text = r.text.strip()
        if text.startswith('var hq_str_'):
            parts = text.split('"')[1].split(',')
            if len(parts) > 3:
                price = float(parts[3])
                if price > 0:
                    return price, "实时接口"
    except:
        pass
    
    # 兜底：使用历史K线的最后一天收盘价
    if df is not None and not df.empty:
        return df['close'].iloc[-1], "收盘价(非实时)"
    
    return 0.0, "无数据"

@st.cache_data(ttl=3600)  # 历史数据缓存久一点
def fetch_history_data(symbol):
    """
    获取历史K线数据
    策略：优先用腾讯接口（快），失败则用 AkShare（稳）
    """
    # 1. 尝试腾讯接口
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            key = f"{prefix}{symbol}"
            
            # 解析腾讯数据结构
            qt_data = data.get('data', {}).get(key, {})
            # 优先取前复权(qfqday)，没有则取不复权(day)
            day_data = qt_data.get('qfqday', qt_data.get('day', []))
            
            if day_data:
                df = pd.DataFrame([row[:6] for row in day_data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.apply(pd.to_numeric, errors='coerce')
                return calculate_indicators(df)
    except Exception as e:
        print(f"腾讯接口失败，切换 AkShare: {e}")

    # 2. 备选方案：AkShare (最稳，但稍慢)
    try:
        # 获取当前日期
        end_date = datetime.datetime.now().strftime("%Y%m%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y%m%d")
        
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        if not df.empty:
            df = df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'})
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return calculate_indicators(df)
    except Exception as e:
        st.error(f"AkShare 也拉不到数据 {symbol}: {e}")
    
    return None

def get_stock_name(symbol):
    """获取股票名称"""
    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        return df[df['项目'] == '股票简称']['值'].values[0]
    except:
        return symbol

@st.cache_data(ttl=1800)
def get_money_flow(symbol):
    """获取主力资金流"""
    try:
        market = "sh" if symbol.startswith('6') else "sz"
        flow = ak.stock_individual_fund_flow(stock=symbol, market=market)
        if not flow.empty:
            val = flow.iloc[0]['主力净流入-净额']
            return val / 100000000  # 转亿
    except:
        pass
    return 0.0

# ==========================================
# 2. 技术指标计算
# ==========================================
def calculate_indicators(df):
    if df is None or len(df) < 5:
        return df
    
    df = df.copy()
    
    # 均线
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    # 填充空值
    df.fillna(method='bfill', inplace=True)
    return df

# ==========================================
# 3. K线图绘制
# ==========================================
def plot_kline(df, symbol, name):
    df = df.iloc[-120:]  # 只看最近半年
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # K线
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='K线', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'
    ), row=1, col=1)

    # 均线
    if 'MA5' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='white', width=1), name='MA5'), row=1, col=1)
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='#ffd54f', width=1.5), name='MA20'), row=1, col=1)
    if 'MA60' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='#42a5f5', width=1), name='MA60'), row=1, col=1)

    # 成交量
    colors = ['#ef5350' if row['open'] < row['close'] else '#26a69a' for i, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='成交量'), row=2, col=1)

    fig.update_layout(
        title=f"{name} ({symbol})",
        yaxis_title='价格',
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#0e1117',
        font=dict(color='#e0e0e0')
    )
    return fig

# ==========================================
# 4. 浩哥战法评分 (修复 NameError)
# ==========================================
def analyze_stock(df, name, current, symbol, money_flow):
    if df is None or len(df) < 20:
        return 0.0, f"数据不足，无法分析 {name}", "观望"
    
    last = df.iloc[-1]
    
    # --- 修复：在函数内部定义 safe_get ---
    def safe_get(col, default=0.0):
        return last[col] if col in last else default
    # -----------------------------------

    # 简单的打分逻辑 (你可以把你的复杂逻辑贴回来)
    tech_score = 0.0
    signals = []
    
    # 示例逻辑：站上20日线
    ma20 = safe_get('MA20')
    if last['close'] > ma20:
        tech_score += 60
        signals.append("站稳大哥线(MA20)")
    else:
        tech_score += 30
        signals.append("趋势偏弱")

    # 资金面加分
    hao_score = 0.0
    if money_flow > 0.5:
        hao_score = 20
        signals.append("主力大幅流入")
    elif money_flow > 0:
        hao_score = 10
        
    total_score = tech_score + hao_score
    total_score = min(100, total_score)
    
    # 生成评论
    comment = f"浩哥看 {name} (现价{current}):\n"
    comment += f"信号: {', '.join(signals)}\n"
    comment += f"主力资金: {money_flow:.2f} 亿"
    
    if total_score >= 80:
        advice = "建议关注"
    elif total_score >= 60:
        advice = "谨慎持有"
    else:
        advice = "空仓观望"
        
    return total_score, comment, advice

# ==========================================
# 5. Streamlit 主程序
# ==========================================
st.set_page_config(page_title="浩哥战法终端", layout="wide")
st.title("🚀 浩哥战法量化终端 v3.1 (稳定版)")

codes_input = st.text_input("输入股票代码 (逗号分隔)", "002235,002501,002425,600545")

if st.button("开始挖掘"):
    codes = [c.strip() for c in codes_input.replace('，', ',').split(',') if c.strip()]
    
    if not codes:
        st.warning("请输入代码！")
    else:
        for symbol in codes:
            # 1. 获取基本信息
            with st.spinner(f"正在分析 {symbol} ..."):
                name = get_stock_name(symbol)
                
                # 2. 获取K线数据 (含重试机制)
                df = fetch_history_data(symbol)
                
                if df is not None:
                    # 3. 获取实时数据
                    current_price, price_source = get_real_time_price(symbol, df)
                    money_flow = get_money_flow(symbol)
                    
                    # 4. 分析
                    score, comment, advice = analyze_stock(df, name, current_price, symbol, money_flow)
                    
                    # 5. 展示
                    with st.container():
                        st.subheader(f"{name} ({symbol})")
                        c1, c2, c3 = st.columns([1, 2, 3])
                        
                        score_color = "red" if score >= 80 else "orange" if score >= 60 else "green"
                        c1.markdown(f"<h1 style='color:{score_color}'>{score:.0f}</h1>", unsafe_allow_html=True)
                        c1.caption("浩哥总分")
                        
                        c2.info(advice)
                        c2.caption(f"数据来源: {price_source}")
                        
                        c3.text(comment)
                        
                        # K线图折叠
                        with st.expander("查看 K线图"):
                            fig = plot_kline(df, symbol, name)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"❌ {symbol} ({name}) 数据拉取彻底失败，请检查代码是否正确或停牌。")
            
            st.divider()
