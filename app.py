import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import numpy as np

# 函数：获取数据
def fetch_stock_data(symbol):
    ticker = f"{symbol}.SS" if symbol.startswith('6') else f"{symbol}.SZ"
    stock = yf.Ticker(ticker)
    info = stock.info
    name = info.get('shortName', '未知股票')
    current = info.get('currentPrice', 0.0)
    market_cap = info.get('marketCap', 0) / 100000000  # 亿
    
    hist = stock.history(period="1y", interval="1d")
    if hist.empty:
        return None, name, current, market_cap
    
    df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL']
    df['C'] = df['CLOSE']
    df['O'] = df['OPEN']
    df['H'] = df['HIGH']
    df['L'] = df['LOW']
    df['V'] = df['VOL']
    
    return df, name, current, market_cap

# 函数：计算指标
def calculate_indicators(df):
    # 参数
    Z1 = 2
    Z2 = 0
    最小流通值 = 50
    N = 20
    M = 50
    
    # BBI
    df['BBI'] = (df['C'].rolling(3).mean() + df['C'].rolling(6).mean() + 
                 df['C'].rolling(12).mean() + df['C'].rolling(24).mean()) / 4
    
    # 趋势白线
    df['趋势白线'] = df['C'].ewm(span=9, adjust=False).mean().ewm(span=11, adjust=False).mean()
    
    # 大哥黄线
    df['大哥黄线'] = (df['C'].ewm(span=7, adjust=False).mean().ewm(span=7, adjust=False).mean() + 
                   df['C'].ewm(span=14, adjust=False).mean().ewm(span=14, adjust=False).mean() + 
                   df['C'].ewm(span=28, adjust=False).mean().ewm(span=28, adjust=False).mean() + 
                   df['C'].ewm(span=56, adjust=False).mean().ewm(span=56, adjust=False).mean()) / 4
    
    # 振幅区间原始
    code_prefix = symbol[0:2]  # 假设 symbol 是字符串
    is_special = code_prefix in ['68', '30', '4', '8', '9'] or (df['C'] / df['C'].shift(1) > 1.15).any()
    振幅区间原始 = 8 if is_special else 5
    振幅区间 = 振幅区间原始 + Z2
    涨跌放宽系数 = 0.9 if is_special else 1
    df['当日振幅'] = (df['H'] - df['L']) / df['L'] * 100
    df['当日涨跌幅'] = abs(df['C'] - df['C'].shift(1)) / df['C'].shift(1) * 100 * 涨跌放宽系数
    df['上涨十字星'] = (df['C'] > df['C'].shift(1)) & (abs(df['C'] - df['O']) / df['O'] * 100 * 涨跌放宽系数 < 1.8)
    
    # SHORT 和 LONG
    df['SHORT'] = 100 * (df['C'] - df['L'].rolling(3).min()) / (df['H'].rolling(3).max() - df['L'].rolling(3).min())
    df['LONG'] = 100 * (df['C'] - df['L'].rolling(21).min()) / (df['H'].rolling(21).max() - df['L'].rolling(21).min())
    df['单针下20'] = (df['SHORT'] <= 20 & df['LONG'] >= 75) | ((df['LONG'] - df['SHORT']) >= 70)
    df['聚宝盆'] = (df['LONG'] >= 75).rolling(8).sum() >= 6 & (df['SHORT'] <= 70).rolling(7).sum() >= 4 & (df['SHORT'] <= 50).rolling(8).sum() >= 1
    df['双叉戟'] = (df['LONG'] >= 75).rolling(8).min() == 8 & (df['SHORT'] <= 50).rolling(6).sum() >= 2 & (df['SHORT'] <= 20).rolling(7).sum() >= 1
    df['红肥绿瘦'] = (df['C'] >= df['O']).rolling(15).sum() > 7 | (df['C'] > df['C'].shift(1)).rolling(11).sum() > 5
    
    # 大绿棒和缩量
    VDAY = df['V'].rolling(40).idxmax()
    df['不是大绿棒'] = df['C'].shift(VDAY.astype(int)) >= df['C'].shift(VDAY.astype(int) + 1) | df['C'].shift(VDAY.astype(int)) >= df['O'].shift(VDAY.astype(int))
    df['大绿棒'] = ~df['不是大绿棒']
    df['大绿棒离得远'] = VDAY >= 15 & df['大绿棒']
    df['缩量'] = (df['V'] < df['V'].rolling(20).max() * 0.416) | (df['V'] < df['V'].rolling(50).max() / 3)
    df['回踩缩量'] = (df['V'] < df['V'].rolling(20).max() * 0.45) | (df['V'] < df['V'].rolling(50).max() / 3)
    df['适当缩量'] = (df['V'] < df['V'].rolling(20).max() * 0.618) | (df['V'] < df['V'].rolling(50).max() / 3)
    df['超缩量'] = (df['V'] < df['V'].rolling(30).max() / 4) | (df['V'] < df['V'].rolling(50).max() / 6)
    
    # KDJ
    df['J'] = df['j']
    df['K'] = df['k']
    
    # RSI
    LC = df['C'].shift(1)
    TEMP1 = np.maximum(df['C'] - LC, 0)
    TEMP2 = np.abs(df['C'] - LC)
    df['RSI'] = TEMP1.rolling(3).mean() / TEMP2.rolling(3).mean() * 100
    
    # 振幅
    LOW N = df['L'].rolling(N).min()
    HIGHN = df['H'].rolling(N).max()
    近期振幅 = (HIGHN - LOWN) / LOWN * 100
    近期异动 = 近期振幅 >= 15 | (df['H'].rolling(12).max() - df['L'].rolling(14).min()) / df['L'].rolling(14).min() * 100 >= 11
    LOWM = df['L'].rolling(M).min()
    HIGHM = df['H'].rolling(M).max()
    远期振幅 = (HIGHM - LOWM) / LOWM * 100
    远期异动 = 远期振幅 >= 30
    超级异动 = 近期振幅 >= 60
    洗盘异动 = (df['单针下20'].rolling(10).sum() >= 2) | df['聚宝盆'] | df['双叉戟']
    
    # 趋势股
    df['做上涨趋势'] = df['趋势白线'] >= df['大哥黄线'] * 0.999 & (df['C'] >= df['大哥黄线'] | (df['C'] > df['大哥黄线'] * 0.975 & df['C'] > df['O']))
    df['强趋势股'] = (df['大哥黄线'] >= df['大哥黄线'].shift(1) * 0.999).rolling(13).min() == 1 & (df['趋势白线'] >= df['趋势白线'].shift(1)) & (df['趋势白线'] > df['大哥黄线']).rolling(20).min() == 1 & (df['趋势白线'] >= df['趋势白线'].shift(1)).rolling(11).min() == 1 & df['红肥绿瘦']
    df['超牛股'] = ((df['BBI'] >= df['BBI'].shift(1) * 0.999).rolling(20).min() == 1 | (df['BBI'] >= df['BBI'].shift(1)).rolling(25).sum() >= 23) & (近期振幅 >= 30 | 远期振幅 > 80) & (df['C'] > df['大哥黄线']).shift().idxmax().dt.days > 12
    
    # 回踩白线
    df['距离白线'] = abs(df['C'] - df['趋势白线']) / df['C'] * 100
    df['L距离白线'] = abs(df['L'] - df['趋势白线']) / df['趋势白线'] * 100
    df['距离BBI'] = abs(df['C'] - df['BBI']) / df['C'] * 100
    df['L距离BBI'] = abs(df['L'] - df['BBI']) / df['BBI'] * 100
    df['回踩白线'] = (df['C'] >= df['趋势白线'] & df['距离白线'] <= 2) | (df['C'] < df['趋势白线'] & df['距离白线'] < 0.8) | (df['C'] >= df['BBI'] & df['距离BBI'] < 2.5 & df['L距离BBI'] < 1 & df['距离白线'] <= 3 & df['当日涨跌幅'] < 1 & df['C'] > df['C'].shift(1))
    df['白线支撑'] = df['C'] >= df['趋势白线'] & df['距离白线'] < 1.5
    df['强势回踩不破'] = (df['L距离白线'] < 1 | df['L距离BBI'] < 0.5) & (df['C'] > df['趋势白线']) & (df['距离白线'] <= 3.5)
    
    # 回踩黄线
    df['距离黄线'] = abs(df['C'] - df['大哥黄线']) / df['大哥黄线'] * 100
    df['回踩黄线'] = (df['C'] >= df['大哥黄线'] & (df['距离黄线'] <= 1.5 | (df['距离黄线'] <= 2 & df['当日涨跌幅'] < 1))) | (df['C'] < df['大哥黄线'] & df['距离黄线'] <= 0.8)
    
    # 条件计算
    df['超卖缩量拐头B'] = df['做上涨趋势'] & (df['RSI'] - 15) >= df['RSI'].shift(1) & (df['RSI'].shift(1) < 20 | df['J'].shift(1) < 14) & df['当日振幅'] < (振幅区间 + 0.5) & (df['当日涨跌幅'] < (2.3 + Z1) | (df['上涨十字星'] & df['当日涨跌幅'] < 4)) & (df['不是大绿棒'] | df['大绿棒离得远']) & (df['近期异动'] | df['远期异动'] | df['洗盘异动']) & df['C'] >= df['大哥黄线']
    # 类似计算其他条件（超卖缩量B, 原始B1, 超卖超缩量B, 回踩白线B, 回踩超级B, 回踩黄线B）

    # 评分
    score = 0
    active_conditions = []
    for cond in ['超卖缩量拐头B', '超卖缩量B', '原始B1', '超卖超缩量B', '回踩白线B', '回踩超级B', '回踩黄线B']:
        if df[cond].iloc[-1]:
            score += weights.get(cond, 10)
            active_conditions.append(cond)
    
    # 个性化评论
    comment = f"股票 {name} 当前价 {current:.2f}，流通市值 {market_cap:.2f} 亿。"
    if market_cap < 最小流通值:
        comment += " 流通市值太小，风险高，不符合超级 B1 标准。"
    if active_conditions:
        comment += " 符合以下 B1 条件： " + "、".join(active_conditions) + "。"
        if '原始B1' in active_conditions:
            comment += " 原始 B1 信号强，首踩机会大，建议低吸。"
        if '超卖缩量B' in active_conditions:
            comment += " 超卖缩量，量价健康，情绪好，股性活跃。"
        # ... 为每个条件加独特描述
    else:
        comment += " 不符合任何 B1 条件，基本面和技术面一般，情绪低迷。"
    
    buy_advice = "可以买" if score > 60 else "不能买"
    
    return score, comment, buy_advice, b1_criteria

# 主界面
st.title("Z哥 AI 分析师 - 少妇 & B1 战法")

codes_input = st.text_input("输入股票代码（逗号分隔，如 600519,601218）")
if st.button("让 Z哥分析"):
    codes = [c.strip() for c in codes_input.split(',') if c.strip()]
    for symbol in codes:
        st.subheader(f"Z哥看 {symbol}")
        
        df, name, current, market_cap = fetch_stock_data(symbol)
        if df is None:
            st.error(f"无法获取 {symbol} 数据")
            continue
        
        df = calculate_indicators(df)
        
        score, comment, buy_advice, b1_criteria = analyze_stock(df, name, current)
        
        st.write("**Z哥打分：**", score)
        st.write("**Z哥评论：**", comment)
        st.write("**能不能买？**", buy_advice)
        
        # K 线图
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['OPEN'], high=df['HIGH'], low=df['LOW'], close=df['CLOSE'], increasing_line_color='red', decreasing_line_color='green')])
        fig.add_trace(go.Scatter(x=df.index, y=df['趋势白线'], name='趋势白线', line=dict(color='white')))
        fig.add_trace(go.Scatter(x=df.index, y=df['大哥黄线'], name='大哥黄线', line=dict(color='yellow')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BBI'], name='BBI', line=dict(color='blue')))
        fig.update_layout(title=f"{symbol} K线图", xaxis_rangeslider_visible=True, height=500)
        st.plotly_chart(fig)
        
        st.write("**B1 检查清单：**")
        for k, v in b1_criteria.items():
            st.write(f"- {k}：{'✅' if v else '❌'}")

st.sidebar.info("个性化评分和评论基于你的选股公式")
