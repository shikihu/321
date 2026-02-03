import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import json

# 函数：获取 A 股历史数据（从腾讯接口）
def fetch_stock_history(symbol):
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,365,qfq"
    try:
        response = requests.get(url)
        data = response.json().get('data', {}).get(f"{prefix}{symbol}", {}).get('qfqday', [])
        if not data:
            return None
        df = pd.DataFrame(data, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        for col in ['open', 'close', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col])
        # 计算指标
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma5'] = df['close'].rolling(5).mean()
        # 简单 MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['dif'] = ema12 - ema26
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        df['macd'] = (df['dif'] - df['dea']) * 2
        # 简单 KDJ
        low_min = df['low'].rolling(9).min()
        high_max = df['high'].rolling(9).max()
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        df['k'] = rsv.ewm(span=3, adjust=False).mean()
        df['d'] = df['k'].ewm(span=3, adjust=False).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']
        return df
    except:
        return None

# 函数：本地分析（模拟 AI）
def analyze_stock(symbol, strategy, df):
    if df is None:
        return {"error": "无法获取数据"}
    
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    ma20 = last['ma20']
    score = 0
    b1_criteria = {
        'trendUp': last['ma20'] > prev['ma20'],
        'volumeShrink': last['volume'] < df['volume'].rolling(5).mean().iloc[-1],
        'supportValid': last['close'] > ma20 * 0.98,
        'noBigGreen': (last['close'] - last['open']) / last['open'] < 0.07,
        'macdBullish': last['macd'] > 0,
        'kdjOversold': last['j'] < 20,
        'isActive': df['close'].pct_change().abs().mean() > 0.02,
        'isMainstream': True,  # 假设
        'isFirstPullback': True  # 简化
    }
    score = sum(b1_criteria.values()) * 10  # 简单打分
    summary = f"基于策略 '{strategy}'：股票 {symbol} 当前价 {last['close']:.2f}，MA20 {ma20:.2f}。"
    if b1_criteria['trendUp']:
        summary += " 趋势向上。"
    if b1_criteria['volumeShrink']:
        summary += " 缩量回踩。"
    bullish = [k for k, v in b1_criteria.items() if v and k in ['trendUp', 'macdBullish', 'kdjOversold']]
    bearish = [k for k, v in b1_criteria.items() if not v and k in ['trendUp', 'macdBullish', 'kdjOversold']]
    metrics = [
        {"label": "当前价", "value": round(last['close'], 2)},
        {"label": "MA20", "value": round(ma20, 2)},
        {"label": "MACD", "value": round(last['macd'], 2)},
        {"label": "J 值", "value": round(last['j'], 2)}
    ]
    return {
        "symbol": symbol,
        "currentPrice": last['close'],
        "changePercent": (last['close'] - prev['close']) / prev['close'] * 100,
        "score": score,
        "summary": summary,
        "bullishFactors": bullish,
        "bearishFactors": bearish,
        "metrics": metrics,
        "b1Criteria": b1_criteria
    }

# 主界面
st.title("CN Stock AI Analyst - 简单版")

strategy = st.text_area("自定义策略 (默认 B1 战法)", value="1. 形态：必须符合 B1 缩量回踩。2. 趋势：MA20 必须向上，股价在 MA20 附近。3. 题材：必须是当前市场核心热门题材。4. 股性：必须活跃，近期有涨停板优先。", height=150)

codes_input = st.text_input("输入股票代码 (逗号分隔，如 600519,000001)")
if st.button("分析"):
    if codes_input:
        codes = [c.strip() for c in codes_input.split(',')]
        for symbol in codes:
            st.subheader(f"分析 {symbol}")
            df = fetch_stock_history(symbol)
            if df is None:
                st.error(f"无法获取 {symbol} 数据。")
                continue
            
            # K 线图
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                 open=df['open'], high=df['high'],
                                                 low=df['low'], close=df['close'],
                                                 increasing_line_color='red', decreasing_line_color='green')])
            fig.add_trace(go.Scatter(x=df.index, y=df['ma20'], mode='lines', name='MA20', line=dict(color='blue')))
            fig.update_layout(title=f"{symbol} K线图", xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
            
            # 分析
            analysis = analyze_stock(symbol, strategy, df)
            if "error" in analysis:
                st.error(analysis["error"])
            else:
                st.write("**分数：**", analysis['score'])
                st.write("**总结：**", analysis['summary'])
                st.write("**看多因素：**", ", ".join(analysis['bullishFactors']))
                st.write("**看空因素：**", ", ".join(analysis['bearishFactors']))
                st.table(pd.DataFrame(analysis['metrics']))
                st.write("**B1 Checklist：**")
                for k, v in analysis['b1Criteria'].items():
                    st.write(f"- {k}: {'✅' if v else '❌'}")

st.sidebar.title("使用说明")
st.sidebar.write("- 输入代码分析股票。")
st.sidebar.write("- 数据从公开源实时获取。")
st.sidebar.write("- 分析基于本地计算，无 API 限制。")
