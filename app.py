import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

# 函数：获取 A 股历史数据
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
        df['ma60'] = df['close'].rolling(60).mean()
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['dif'] = ema12 - ema26
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        df['macd'] = (df['dif'] - df['dea']) * 2
        # KDJ
        low_min = df['low'].rolling(9).min()
        high_max = df['high'].rolling(9).max()
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        df['k'] = rsv.ewm(span=3, adjust=False).mean()
        df['d'] = df['k'].ewm(span=3, adjust=False).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']
        # BBI
        df['bbi'] = (df['close'].rolling(3).mean() + df['close'].rolling(6).mean() + 
                     df['close'].rolling(12).mean() + df['close'].rolling(24).mean()) / 4
        return df
    except:
        return None

# 函数：Z哥战法分析 (模拟 AI Z哥)
def analyze_stock(symbol, df):
    if df is None:
        return {"error": "无法获取数据"}
    
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    ma20 = last['ma20']
    
    # 首次回踩判断 (B1核心)
    ma20_break_idx = -1
    for i in range(len(df)-2, -1, -1):
        if df['close'].iloc[i] > df['ma20'].iloc[i] and df['close'].iloc[i-1] <= df['ma20'].iloc[i-1]:
            ma20_break_idx = i
            break
    days_since_break = len(df) - 1 - ma20_break_idx if ma20_break_idx != -1 else 999
    touches_ma20 = sum(df['close'].iloc[ma20_break_idx+1:] < df['ma20'].iloc[ma20_break_idx+1:]) if ma20_break_idx != -1 else 0
    is_first_pullback = (3 <= days_since_break <= 15) and touches_ma20 <= 1
    
    # 量缩 & 量价关系
    peak_vol = df['volume'].iloc[-60:-30].max() if len(df) > 60 else df['volume'].max()
    vol_ratio = last['volume'] / peak_vol if peak_vol > 0 else 1
    volume_shrink = vol_ratio < 0.35
    qty_price_match = (last['close'] > prev['close'] and last['volume'] > prev['volume']) or (volume_shrink and is_first_pullback)  # 量价齐升 or 缩量
    
    # KDJ & 关键K
    j_extreme = last['j'] < -1
    kdj_oversold = last['j'] < 20
    key_k = (last['close'] - last['open']) / last['open'] > 0.03 and last['volume'] > df['volume'].rolling(5).mean().iloc[-1]  # 阳线放量确认
    
    # B2 买点 (主升浪)
    is_b2 = df['close'].iloc[-1] == df['high'].rolling(60).max().iloc[-1] and last['macd'] > 0 and qty_price_match
    
    # 陷阱过滤 (击穿对手盘)
    fake_break = any(df['close'].iloc[-5:] < df['ma20'].iloc[-5:]) and last['close'] > ma20  # 破位但收回 = 假陷阱
    real_trap = sum(df['close'].iloc[-5:] < df['ma20'].iloc[-5:]) > 2 and not volume_shrink  # 多次破 + 不缩 = 真陷阱
    
    # 完美图形 (选美)
    perfect_pattern = is_first_pullback and volume_shrink and key_k and last['bbi'] > prev['bbi'] and (df['high'].iloc[-60:].max() / df['low'].iloc[-60:].min() - 1) <= 1.0
    
    # 打分 & 符合度
    b1_criteria = {
        'trendUp': last['ma20'] > prev['ma20'],
        'volumeShrink': volume_shrink,
        'supportValid': last['close'] > ma20 * 0.98,
        'noBigGreen': (last['close'] - last['open']) / last['open'] < 0.07,
        'macdBullish': last['macd'] > 0,
        'kdjOversold': kdj_oversold,
        'isActive': df['close'].pct_change().abs().mean() > 0.02,
        'isMainstream': True,  # 假设，用户可手动
        'isFirstPullback': is_first_pullback,
        'jExtreme': j_extreme,
        'qtyPriceMatch': qty_price_match,
        'keyK': key_k,
        'bbiUp': last['bbi'] > prev['bbi'],
        'perfectPattern': perfect_pattern,
        'noRealTrap': not real_trap
    }
    score = sum(b1_criteria.values()) * 7  # 约15项，满分105，规范到100
    score = min(score, 100)
    
    # Z哥式总结 & 建议
    if real_trap:
        summary = "警惕主力陷阱！多次破位不缩量，击穿对手盘真出货，别碰。"
        buy_advice = "不能买，等待下一个机会。"
    elif score >= 80 and (is_first_pullback or is_b2):
        summary = "完美一号！符合少妇战法低买点：首踩/B2 + 缩量 + J负 + 关键K确认，温柔黏人，赚钱机会大。"
        buy_advice = "能买，低吸！按六步法：择时后选股买入，持等利润垫。"
    elif score >= 60:
        summary = "疑似好票，但量价/J值未完全到位，观察放量确认或 B2 主升。"
        buy_advice = "可买，但小仓试水。注意假突破陷阱。"
    else:
        summary = "不符合 Z哥铁律，非首踩或量未缩，告别无效盯盘，别折腾子弹。"
        buy_advice = "不能买，复盘等待更好机会。"
    
    # 卖点 & 心态
    sell_tips = "卖出参考：1. 利润垫出现；2. 破 MA20/60；3. 情绪高潮/放量滞涨；4. 四种卖法。心态：沉没成本不决策，戒骄戒躁，珍惜子弹。"
    
    metrics = [
        {"label": "当前价", "value": round(last['close'], 2)},
        {"label": "MA20", "value": round(ma20, 2)},
        {"label": "MACD", "value": round(last['macd'], 2)},
        {"label": "J 值", "value": round(last['j'], 2)},
        {"label": "BBI", "value": round(last['bbi'], 2)},
        {"label": "量比", "value": round(vol_ratio, 2)}
    ]
    
    return {
        "symbol": symbol,
        "currentPrice": last['close'],
        "changePercent": (last['close'] - prev['close']) / prev['close'] * 100,
        "score": score,
        "summary": summary,
        "buyAdvice": buy_advice,
        "sellTips": sell_tips,
        "bullishFactors": [k for k, v in b1_criteria.items() if v],
        "bearishFactors": [k for k, v in b1_criteria.items() if not v],
        "metrics": metrics,
        "b1Criteria": b1_criteria
    }

# 主界面
st.title("Z哥 AI Analyst - 少妇 & B1 战法")

st.sidebar.title("Z哥六步法背诵")
st.sidebar.write("""
1. 择时：周日看大盘温度。
2. 选股：强势基因 + 题材热。
3. 买点：B1首踩 or B2主升。
4. 持仓：等利润垫，不折腾。
5. 卖点：四种卖法 (利润/破位/高潮/情绪)。
6. 复盘：每笔记录，避免情绪。
""")
st.sidebar.write("心态：沉没成本不决策，珍惜子弹！")

codes_input = st.text_input("输入股票代码 (逗号分隔，如 600519,000001)")
if st.button("Z哥分析"):
    if codes_input:
        codes = [c.strip() for c in codes_input.split(',')]
        for symbol in codes:
            st.subheader(f"Z哥看 {symbol}")
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
            fig.add_trace(go.Scatter(x=df.index, y=df['ma60'], mode='lines', name='MA60', line=dict(color='yellow')))
            fig.update_layout(title=f"{symbol} K线图 (盯关键K)", xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
            
            # 分析
            analysis = analyze_stock(symbol, df)
            if "error" in analysis:
                st.error(analysis["error"])
            else:
                st.write("**Z哥分数：**", analysis['score'])
                st.write("**Z哥总结：**", analysis['summary'])
                st.write("**能不能买？**", analysis['buyAdvice'])
                st.write("**看多因素：**", ", ".join(analysis['bullishFactors']))
                st.write("**看空因素：**", ", ".join(analysis['bearishFactors']))
                st.write("**卖出提醒：**", analysis['sellTips'])
                st.table(pd.DataFrame(analysis['metrics']))
                st.write("**B1/B2 Checklist：**")
                for k, v in analysis['b1Criteria'].items():
                    st.write(f"- {k}: {'✅' if v else '❌'}")

st.sidebar.title("使用说明")
st.sidebar.write("- 输入代码，Z哥帮你判是否符合少妇/B1。")
st.sidebar.write("- 基于视频提炼，本地计算，无限用。")
st.sidebar.write("- 炒股风险自负，仅参考。")
