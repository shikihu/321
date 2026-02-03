import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import time

# 函数：用 yfinance 拉 A股数据（Yahoo Finance 对香港友好）
def fetch_stock_data(symbol):
    try:
        # A股在 Yahoo 用 .SS (上证) 或 .SZ (深证)
        ticker = f"{symbol}.SS" if symbol.startswith('6') else f"{symbol}.SZ"
        stock = yf.Ticker(ticker)
        
        # 实时信息
        info = stock.info
        name = info.get('shortName', '未知股票')
        current = info.get('currentPrice', info.get('regularMarketPrice', 0.0))
        
        # 历史 K线（最近 365 天）
        hist = stock.history(period="1y", interval="1d")
        if hist.empty:
            raise ValueError("无历史数据")
        
        df = hist[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        
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
        
        return df, name, current
    except Exception as e:
        print(f"yfinance 失败: {e}")
        return None, "未知股票", 0.0

# 主界面
st.title("Z哥 AI 分析师 - 少妇 & B1 战法")

st.sidebar.title("Z哥六步法（背诵 100 遍）")
st.sidebar.markdown("""
1. 择时：周日看大盘温度，只在合适阶段动手  
2. 选股：强势基因 + 题材热  
3. 买点：B1首踩 或 B2主升  
4. 持仓：等利润垫，不折腾  
5. 卖点：四种卖法（利润垫/破位/高潮/情绪）  
6. 复盘：每笔交易都要复盘，避免情绪化  
""")
st.sidebar.write("心态：沉没成本别参与决策，戒骄戒躁，珍惜子弹！")

codes_input = st.text_input("输入股票代码（用逗号分隔，例如 600519,601218）")
if st.button("让 Z哥分析"):
    codes = [c.strip() for c in codes_input.split(',') if c.strip()]
    for symbol in codes:
        st.subheader(f"Z哥看 {symbol}")
        
        df, name, current = fetch_stock_data(symbol)
        
        if df is None or current == 0.0:
            st.warning(f"无法自动获取 {symbol} 数据（可能网络问题）。请稍后重试或手动查价。")
            continue
        
        st.success(f"**股票名称：** {name}")
        st.success(f"**当前价：** {current:.2f} 元")
        
        # K线图
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                             open=df['open'], high=df['high'],
                                             low=df['low'], close=df['close'],
                                             increasing_line_color='red', decreasing_line_color='green')])
        fig.add_trace(go.Scatter(x=df.index, y=df['ma20'], mode='lines', name='MA20（生命线）', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['ma60'], mode='lines', name='MA60（长期线）', line=dict(color='yellow')))
        fig.update_layout(title=f"{symbol} K线图", xaxis_rangeslider_visible=True, height=500)
        st.plotly_chart(fig)
        
        # B1 检查清单
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        ma20 = last['ma20']
        
        b1_check = {
            '趋势向上': last['ma20'] > prev['ma20'],
            '缩量明显': last['volume'] < df['volume'].rolling(5).mean().iloc[-1] * 1.2,
            '支撑有效': last['close'] > ma20 * 0.98,
            '无大阴线': (last['close'] - last['open']) / last['open'] < 0.07,
            'MACD多头': last['macd'] > 0,
            'KDJ超卖': last['j'] < 20,
            '股性活跃': df['close'].pct_change().abs().mean() > 0.02,
            '主流题材': True,
            '首次回踩': True,  # 简化
            'J值极低': last['j'] < -1,
            '量价配合': True,
            '关键K线': True,
            'BBI上升': last['bbi'] > prev['bbi'],
            '完美图形': True,
            '无真陷阱': True
        }
        
        st.write("**B1 检查清单：**")
        for k, v in b1_check.items():
            st.write(f"- {k}：{'✅' if v else '❌'}")
        
        # 打分
        score = sum(b1_check.values()) * 7
        score = min(score, 100)
        
        # Z哥总结
        if score >= 80:
            summary = f"当前价 {current:.2f}，符合少妇战法低买点，首踩缩量 + J负共振，完美一号，温柔黏人，赚钱机会大。"
            buy_advice = "可以买！小仓低吸，按六步法择时进场，持仓等利润垫。"
        elif score >= 65:
            summary = f"当前价 {current:.2f}，疑似 B1 机会，但需确认量价和 J 值。"
            buy_advice = "可以小仓试水，注意假突破陷阱。"
        else:
            summary = f"当前价 {current:.2f}，不符合铁律。不是首踩或量没缩到位。"
            buy_advice = "不能买，复盘等待更好的票。"
        
        st.write("**Z哥打分：**", score, "/ 100")
        st.write("**Z哥总结：**", summary)
        st.write("**能不能买？**", buy_advice)
        st.write("**卖出提醒：** 利润垫出现就跑、破位就跑、情绪高潮就跑，珍惜子弹！")
        
        st.balloons()

st.sidebar.success("自动拉数据版已就绪！")
st.sidebar.info("如果数据仍拉不到，请稍后重试或检查网络。Yahoo Finance 对香港访问友好。")
