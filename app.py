import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

# 函数：获取 A 股历史数据（使用东方财富接口，更稳定）
def fetch_stock_history(symbol):
    # 东方财富 secid：沪市用 1. 前缀，深市用 0. 前缀，北京用 2. 前缀
    if symbol.startswith('6'):
        secid = f"1.{symbol}"
    elif symbol.startswith('0') or symbol.startswith('3'):
        secid = f"0.{symbol}"
    elif symbol.startswith('4') or symbol.startswith('8'):
        secid = f"2.{symbol}"
    else:
        secid = f"1.{symbol}"  # 默认沪市
    
    url = f"http://push2.eastmoney.com/api/qt/stock/kline?secid={secid}&fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=1&lmt=365"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return None
        
        data = response.json()
        if 'data' not in data or 'klines' not in data['data']:
            return None
        
        klines = data['data']['klines']
        if not klines:
            return None
        
        # 格式：date,open,close,high,low,volume,成交额,振幅,涨跌幅,涨跌额,换手率
        rows = [line.split(',') for line in klines]
        df = pd.DataFrame(rows, columns=['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amp', 'pct_chg', 'chg', 'turnover'])
        df = df[['date', 'open', 'close', 'high', 'low', 'volume']]
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        for col in ['open', 'close', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 计算技术指标
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
    except Exception as e:
        print(f"获取 {symbol} 数据失败: {str(e)}")
        return None

# 函数：Z哥战法分析（模拟 AI Z哥）
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
    qty_price_match = (last['close'] > prev['close'] and last['volume'] > prev['volume']) or (volume_shrink and is_first_pullback)
    
    # KDJ & 关键K
    j_extreme = last['j'] < -1
    kdj_oversold = last['j'] < 20
    key_k = (last['close'] - last['open']) / last['open'] > 0.03 and last['volume'] > df['volume'].rolling(5).mean().iloc[-1]
    
    # B2 买点
    is_b2 = df['close'].iloc[-1] == df['high'].rolling(60).max().iloc[-1] and last['macd'] > 0 and qty_price_match
    
    # 陷阱过滤
    fake_break = any(df['close'].iloc[-5:] < df['ma20'].iloc[-5:]) and last['close'] > ma20
    real_trap = sum(df['close'].iloc[-5:] < df['ma20'].iloc[-5:]) > 2 and not volume_shrink
    
    # 完美图形
    perfect_pattern = is_first_pullback and volume_shrink and key_k and last['bbi'] > prev['bbi'] and (df['high'].iloc[-60:].max() / df['low'].iloc[-60:].min() - 1) <= 1.0
    
    # 打分 & 符合度
    b1_criteria = {
        '趋势向上': last['ma20'] > prev['ma20'],
        '缩量明显': volume_shrink,
        '支撑有效': last['close'] > ma20 * 0.98,
        '无大阴线': (last['close'] - last['open']) / last['open'] < 0.07,
        'MACD多头': last['macd'] > 0,
        'KDJ超卖': kdj_oversold,
        '股性活跃': df['close'].pct_change().abs().mean() > 0.02,
        '主流题材': True,  # 可手动输入
        '首次回踩': is_first_pullback,
        'J值极低': j_extreme,
        '量价配合': qty_price_match,
        '关键K线': key_k,
        'BBI上升': last['bbi'] > prev['bbi'],
        '完美图形': perfect_pattern,
        '无真陷阱': not real_trap
    }
    score = sum(b1_criteria.values()) * 7
    score = min(score, 100)
    
    # Z哥式中文总结 & 建议
    if real_trap:
        summary = "小心主力陷阱！多次破位还不缩量，击穿对手盘真出货，别碰！"
        buy_advice = "不能买，等下一个机会。"
    elif score >= 80 and (is_first_pullback or is_b2):
        summary = "完美一号！符合少妇战法低买点：首踩/B2 + 缩量 + J负 + 关键K确认，温柔黏人，赚钱机会很大。"
        buy_advice = "可以买，低吸！按六步法：择时后选股买入，持仓等利润垫。"
    elif score >= 60:
        summary = "疑似好票，但量价或 J 值还没完全到位，观察放量确认或等 B2 主升。"
        buy_advice = "可以小仓试水，但注意假突破陷阱。"
    else:
        summary = "不符合 Z哥铁律，不是首踩或者量没缩到位，告别无效盯盘，别折腾子弹。"
        buy_advice = "不能买，复盘等待更好的机会。"
    
    # 卖点 & 心态
    sell_tips = "卖出参考：1. 利润垫出现；2. 破 MA20/60；3. 情绪高潮或放量滞涨；4. 四种卖法。心态：沉没成本别决策，戒骄戒躁，珍惜子弹！"
    
    metrics = [
        {"指标": "当前价", "数值": round(last['close'], 2)},
        {"指标": "MA20", "数值": round(ma20, 2)},
        {"指标": "MACD", "数值": round(last['macd'], 2)},
        {"指标": "J 值", "数值": round(last['j'], 2)},
        {"指标": "BBI", "数值": round(last['bbi'], 2)},
        {"指标": "量比", "数值": round(vol_ratio, 2)}
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
st.title("Z哥 AI 分析师 - 少妇 & B1 战法")

st.sidebar.title("Z哥六步法（背诵 100 遍）")
st.sidebar.write("""
1. 择时：周日看大盘温度，只在合适阶段动手  
2. 选股：强势基因 + 题材热  
3. 买点：B1首踩 或 B2主升  
4. 持仓：等利润垫，不折腾  
5. 卖点：四种卖法（利润垫/破位/高潮/情绪）  
6. 复盘：每笔交易都要复盘，避免情绪化  
""")
st.sidebar.write("心态：沉没成本别参与决策，戒骄戒躁，珍惜子弹！")

codes_input = st.text_input("输入股票代码（用逗号分隔，例如 600519,000001,601218）")
if st.button("让 Z哥分析"):
    if codes_input:
        codes = [c.strip() for c in codes_input.split(',')]
        for symbol in codes:
            st.subheader(f"Z哥看 {symbol}")
            df = fetch_stock_history(symbol)
            if df is None:
                st.error(f"无法获取 {symbol} 的数据，请稍后重试或检查网络。")
                continue
            
            # K 线图
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                 open=df['open'], high=df['high'],
                                                 low=df['low'], close=df['close'],
                                                 increasing_line_color='red', decreasing_line_color='green')])
            fig.add_trace(go.Scatter(x=df.index, y=df['ma20'], mode='lines', name='MA20（生命线）', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df.index, y=df['ma60'], mode='lines', name='MA60（长期线）', line=dict(color='yellow')))
            fig.update_layout(title=f"{symbol} K线图（重点盯关键K）", xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
            
            # 分析结果
            analysis = analyze_stock(symbol, df)
            if "error" in analysis:
                st.error(analysis["error"])
            else:
                st.write("**Z哥打分：**", analysis['score'])
                st.write("**Z哥总结：**", analysis['summary'])
                st.write("**能不能买？**", analysis['buyAdvice'])
                st.write("**看多因素：**", "、".join(analysis['bullishFactors']))
                st.write("**看空因素：**", "、".join(analysis['bearishFactors']))
                st.write("**卖出提醒：**", analysis['sellTips'])
                st.table(pd.DataFrame(analysis['metrics']))
                st.write("**B1/B2 检查清单：**")
                for k, v in analysis['b1Criteria'].items():
                    st.write(f"- {k}：{'✅' if v else '❌'}")

st.sidebar.title("使用小贴士")
st.sidebar.write("- 输入股票代码，Z哥帮你判是否符合少妇/B1战法")
st.sidebar.write("- 数据来自东方财富，实时获取，本地计算，无需密钥，无限使用")
st.sidebar.write("- 炒股有风险，仅供参考，不构成投资建议")
