import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

# 函数：获取 A 股历史数据（使用腾讯财经接口）
def fetch_stock_history(symbol):
    if not symbol.isdigit() or len(symbol) != 6:
        return None
        
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        raw = response.json()
        
        # 安全提取数据：处理 key 为空字符串 "" 的情况（常见于腾讯接口）
        data_section = raw.get('data', {})
        stock_key = f"{prefix}{symbol}"
        stock_data = []

        # 尝试正常 key
        if stock_key in data_section:
            stock_data = data_section[stock_key].get('qfqday', []) or data_section[stock_key].get('day', [])
        # 尝试空字符串 key（如 {"data": {"": {...}} }）
        elif "" in data_section:
            inner = data_section[""]
            if isinstance(inner, dict):
                stock_data = inner.get('qfqday', []) or inner.get('day', [])
        
        if not stock_data:
            st.warning(f"⚠️ {symbol}：未获取到K线数据，请检查代码或稍后再试。")
            return None

        # 创建 DataFrame 并立即复制，避免 SettingWithCopyWarning
        df = pd.DataFrame(stock_data, columns=['date', 'open', 'close', 'high', 'low', 'volume']).copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 转换数值类型，错误转为 NaN 后删除
        for col in ['open', 'close', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        if df.empty or len(df) < 20:
            return None

        # 计算技术指标
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['dif'] = ema12 - ema26
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        df['macd'] = (df['dif'] - df['dea']) * 2

        # KDJ（加防除零）
        low_min = df['low'].rolling(9).min()
        high_max = df['high'].rolling(9).max()
        denominator = (high_max - low_min).replace(0, 1)  # 防止除零
        rsv = (df['close'] - low_min) / denominator * 100
        df['k'] = rsv.ewm(span=3, adjust=False).mean()
        df['d'] = df['k'].ewm(span=3, adjust=False).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']

        # BBI
        ma3 = df['close'].rolling(3).mean()
        ma6 = df['close'].rolling(6).mean()
        ma12 = df['close'].rolling(12).mean()
        ma24 = df['close'].rolling(24).mean()
        df['bbi'] = (ma3 + ma6 + ma12 + ma24) / 4

        return df

    except Exception as e:
        st.error(f"❌ 获取 {symbol} 数据失败: {str(e)}")
        return None


# 函数：Z哥战法分析（模拟 AI Z哥）
def analyze_stock(symbol, df):
    if df is None or len(df) < 2:
        return {"error": "数据不足，无法分析"}
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    ma20 = last['ma20']
    
    # 首次回踩判断 (B1核心) —— 修复越界问题
    ma20_break_idx = -1
    for i in range(len(df)-2, 0, -1):  # 从倒数第2天到第1天（避免 i=0 时 i-1=-1）
        if df['close'].iloc[i] > df['ma20'].iloc[i] and df['close'].iloc[i-1] <= df['ma20'].iloc[i-1]:
            ma20_break_idx = i
            break
    days_since_break = len(df) - 1 - ma20_break_idx if ma20_break_idx != -1 else 999
    touches_ma20 = sum(
        df['close'].iloc[ma20_break_idx+1:] < df['ma20'].iloc[ma20_break_idx+1:]
    ) if ma20_break_idx != -1 else 0
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
    is_b2 = (last['close'] == df['high'].rolling(60).max().iloc[-1]) and last['macd'] > 0 and qty_price_match
    
    # 陷阱过滤
    fake_break = any(df['close'].iloc[-5:] < df['ma20'].iloc[-5:]) and last['close'] > ma20
    real_trap = sum(df['close'].iloc[-5:] < df['ma20'].iloc[-5:]) > 2 and not volume_shrink
    
    # 完美图形（放宽涨幅限制）
    price_range = (df['high'].iloc[-60:].max() / df['low'].iloc[-60:].min() - 1)
    perfect_pattern = is_first_pullback and volume_shrink and key_k and last['bbi'] > prev['bbi'] and price_range <= 1.5
    
    # 打分 & 符合度
    b1_criteria = {
        '趋势向上': last['ma20'] > prev['ma20'],
        '缩量明显': volume_shrink,
        '支撑有效': last['close'] > ma20 * 0.98,
        '无大阴线': (last['close'] - last['open']) / last['open'] < 0.07,
        'MACD多头': last['macd'] > 0,
        'KDJ超卖': kdj_oversold,
        '股性活跃': df['close'].pct_change().abs().mean() > 0.02,
        '主流题材': True,  # 可手动扩展
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
st.set_page_config(page_title="Z哥 AI 分析师", layout="wide")
st.title("🎯 Z哥 AI 分析师 - 少妇 & B1 战法")

st.sidebar.title("📌 Z哥六步法（背诵 100 遍）")
st.sidebar.write("""
1. 择时：周日看大盘温度，只在合适阶段动手  
2. 选股：强势基因 + 题材热  
3. 买点：B1首踩 或 B2主升  
4. 持仓：等利润垫，不折腾  
5. 卖点：四种卖法（利润垫/破位/高潮/情绪）  
6. 复盘：每笔交易都要复盘，避免情绪化  
""")
st.sidebar.markdown("**心态**：沉没成本别参与决策，戒骄戒躁，珍惜子弹！")

codes_input = st.text_input("🔍 输入股票代码（用逗号分隔，例如 600519,000858）", placeholder="600519,000001")
if st.button("🚀 让 Z哥分析"):
    if codes_input:
        codes = [c.strip() for c in codes_input.split(',') if c.strip()]
        if not codes:
            st.warning("请输入有效的股票代码")
        else:
            for symbol in codes:
                st.subheader(f"📊 Z哥看 {symbol}")
                df = fetch_stock_history(symbol)
                if df is None:
                    st.error(f"无法获取 {symbol} 的数据，请检查代码是否正确（6位数字）。")
                    continue
                
                # 绘制K线图
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index,
                    open=df['open'], high=df['high'],
                    low=df['low'], close=df['close'],
                    increasing_line_color='red', decreasing_line_color='green'
                )])
                fig.add_trace(go.Scatter(x=df.index, y=df['ma20'], mode='lines', name='MA20（生命线）', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=df.index, y=df['ma60'], mode='lines', name='MA60（长期线）', line=dict(color='orange')))
                fig.update_layout(title=f"{symbol} K线图（重点盯关键K）", xaxis_rangeslider_visible=True, height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # 分析结果
                analysis = analyze_stock(symbol, df)
                if "error" in analysis:
                    st.error(analysis["error"])
                else:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Z哥打分", f"{analysis['score']}/100")
                        st.metric("当前价", f"¥{analysis['currentPrice']:.2f}")
                    with col2:
                        st.write(f"**💡 Z哥总结**：{analysis['summary']}")
                        st.write(f"**✅ 能不能买？** {analysis['buyAdvice']}")
                    
                    st.write("**📈 看多因素**：", "、".join(analysis['bullishFactors']) if analysis['bullishFactors'] else "无")
                    st.write("**📉 看空因素**：", "、".join(analysis['bearishFactors']) if analysis['bearishFactors'] else "无")
                    st.write("**📤 卖出提醒**：", analysis['sellTips'])
                    
                    st.table(pd.DataFrame(analysis['metrics']))
                    
                    st.write("**✅ B1/B2 检查清单**：")
                    for k, v in analysis['b1Criteria'].items():
                        st.write(f"- {k}：{'✅' if v else '❌'}")

st.sidebar.title("ℹ️ 使用小贴士")
st.sidebar.write("- 支持沪市（600/601/603/688）和深市（000/002/300）")
st.sidebar.write("- 数据来自腾讯财经，免费但偶有延迟")
st.sidebar.write("- ⚠️ 本工具仅用于学习交流，不构成投资建议")
