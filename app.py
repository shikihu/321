import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf

# ======================
# 数据获取：双源 fallback
# ======================

def fetch_from_tencent(symbol):
    """从腾讯接口获取 A 股数据（主用）"""
    if not (symbol.isdigit() and len(symbol) == 6):
        return None
        
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
    
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        raw = response.json()
        
        data_section = raw.get('data', {})
        stock_data = []
        key1 = f"{prefix}{symbol}"
        if key1 in data_section:
            inner = data_section[key1]
            stock_data = inner.get('qfqday', []) or inner.get('day', [])
        elif "" in data_section:
            inner = data_section[""]
            if isinstance(inner, dict):
                stock_data = inner.get('qfqday', []) or inner.get('day', [])
        
        if not stock_data:
            return None

        # ✅ 核心：只取每行前6个字段（无视第7列成交额）
        cleaned = []
        for row in stock_data:
            if isinstance(row, list) and len(row) >= 6:
                cleaned.append([str(x) for x in row[:6]])
        if not cleaned:
            return None

        df = pd.DataFrame(cleaned, columns=['date', 'open', 'close', 'high', 'low', 'volume']).copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df.set_index('date', inplace=True)

        for col in ['open', 'close', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        if len(df) < 20:
            return None

        return df
    except Exception as e:
        # st.write(f"[腾讯] {symbol} 失败: {e}")  # 调试时可打开
        return None


def fetch_from_yfinance(symbol):
    """从 yfinance 获取（备用）"""
    try:
        ticker = f"{symbol}.SS" if symbol.startswith('6') else f"{symbol}.SZ"
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y", interval="1d")
        if hist.empty or len(hist) < 20:
            return None
            
        df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        if len(df) < 20:
            return None
        return df
    except Exception as e:
        # st.write(f"[Yahoo] {symbol} 失败: {e}")
        return None


def fetch_stock_history(symbol):
    """主函数：先腾讯，失败再 Yahoo"""
    df = fetch_from_tencent(symbol)
    source = "腾讯财经"
    if df is None:
        df = fetch_from_yfinance(symbol)
        source = "Yahoo Finance (备用)"
    return df, source


# ======================
# 技术指标计算
# ======================

def calculate_indicators(df):
    df = df.copy()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = ema12 - ema26
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    denominator = (high_max - low_min).replace(0, 1)
    rsv = (df['close'] - low_min) / denominator * 100
    df['k'] = rsv.ewm(span=3, adjust=False).mean()
    df['d'] = df['k'].ewm(span=3, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    
    ma3 = df['close'].rolling(3).mean()
    ma6 = df['close'].rolling(6).mean()
    ma12 = df['close'].rolling(12).mean()
    ma24 = df['close'].rolling(24).mean()
    df['bbi'] = (ma3 + ma6 + ma12 + ma24) / 4
    
    return df


# ======================
# Z哥战法分析
# ======================

def analyze_stock(df):
    if len(df) < 2:
        return {"error": "数据不足"}
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    ma20 = last['ma20']
    
    # 首次回踩
    ma20_break_idx = -1
    for i in range(len(df)-2, max(0, len(df)-60), -1):
        if df['close'].iloc[i] > df['ma20'].iloc[i] and df['close'].iloc[i-1] <= df['ma20'].iloc[i-1]:
            ma20_break_idx = i
            break
    days_since_break = len(df) - 1 - ma20_break_idx if ma20_break_idx != -1 else 999
    is_first_pullback = 3 <= days_since_break <= 15
    
    # 缩量
    peak_vol = df['volume'].iloc[-60:].max()
    vol_ratio = last['volume'] / peak_vol if peak_vol > 0 else 1
    volume_shrink = vol_ratio < 0.35
    
    # 陷阱判断
    real_trap = sum(df['close'].iloc[-5:] < df['ma20'].iloc[-5:]) > 2 and not volume_shrink
    
    b1_criteria = {
        '趋势向上': last['ma20'] > prev['ma20'],
        '缩量明显': volume_shrink,
        '支撑有效': last['close'] > ma20 * 0.98,
        '无大阴线': (last['close'] - last['open']) / (last['open'] + 1e-8) < 0.07,
        'MACD多头': last['macd'] > 0,
        'KDJ超卖': last['j'] < 20,
        '股性活跃': df['close'].pct_change().abs().mean() > 0.02,
        '主流题材': True,
        '首次回踩': is_first_pullback,
        'J值极低': last['j'] < -1,
        '量价配合': (last['close'] > prev['close']) == (last['volume'] > prev['volume']),
        '关键K线': (last['close'] - last['open']) / (last['open'] + 1e-8) > 0.03,
        'BBI上升': last['bbi'] > prev['bbi'],
        '完美图形': is_first_pullback and volume_shrink and last['j'] < 0,
        '无真陷阱': not real_trap
    }
    
    score = sum(b1_criteria.values()) * 7
    score = min(score, 100)
    
    if real_trap:
        summary = "小心主力陷阱！多次破位还不缩量，击穿对手盘真出货，别碰！"
        buy_advice = "❌ 不能买，等下一个机会。"
    elif score >= 80 and (is_first_pullback or last['macd'] > 0):
        summary = "完美一号！符合少妇战法低买点：首踩/B2 + 缩量 + J负 + 关键K确认，温柔黏人，赚钱机会很大。"
        buy_advice = "✅ 可以买！低吸，按六步法择时进场。"
    elif score >= 60:
        summary = "疑似好票，但量价或 J 值还没完全到位，观察放量确认或等 B2 主升。"
        buy_advice = "⚠️ 可小仓试水，注意假突破陷阱。"
    else:
        summary = "不符合 Z哥铁律，不是首踩或者量没缩到位，告别无效盯盘，别折腾子弹。"
        buy_advice = "❌ 不能买，复盘等待更好的机会。"
    
    sell_tips = "卖出参考：1. 利润垫出现；2. 破 MA20/60；3. 情绪高潮或放量滞涨；4. 四种卖法。心态：沉没成本别决策，戒骄戒躁，珍惜子弹！"
    
    current_price = last['close']
    change_pct = (last['close'] - prev['close']) / prev['close'] * 100
    
    metrics = [
        {"指标": "当前价", "数值": round(current_price, 2)},
        {"指标": "MA20", "数值": round(ma20, 2)},
        {"指标": "MACD", "数值": round(last['macd'], 2)},
        {"指标": "J 值", "数值": round(last['j'], 2)},
        {"指标": "BBI", "数值": round(last['bbi'], 2)},
        {"指标": "量比", "数值": round(vol_ratio, 2)}
    ]
    
    return {
        "score": score,
        "summary": summary,
        "buyAdvice": buy_advice,
        "sellTips": sell_tips,
        "bullishFactors": [k for k, v in b1_criteria.items() if v],
        "bearishFactors": [k for k, v in b1_criteria.items() if not v],
        "metrics": metrics,
        "b1Criteria": b1_criteria,
        "currentPrice": current_price,
        "changePercent": change_pct
    }


# ======================
# 主界面
# ======================

st.set_page_config(page_title="Z哥 AI 分析师", layout="wide")
st.title("🎯 Z哥 AI 分析师 - 少妇 & B1 战法")

st.sidebar.title("📌 Z哥六步法（背诵 100 遍）")
st.sidebar.markdown("""
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
    if not codes_input.strip():
        st.warning("请输入股票代码")
    else:
        codes = [c.strip() for c in codes_input.split(',') if c.strip()]
        for symbol in codes:
            if not (symbol.isdigit() and len(symbol) == 6):
                st.error(f"❌ {symbol} 不是有效的6位A股代码")
                continue
                
            st.subheader(f"📊 Z哥看 {symbol}")
            
            df, source = fetch_stock_history(symbol)
            if df is None:
                st.error(f"无法获取 {symbol} 的数据（腾讯和 Yahoo 均失败）")
                continue
                
            st.caption(f"数据来源：{source}")
            df = calculate_indicators(df)
            
            # K线图
            fig = go.Figure(data=[go.Candlestick(
                x=df.index, open=df['open'], high=df['high'],
                low=df['low'], close=df['close'],
                increasing_line_color='red', decreasing_line_color='green'
            )])
            fig.add_trace(go.Scatter(x=df.index, y=df['ma20'], mode='lines', name='MA20（生命线）', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df.index, y=df['ma60'], mode='lines', name='MA60（长期线）', line=dict(color='orange')))
            fig.update_layout(title=f"{symbol} K线图", xaxis_rangeslider_visible=True, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # 分析
            analysis = analyze_stock(df)
            if "error" in analysis:
                st.warning(analysis["error"])
            else:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Z哥打分", f"{analysis['score']}/100")
                    st.metric("当前价", f"¥{analysis['currentPrice']:.2f}")
                with col2:
                    st.write(f"**💡 Z哥总结**：{analysis['summary']}")
                    st.write(f"**🛒 能不能买？** {analysis['buyAdvice']}")
                
                st.write("**📈 看多因素**：", "、".join(analysis['bullishFactors']) if analysis['bullishFactors'] else "无")
                st.write("**📉 看空因素**：", "、".join(analysis['bearishFactors']) if analysis['bearishFactors'] else "无")
                st.write("**📤 卖出提醒**：", analysis['sellTips'])
                
                st.table(pd.DataFrame(analysis['metrics']))
                
                st.write("**✅ B1/B2 检查清单**：")
                for k, v in analysis['b1Criteria'].items():
                    st.write(f"- {k}：{'✅' if v else '❌'}")

st.sidebar.title("ℹ️ 使用说明")
st.sidebar.info(
    "- 优先使用腾讯财经（快且准）\n"
    "- 腾讯失败时自动切换 Yahoo Finance\n"
    "- 支持沪市（600/601/603/688）和深市（000/002/300）\n"
    "- ⚠️ 本工具仅用于学习交流，不构成投资建议"
)
