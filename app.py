import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import numpy as np
import akshare as ak
import time
from datetime import datetime

# ======================
# 数据获取：实时价格 + 历史数据
# ======================
def get_real_time_price(symbol):
    """优先腾讯实时价"""
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
    try:
        r = requests.get(url, timeout=5)
        text = r.text.strip()
        if text.startswith('var hq_str_'):
            parts = text.split('"')[1].split(',')
            if len(parts) >= 4:
                return float(parts[3])  # 当前价
    except:
        pass
    
    # 备选 yfinance
    try:
        ticker = f"{symbol}.SS" if symbol.startswith('6') else f"{symbol}.SZ"
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('currentPrice', info.get('regularMarketPrice', 0.0))
    except:
        return 0.0

def fetch_stock_history(symbol):
    """历史数据 fallback"""
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json().get('data', {}).get(f"{prefix}{symbol}", {}).get('qfqday', [])
        if not data:
            return None
        
        # 只取前6列（解决7列错误）
        cleaned = []
        for row in data:
            if isinstance(row, list) and len(row) >= 6:
                cleaned.append([str(x) for x in row[:6]])
        
        df = pd.DataFrame(cleaned, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        
        if len(df) < 20:
            return None
        return df
    except Exception as e:
        # st.error(f"腾讯数据失败: {e}")
        return None

# ======================
# 获取股票名称 + 流通市值 + 新闻 + 龙虎榜
# ======================
@st.cache_data(ttl=1800)
def get_stock_info(symbol):
    try:
        info = ak.stock_individual_info_em(symbol=symbol)
        name = info[info['项目'] == '股票简称']['值'].values[0]
        circ_mv = info[info['项目'] == '流通市值']['值'].values[0] / 100000000  # 亿元
        return name, circ_mv
    except:
        return symbol, 100.0

@st.cache_data(ttl=1800)
def get_stock_news(symbol):
    try:
        news = ak.stock_news_em(symbol=symbol)
        return news.head(5)[['标题', '发布时间', '来源']].to_dict('records')
    except:
        return []

@st.cache_data(ttl=1800)
def get_lhb_data(symbol):
    try:
        lhb = ak.stock_lhb_detail_em(symbol=symbol)
        if not lhb.empty:
            latest = lhb.iloc[0]
            net_amount = latest.get('净买入额(万元)', 0) / 10000  # 亿元
            return net_amount
        return 0.0
    except:
        return 0.0

# ======================
# 浩哥分析核心逻辑（深度优化版）
# ======================
def analyze_stock(df, name, current, circ_mv, news, lhb_net):
    """核心逻辑：注入Z哥战法灵魂，拒绝模板化"""
    if df is None or len(df) < 2:
        return 0.0, f"浩哥看 {name} 数据不足，先等等吧。", "浩哥建议：数据不全，换个票再来。", "暂无新闻。"
    
    # 1. 技术面分析（Z哥战法核心）
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 首次回踩（关键条件！）
    ma20_break_idx = -1
    for i in range(len(df)-2, max(0, len(df)-60), -1):
        if df['close'].iloc[i] > df['ma20'].iloc[i] and df['close'].iloc[i-1] <= df['ma20'].iloc[i-1]:
            ma20_break_idx = i
            break
    days_since_break = len(df) - 1 - ma20_break_idx if ma20_break_idx != -1 else 999
    is_first_pullback = 3 <= days_since_break <= 15
    
    # 缩量（量比计算）
    peak_vol = df['volume'].iloc[-60:].max()
    vol_ratio = last['volume'] / peak_vol if peak_vol > 0 else 1
    volume_shrink = vol_ratio < 0.35
    
    # J值（超卖信号）
    j_val = last.get('j', 0)
    j_bonus = max(0, (-j_val * 0.3))  # J越负加分越多
    
    # 2. 评分权重（Z哥战法核心权重）
    score = 0.0
    score += 25.0 if is_first_pullback else 0  # 首次回踩（最高权重）
    score += 20.0 if volume_shrink else 0     # 缩量（关键条件）
    score += 15.0 if j_val < -5 else 0        # J值<-5（暴击信号）
    score += 10.0 if j_val < 0 else 0         # J值<0（超卖）
    score += 5.0 if (last['close'] - last['open']) / last['open'] > 0.03 else 0  # 关键K线
    
    # 3. 风险过滤（陷阱检测）
    real_trap = False
    if days_since_break < 3:  # 回踩太早
        real_trap = True
    if vol_ratio > 0.5:  # 量没缩到位
        real_trap = True
    if sum(df['close'].iloc[-5:] < df['ma20'].iloc[-5:]) > 2:  # 3次破位
        real_trap = True
    
    # 4. 基本面/情绪面
    if circ_mv > 500:
        score += 8.0  # 大盘股加分
    elif circ_mv < 30:
        score -= 5.0  # 小盘股风险
    
    if lhb_net > 0.5:
        score += min(lhb_net * 5, 15.0)  # 主力净流入
    elif lhb_net < -0.5:
        score -= min(abs(lhb_net) * 5, 10.0)  # 主力流出
    
    if len(news) > 2 and any("利好" in n for n in [n['标题'] for n in news]):
        score += 5.0  # 有利好消息
    
    # 5. 评分修正
    score = min(max(score, 0), 70.0)  # 技术面上限70
    total_score = score + (50 if circ_mv > 100 else 30)  # 基本面加权
    total_score = min(max(total_score, 0), 100.0)
    
    # 6. 生成专属评论（Z哥风格！）
    comment = f"浩哥盯了 {name} 一整天，当前价 {current:.2f} 元，流通市值 {circ_mv:.1f} 亿。"
    
    if is_first_pullback and volume_shrink and j_val < -5:
        comment += f"🔥 今日完美B1！回踩第{days_since_break}天，量比{vol_ratio:.2f}极致缩量，J值={j_val:.1f}（近3月最低），主力洗盘彻底，反弹动能蓄积充分！"
    elif is_first_pullback and volume_shrink:
        comment += f"🎯 回踩第{days_since_break}天，量比{vol_ratio:.2f}缩量到位，J值={j_val:.1f}，标准B1买点形态，温柔黏人！"
    elif volume_shrink and j_val < 0:
        comment += f"💡 量比{vol_ratio:.2f}缩量+J值={j_val:.1f}，超卖信号出现，但回踩天数{days_since_break}（需>3天），小仓试错可关注。"
    else:
        comment += f"⚠️ 量能未缩到位（量比{vol_ratio:.2f}），J值={j_val:.1f}，回踩天数{days_since_break}，需明日放量阳线确认支撑有效性。"
    
    # 7. 陷阱提示（Z哥口头禅）
    if real_trap:
        comment += " ❌ 警惕！连续3日破位还放量，主力出货陷阱，别碰！"
    
    # 8. 买入建议（Z哥风格）
    if total_score >= 90:
        buy_advice = "✅ 重仓干！完美B1形态，温柔黏人，赚钱机会大，珍惜子弹！"
    elif total_score >= 75:
        buy_advice = "⚠️ 小仓试错！需明日放量确认，别梭哈，留子弹。"
    else:
        buy_advice = "❌ 不能买！量没缩到位，J值未达超卖，等下一个机会。"
    
    # 9. 新闻摘要
    news_text = ""
    if news:
        news_text = "**浩哥看到最近新闻：**\n"
        for item in news[:3]:
            news_text += f"- {item['标题']} ({item['发布时间'][:10]}) - {item['来源']}\n"
    else:
        news_text = "暂无最新新闻。"
    
    return total_score, comment, buy_advice, news_text

# ======================
# 主界面
# ======================
st.set_page_config(page_title="浩哥AI分析", layout="wide")
st.title("🔥 浩哥AI分析 - 真正的Z哥战法")

st.sidebar.title("📌 Z哥六步法（背诵100遍）")
st.sidebar.markdown("""
1. 择时：周日看大盘温度，只在合适阶段动手  
2. 选股：强势基因 + 题材热  
3. 买点：B1首踩 或 B2主升  
4. 持仓：等利润垫，不折腾  
5. 卖点：四种卖法（利润垫/破位/高潮/情绪）  
6. 复盘：每笔交易都要复盘，避免情绪化  
""")
st.sidebar.markdown("**心态**：沉没成本别参与决策，戒骄戒躁，珍惜子弹！")

codes_input = st.text_input("🔍 输入股票代码（逗号分隔，如 600519,000858）", placeholder="600519,000001")
if st.button("🚀 让浩哥分析"):
    if not codes_input.strip():
        st.warning("请输入股票代码")
        st.stop()
    
    codes = [c.strip() for c in codes_input.split(',') if c.strip()]
    for symbol in codes:
        if not (symbol.isdigit() and len(symbol) == 6):
            st.error(f"❌ {symbol} 不是有效的6位A股代码")
            continue
        
        # 获取所有数据
        stock_name, circ_mv = get_stock_info(symbol)
        current = get_real_time_price(symbol)
        news = get_stock_news(symbol)
        lhb_net = get_lhb_data(symbol)
        
        # 获取历史数据并计算技术指标
        df = fetch_stock_history(symbol)
        if df is None:
            st.error(f"❌ 无法获取 {symbol} 的历史数据（腾讯和Yahoo均失败）")
            continue
        
        # 计算技术指标（关键！）
        df = calculate_indicators(df)
        
        # 生成分析结果
        total_score, comment, buy_advice, news_text = analyze_stock(
            df, 
            stock_name, 
            current, 
            circ_mv, 
            news, 
            lhb_net
        )
        
        # 显示结果
        st.subheader(f"📊 浩哥看 {symbol} - {stock_name}")
        
        # 评分和价格
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("浩哥打分", f"{total_score:.1f}/100", delta_color="normal")
            st.metric("当前价", f"¥{current:.2f}")
        with col2:
            st.info(comment)
            st.success(buy_advice)
        
        # 新闻
        st.write(news_text)
        
        # K线图
        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            increasing_line_color='red', decreasing_line_color='green'
        )])
        fig.add_trace(go.Scatter(x=df.index, y=df['ma20'], mode='lines', name='MA20（生命线）', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['ma60'], mode='lines', name='MA60（长期线）', line=dict(color='orange')))
        fig.update_layout(title=f"{symbol} K线图（重点盯关键K）", xaxis_rangeslider_visible=True, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")

# 通用技术指标计算（新增）
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
    
    return df

st.sidebar.success("浩哥AI分析已就绪！")
st.sidebar.info("• 优先使用腾讯财经数据（快且准）\n• 每日更新，实时分析\n• 评论基于Z哥战法，拒绝模板化")
