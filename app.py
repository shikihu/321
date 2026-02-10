import streamlit as st
import pandas as pd
import requests
import numpy as np
import time
import akshare as ak

# ======================
# 数据获取
# ======================
def get_real_time_price(symbol):
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
    try:
        r = requests.get(url, timeout=5)
        text = r.text.strip()
        if text.startswith('var hq_str_'):
            parts = text.split('"')[1].split(',')
            if len(parts) >= 4:
                return float(parts[3])
    except:
        pass
    return 0.0

def fetch_stock_history(symbol):
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
    try:
        r = requests.get(url, timeout=8).json()
        data = r.get('data', {}).get(f"{prefix}{symbol}", {}).get('qfqday', [])
        if not data:
            return None
        df = pd.DataFrame([row[:6] for row in data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        if len(df) < 20:
            return None
        return df
    except:
        return None

# ======================
# 股票名称 + 新闻 + 龙虎榜
# ======================
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

def get_stock_news(symbol):
    try:
        news = ak.stock_news_em(symbol=symbol)
        return news.head(3)[['标题', '发布时间', '来源']].to_dict('records')
    except:
        return []

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

# ======================
# 浩哥战法评分（专业版 + 7种浩哥战法体现）
# ======================
def analyze_stock(df, name, current):
    if df is None or len(df) < 2:
        return 0.0, f"浩哥看 {name} 数据不足，无法分析。", "浩哥建议：暂缓操作。"
    
    last = df.iloc[-1]
    
    # 安全访问
    def safe_get(col, default=0.0):
        return last.get(col, default) if col in last else default
    
    # 7种浩哥战法激活判断（内部，不显示原名）
    signals = {
        '浩哥拐头战法': (safe_get('rsi') - 15 >= df['rsi'].shift(1).iloc[-1]) and (df['rsi'].shift(1).iloc[-1] < 20) and safe_get('当日振幅', 999) < 8,
        '浩哥缩量战法': safe_get('缩量', False) and safe_get('j', 0) < 14,
        '浩哥1.0战法': safe_get('趋势白线', 0) > safe_get('大哥黄线', 0) and safe_get('缩量', False),
        '浩哥极缩战法': safe_get('超缩量', False) and safe_get('j', 0) < 14,
        '浩哥白线战法': abs(last['close'] - safe_get('趋势白线', last['close'])) / last['close'] * 100 < 2 and safe_get('缩量', False),
        '浩哥超级战法': safe_get('超牛股', False) and safe_get('缩量', False),
        '浩哥黄线战法': abs(last['close'] - safe_get('大哥黄线', last['close'])) / last['close'] * 100 <= 1.5 and safe_get('缩量', False)
    }
    
    # 权重（回测胜率越高权重越高）
    weights = {
        '浩哥超级战法': 25.0,
        '浩哥极缩战法': 22.0,
        '浩哥白线战法': 18.0,
        '浩哥1.0战法': 15.0,
        '浩哥拐头战法': 10.0,
        '浩哥黄线战法': 8.0,
        '浩哥缩量战法': 5.0
    }
    
    # 技术分计算
    tech_score = 0.0
    triggered_signals = []
    for sig, active in signals.items():
        if active:
            tech_score += weights[sig]
            triggered_signals.append(sig)
    
    # J值动态加分（不提 J 值，只加分）
    j_val = safe_get('j', 0)
    if j_val < 0:
        tech_score += min(abs(j_val) * 0.3, 4.0)
    
    # 低价股复活机制
    price_correction = 0.0
    if current < 12:
        price_correction = -5.0
        is_active = (safe_get('换手率', 0) > 5) or (safe_get('量比', 0) > 1.5) or \
                    (last['close'] > safe_get('大哥黄线', last['close']) and safe_get('macd', 0) > 0)
        if is_active:
            price_correction = +3.0
    tech_score += price_correction
    
    tech_score = min(max(tech_score, 0), 70.0)
    
    # AI 面（0-30）
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
    
    # 触发信号提示（让你知道是哪种浩哥战法）
    if triggered_signals:
        comment += f"浩哥检测到：{ ' + '.join(triggered_signals) }\n\n"
    
    # 分数组成
    comment += f"【技术面评分】{tech_score:.1f}/70\n"
    comment += f"【AI 面评分】{ai_score:.1f}/30\n"
    comment += f"【浩哥综合打分】{total_score:.1f}/100\n\n"
    
    # 浩哥点评（专业版）
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

# 主界面
st.title("浩哥战法")

codes_input = st.text_input("输入股票代码（逗号分隔，如 600519,601218）")
if st.button("让浩哥分析"):
    codes = [c.strip() for c in codes_input.split(',') if c.strip()]
    for symbol in codes:
        stock_name = get_stock_name(symbol)
        st.subheader(f"浩哥看 {symbol} - {stock_name}")
        
        df = fetch_stock_history(symbol)
        current = get_real_time_price(symbol)
        news = get_stock_news(symbol)
        lhb_net = get_lhb_data(symbol)
        
        total_score, comment, buy_advice = analyze_stock(df, stock_name, current)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("浩哥打分", f"{total_score:.1f}/100", delta_color="normal")
        with col2:
            st.write("**浩哥评论：**")
            st.info(comment)
            st.write("**浩哥建议：**", buy_advice)
        
        st.markdown("---")

st.sidebar.success("浩哥战法已就绪！")
st.sidebar.info("浩哥亲自点评，实时价格 + 股票名称，评论专业理性。分享给朋友们用吧！")
