import streamlit as st
import pandas as pd
import requests
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# ==========================================
# 1. 核心计算与数据获取模块
# ==========================================

def calculate_indicators(df):
    """计算战法所需的技术指标 (MA, BBI, KDJ, 量比)"""
    if df is None or len(df) < 20:
        return df
    
    # 移动平均线 (白线MA5, 黄线MA20, 趋势线MA60)
    df['MA5'] = df['close'].rolling(window=5).mean()    # 白线
    df['MA20'] = df['close'].rolling(window=20).mean()  # 黄线 (大哥线)
    df['MA60'] = df['close'].rolling(window=60).mean()
    
    # BBI 多空指标
    ma3 = df['close'].rolling(window=3).mean()
    ma6 = df['close'].rolling(window=6).mean()
    ma12 = df['close'].rolling(window=12).mean()
    ma24 = df['close'].rolling(window=24).mean()
    df['BBI'] = (ma3 + ma6 + ma12 + ma24) / 4
    
    # 成交量均线
    df['VOL5'] = df['volume'].rolling(window=5).mean()
    
    # KDJ 指标 (用于计算 J 值超卖)
    low_list = df['low'].rolling(window=9, min_periods=9).min()
    high_list = df['high'].rolling(window=9, min_periods=9).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    return df

def get_real_time_price(symbol):
    """获取实时价格和量比数据"""
    try:
        # 使用新浪/腾讯接口获取实时快照
        prefix = 'sh' if symbol.startswith('6') else 'sz'
        url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
        r = requests.get(url, timeout=3)
        parts = r.text.split('"')[1].split(',')
        if len(parts) > 30:
            current_price = float(parts[3])
            pre_close = float(parts[2])
            open_price = float(parts[1])
            volume = float(parts[8]) # 股数
            # 简易量比计算 (大概估算，AKShare其实有更准的但较慢)
            return current_price, pre_close, volume
    except:
        pass
    return 0.0, 0.0, 0.0

@st.cache_data(ttl=600)
def fetch_history_data(symbol):
    """获取历史K线数据"""
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    # 腾讯接口 (稳定)
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
    try:
        r = requests.get(url, timeout=5).json()
        data = r.get('data', {}).get(f"{prefix}{symbol}", {}).get('qfqday', [])
        if not data:
            return None
        
        # 解析数据
        df = pd.DataFrame([row[:6] for row in data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        cols = ['open', 'close', 'high', 'low', 'volume']
        df[cols] = df[cols].apply(pd.to_numeric)
        
        # 计算指标
        df = calculate_indicators(df)
        return df
    except Exception as e:
        print(f"Error fetching history: {e}")
        return None

@st.cache_data(ttl=1800)
def get_market_info(symbol):
    """获取个股名称、新闻、资金流"""
    name = symbol
    news = []
    flow = 0.0
    
    try:
        # 1. 名称
        stock_info = ak.stock_individual_info_em(symbol=symbol)
        name = stock_info[stock_info['项目'] == '股票简称']['值'].values[0]
        
        # 2. 新闻
        news_df = ak.stock_news_em(symbol=symbol)
        news = news_df.head(3)[['标题', '发布时间']].to_dict('records')
        
        # 3. 资金流 (个股资金流向)
        flow_df = ak.stock_individual_fund_flow(stock=symbol, market="sh" if symbol.startswith('6') else "sz")
        # 取最近一日的主力净流入 (单位：元 -> 换算成亿)
        if not flow_df.empty:
            flow = flow_df.iloc[0]['主力净流入-净额'] / 100000000 
            
    except:
        pass
        
    return name, news, flow

# ==========================================
# 2. 浩哥战法评分逻辑 (核心)
# ==========================================

def analyze_logic(df, current_price, symbol, name, money_flow):
    """
    评分核心算法
    返回: total_score, comment, advice
    """
    if df is None or len(df) < 20:
        return 0, "数据不足，浩哥没法算。", "观望"

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    # --- A. 基础数据准备 ---
    close = last_row['close']
    vol = last_row['volume']
    ma5 = last_row['MA5']     # 白线
    ma20 = last_row['MA20']   # 黄线 (大哥线)
    j_val = last_row['J']     # KDJ之J值
    vol5 = last_row['VOL5']
    
    # 模拟“今日”换手率 (简易计算: 成交量/预估流通股，这里用量比替代活跃度)
    # 真实换手率需要流通盘数据，这里简化：用当前量/5日均量判断活跃度
    vol_ratio = vol / vol5 if vol5 > 0 else 0
    
    tech_score = 0.0
    signals_triggered = []
    
    # --- B. 信号判定 (模拟 Z哥 战法条件) ---
    
    # 1. 回踩超级B (25分): 极度缩量 + 贴近黄线 + J值极低
    cond_super_b = (vol_ratio < 0.6) and (abs(close - ma20)/ma20 < 0.03) and (j_val < -5)
    
    # 2. 超卖超缩量B (22分): 缩量 + J值负
    cond_super_shrink = (vol_ratio < 0.5) and (j_val < 0)
    
    # 3. 回踩白线B (18分): 贴近白线 + 趋势向上
    cond_white_line = (abs(close - ma5)/ma5 < 0.02) and (ma5 > ma20)
    
    # 4. 原始B1 (15分): 基础分
    cond_basic_b1 = (j_val < 10) and (close > ma20)
    
    # --- C. 权重计算 (取最高满足的信号) ---
    if cond_super_b:
        tech_score += 25.0
        signals_triggered.append("回踩超级B(25分)")
    elif cond_super_shrink:
        tech_score += 22.0
        signals_triggered.append("超卖超缩量B(22分)")
    elif cond_white_line:
        tech_score += 18.0
        signals_triggered.append("回踩白线B(18分)")
    elif cond_basic_b1:
        tech_score += 15.0
        signals_triggered.append("原始B1(15分)")
    else:
        tech_score += 5.0 # 没信号给点辛苦分
    
    # --- D. 精细化加分 (小数) ---
    # J值越低越好：每低1个点，加0.2分，最多加4分
    if j_val < 0:
        j_bonus = min(abs(j_val) * 0.2, 4.0)
        tech_score += j_bonus
    
    # --- E. 低价股复活机制 (关键！) ---
    price_penalty = 0.0
    resurrection_msg = ""
    
    if close < 12.0:
        # 默认惩罚
        base_penalty = -4.0
        
        # 检查复活条件: 量比活跃 或 站上黄线
        is_active = (vol_ratio > 1.2) or (close > ma20 * 1.02)
        
        if is_active:
            price_penalty = 2.0  # 不扣反加
            resurrection_msg = "【低价复活】股价虽低但主力点火，浩哥额外加分！"
        else:
            price_penalty = -4.0
            resurrection_msg = "【低价风险】股价低且无量织布，扣分避坑。"
            
        tech_score += price_penalty
    else:
        # 12-50元 黄金区间
        if 12 <= close <= 50:
            tech_score += 2.0
    
    # --- F. AI 资金流与情绪分 (0-30分) ---
    ai_score = 0.0
    
    # 资金流逻辑
    if money_flow > 1.0: # 流入超1亿
        ai_score += 15.0
    elif money_flow > 0.1: # 小幅流入
        ai_score += 8.0
    elif money_flow < -0.5: # 大幅流出
        ai_score -= 10.0
    
    # 趋势加分
    if close > ma60:
        ai_score += 5.0 # 站稳生命线
        
    # 限制分数范围
    total_score = tech_score + ai_score
    total_score = min(max(total_score, 0), 100.0) # 封顶100
    
    # --- G. 浩哥生成评论 ---
    comment = f"浩哥瞅了瞅 {name}，现价 {close:.2f}。{resurrection_msg} "
    
    if total_score > 85:
        comment += f"🔥 卧槽！这票数据炸裂（{', '.join(signals_triggered)}），资金也在抢筹，这种极品机会别犹豫，干就完了！"
        advice = "重仓出击 (Stop Loss: -5%)"
    elif total_score > 70:
        comment += f"👍 哎哟不错哦，触发了 {signals_triggered[0] if signals_triggered else '优质信号'}，J值漂亮，浩哥觉得可以搞点仓位试试。"
        advice = "中仓买入 (3-5成)"
    elif total_score > 50:
        comment += "🤔 形态一般般，虽然有点小机会，但没到浩哥的心动点。想玩就轻仓，别上头。"
        advice = "轻仓博弈 (1-2成)"
    else:
        comment += "💩 这票走得太臭了，没信号也没钱进，浩哥劝你离远点，别当接盘侠。"
        advice = "空仓观望"

    if money_flow > 0:
        comment += f" 主力资金今日净流入 {money_flow:.2f} 亿，这是真金白银在挺啊！"
    else:
        comment += f" 主力资金今日净流出 {abs(money_flow):.2f} 亿，小心庄家跑路。"

    return total_score, comment, advice

# ==========================================
# 3. K线绘图模块 (Plotly)
# ==========================================
def plot_kline(df, symbol, name):
    df = df.iloc[-120:] # 只看最近半年
    
    # 创建子图: K线 + 成交量
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # K线
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='K线', increasing_line_color='red', decreasing_line_color='green'
    ), row=1, col=1)

    # 均线
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='white', width=1), name='白线(MA5)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='yellow', width=1.5), name='黄线(大哥线)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue', width=1), name='生命线(MA60)'), row=1, col=1)

    # 成交量
    colors = ['red' if row['open'] < row['close'] else 'green' for i, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='成交量'), row=2, col=1)

    # 布局设置
    fig.update_layout(
        title=f"{name} ({symbol}) - 浩哥专用图表",
        yaxis_title='价格',
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='#1e1e1e', # 深色背景专业感
        paper_bgcolor='#0e1117',
        font=dict(color='white')
    )
    return fig

# ==========================================
# 4. Streamlit 主界面
# ==========================================

st.set_page_config(page_title="浩哥战法 AI 版", layout="wide")

st.title("🚀 浩哥战法量化终端 v2.0")
st.markdown("### 专抓主升浪，低价妖股不错过！")

# 侧边栏
with st.sidebar:
    st.header("🔍 股票池")
    user_input = st.text_area("输入代码 (逗号分隔)", "600519, 002415, 601138, 000001")
    run_btn = st.button("开始挖掘", type="primary")
    st.info("💡 **浩哥提示**：\n<12元低价股只要有量、有热点，系统会自动复活加分！")

if run_btn:
    codes = [c.strip() for c in user_input.replace('，', ',').split(',') if c.strip()]
    
    if not codes:
        st.error("兄弟，输个代码再点啊！")
    else:
        for symbol in codes:
            # 1. 基础信息
            name, news_list, money_flow = get_market_info(symbol)
            st.markdown(f"### 📊 {name} ({symbol})")
            
            # 2. 获取数据 (加载动画)
            with st.spinner(f"浩哥正在计算 {name} 的核心指标..."):
                current_price, _, _ = get_real_time_price(symbol)
                df = fetch_history_data(symbol)
            
            # 3. 核心分析
            if df is not None:
                # 传入真实数据计算
                score, comment, advice = analyze_logic(df, current_price, symbol, name, money_flow)
                
                # 4. 结果展示卡片
                with st.container():
                    # 动态颜色
                    score_color = "#ff4b4b" if score >= 80 else "#ffa500" if score >= 60 else "#00c853"
                    
                    c1, c2, c3 = st.columns([1.5, 3.5, 2])
                    
                    with c1:
                        st.markdown(f"""
                        <div style="background-color: {score_color}; padding: 15px; border-radius: 10px; text-align: center;">
                            <h1 style="color: white; margin:0;">{score:.1f}</h1>
                            <p style="color: white; margin:0; font-weight: bold;">浩哥评分</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with c2:
                        st.markdown(f"**💬 浩哥犀利点评：**")
                        st.info(comment)
                    
                    with c3:
                        st.markdown(f"**💡 操作建议：**")
                        st.markdown(f"### {advice}")
                        if money_flow != 0:
                            st.caption(f"主力净流入: {money_flow:.2f} 亿")

                # 5. K线图与新闻 (折叠)
                with st.expander("📈 查看 K线图 & 舆情消息", expanded=True):
                    # 画图
                    fig = plot_kline(df, symbol, name)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 新闻
                    st.markdown("**📢 最新动态：**")
                    if news_list:
                        for n in news_list:
                            st.markdown(f"- {n['发布时间'][5:]} | {n['标题']}")
                    else:
                        st.write("暂无重磅消息。")
                        
            else:
                st.error(f"{symbol} 数据拉取失败，可能是停牌或代码错误。")
            
            st.divider()
