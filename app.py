import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import time
import socket

# ==========================================
# 1. 基础配置
# ==========================================
socket.setdefaulttimeout(10)
st.set_page_config(page_title="浩哥战法量化终端 v6.0 (腾讯直连版)", layout="wide")

# ==========================================
# 2. 核心引擎：腾讯直连 (最稳)
# ==========================================
def get_tencent_realtime_data(symbol):
    """
    通过腾讯接口获取：名称、价格、PE、PB、换手率
    接口地址: http://qt.gtimg.cn/q=sh600519
    优点: 纯文本流，速度极快，不封IP，不依赖akshare爬虫
    """
    symbol = str(symbol).strip()
    prefix = 'sh' if symbol.startswith(('6', '9')) else 'sz'
    code = f"{prefix}{symbol}"
    
    try:
        url = f"http://qt.gtimg.cn/q={code}"
        # 腾讯接口通常是GBK编码
        r = requests.get(url, timeout=3)
        r.encoding = 'gbk' 
        text = r.text
        
        # 格式: v_sh600519="1~贵州茅台~600519~1730.00~..."
        if f'v_{code}="' in text:
            data_str = text.split('"')[1]
            parts = data_str.split('~')
            
            if len(parts) > 45:
                # 解析腾讯数据字段 (常用索引)
                name = parts[1]
                price = float(parts[3])
                turnover = float(parts[38]) if parts[38] != '' else 0 # 换手率
                pe = float(parts[39]) if parts[39] != '' else 0       # 市盈率(动态)
                pb = float(parts[46]) if parts[46] != '' else 0       # 市净率
                market_val = float(parts[45]) if parts[45] != '' else 0 # 总市值
                
                return {
                    'name': name,
                    'price': price,
                    'pe': pe,
                    'pb': pb,
                    'turnover': turnover,
                    'mv': market_val
                }
    except Exception as e:
        # print(f"解析失败: {e}")
        pass
    
    return None

@st.cache_data(ttl=3600)
def fetch_history_data(symbol):
    """
    获取K线历史数据 (依然用腾讯，保持数据源一致)
    """
    symbol = str(symbol).strip()
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    try:
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
        r = requests.get(url, headers={'Connection': 'close'}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            key = f"{prefix}{symbol}"
            day_data = data.get('data', {}).get(key, {}).get('qfqday', [])
            if not day_data:
                day_data = data.get('data', {}).get(key, {}).get('day', [])
            
            if day_data:
                df = pd.DataFrame([row[:6] for row in day_data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.apply(pd.to_numeric, errors='coerce')
                return calculate_indicators(df)
    except:
        pass
    return None

# ==========================================
# 3. 指标计算 (浩哥战法核心)
# ==========================================
def calculate_indicators(df):
    if df is None or len(df) < 5: return df
    df = df.copy()
    
    # 基础均线
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    # 浩哥专用线 (加权)
    ema9 = df['close'].ewm(span=9, adjust=False).mean()
    df['趋势白线'] = ema9.ewm(span=11, adjust=False).mean()
    
    ema_vals = [df['close'].ewm(span=x, adjust=False).mean().ewm(span=x, adjust=False).mean() for x in [7,14,28,56]]
    df['大哥黄线'] = sum(ema_vals) / 4
    
    # KDJ
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(com=13).mean() / down.ewm(com=13).mean()
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = (exp12 - exp26 - (exp12 - exp26).ewm(span=9, adjust=False).mean()) * 2
    
    # 量能逻辑
    df['vol_max20'] = df['volume'].rolling(20).max()
    df['缩量'] = (df['volume'] < df['vol_max20'] * 0.5) # 简化判定
    
    # 战法条件
    df['做上涨趋势'] = (df['趋势白线'] >= df['大哥黄线'] * 0.999)
    dist = abs(df['close'] - df['趋势白线']) / df['close'] * 100
    df['回踩白线'] = (dist <= 2.5)
    
    # 信号生成
    df['浩哥缩量战法'] = df['做上涨趋势'] & (df['J'] < 25) & df['缩量']
    df['浩哥超级战法'] = (df['close'] > df['MA20']) & (df['J'] < 45) & df['回踩白线']
    
    df = df.ffill().bfill()
    return df

# ==========================================
# 4. 分析逻辑
# ==========================================
def analyze_stock(code, info_dict, history_df):
    if not info_dict: return None
    
    name = info_dict['name']
    price = info_dict['price']
    pe = info_dict['pe']
    pb = info_dict['pb']
    turnover = info_dict['turnover']
    
    # --- 1. 基本面评分 (20分) ---
    basic_score = 0
    # PE逻辑: 0-35为佳
    if pe > 0 and pe < 35: basic_score += 10
    # PB逻辑: 蓝筹股放宽
    if pb > 0 and pb < 4: basic_score += 10
    elif pb >= 4 and pe < 25: basic_score += 5 # 高PB但低PE也给分
    
    # --- 2. 情绪面 (15分) ---
    emotion_score = 0
    if turnover > 0.5 and turnover < 3: emotion_score += 10
    elif turnover >= 3 and turnover < 8: emotion_score += 15
    elif turnover >= 8: emotion_score += 8
    else: emotion_score += 5
    
    # --- 3. 简单消息面 (10分，这里不再请求防止卡死) ---
    news_score = 5 # 默认给个中间分，不再发请求
    
    # --- 4. 技术面 (55分) ---
    tech_score = 0
    sig_msg = "无特别信号"
    advice = "观望"
    
    if history_df is not None:
        last = history_df.iloc[-1]
        
        if last['做上涨趋势']: tech_score += 15
        if last['macd'] > 0: tech_score += 10
        if last['缩量']: tech_score += 5
        
        if last['浩哥缩量战法']:
            tech_score += 25
            sig_msg = "🎯 浩哥缩量战法 (高胜率)"
        elif last['浩哥超级战法']:
            tech_score += 20
            sig_msg = "⚡ 浩哥超级战法"
            
    # 汇总
    total = basic_score + emotion_score + news_score + tech_score
    total = min(100, total)
    
    if total > 75: advice = "建议重点关注"
    elif total > 60: advice = "可以适当关注"
    
    comment = f"**{name}** ({code}) 现价:{price}\n\n"
    comment += f"📊 PE(动): **{pe:.1f}** | PB: **{pb:.2f}** | 换手: **{turnover}%**\n"
    comment += f"📡 信号: **{sig_msg}**\n"
    comment += f"📝 评分: 基本{basic_score} + 情绪{emotion_score} + 技术{tech_score} + 消息5\n"
    
    return {
        'code': code, 'name': name, 'score': total, 
        'comment': comment, 'advice': advice, 
        'history': history_df
    }

# ==========================================
# 5. 主界面
# ==========================================
st.title("浩哥战法量化终端 v6.0 (腾讯直连版)")
st.caption("🚀 使用腾讯财经原生接口，解决Akshare报错、连接断开等问题，速度极快。")

codes_input = st.text_area("输入股票代码 (例如: 600519, 002446)", height=100)

if st.button("开始分析"):
    codes = re.findall(r'\d{6}', codes_input)
    codes = list(set(codes))[:50]
    
    if not codes:
        st.warning("请输入代码")
    else:
        results = []
        progress = st.progress(0)
        status = st.empty()
        
        for i, code in enumerate(codes):
            status.text(f"正在获取 {code} 数据...")
            
            # 1. 腾讯直连获取实时信息(含PE/PB)
            # 这里绝不会报错 Connection aborted，因为是标准HTTP
            realtime_info = get_tencent_realtime_data(code)
            
            if realtime_info:
                # 2. 获取K线
                hist_df = fetch_history_data(code)
                
                # 3. 分析
                res = analyze_stock(code, realtime_info, hist_df)
                results.append(res)
            else:
                st.warning(f"{code} 获取失败，请检查代码是否正确")
                
            progress.progress((i+1)/len(codes))
            time.sleep(0.2) # 轻微延时即可
            
        status.success("分析完成！")
        results.sort(key=lambda x: x['score'], reverse=True)
        
        for res in results:
            with st.expander(f"{res['name']} ({res['code']}) - {res['score']:.0f}分", expanded=True):
                c1, c2 = st.columns([3,1])
                with c1: st.markdown(res['comment'])
                with c2: 
                    if res['score'] > 70: st.success(res['advice'])
                    else: st.info(res['advice'])
                
                if res['history'] is not None:
                    df_p = res['history'].iloc[-120:]
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
                    
                    # K线
                    fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['open'], high=df_p['high'],
                                               low=df_p['low'], close=df_p['close'], name='K线'), row=1, col=1)
                    # 均线
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['趋势白线'], line=dict(color='white', width=1), name='白线'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['大哥黄线'], line=dict(color='yellow', width=1), name='黄线'), row=1, col=1)
                    # 成交量
                    colors = ['red' if r['close'] >= r['open'] else 'green' for i,r in df_p.iterrows()]
                    fig.add_trace(go.Bar(x=df_p.index, y=df_p['volume'], marker_color=colors, name='成交量'), row=2, col=1)
                    
                    fig.update_layout(height=450, margin=dict(l=0,r=0,t=0,b=0), plot_bgcolor='#111', paper_bgcolor='#111', font=dict(color='white'))
                    fig.update_xaxes(showgrid=False)
                    fig.update_yaxes(showgrid=True, gridcolor='#333')
                    st.plotly_chart(fig, use_container_width=True)
