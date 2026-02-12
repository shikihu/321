import streamlit as st
import pandas as pd
import requests
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import time
import urllib3
import warnings
import socket

# ==========================================
# 1. 防卡死核心设置 (新增)
# ==========================================
# 设置全局网络超时时间为10秒，防止接口无限等待导致程序假死
socket.setdefaulttimeout(10)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

st.set_page_config(page_title="浩哥战法量化终端 v4.2 (防卡死极速版)", layout="wide")

# ==========================================
# 2. 核心逻辑优化：合并数据提取
# ==========================================
def process_stock_info(symbol):
    """
    一次性获取个股资料，同时提取【名称】和【基本面数据】
    避免重复请求导致卡顿
    """
    name = symbol
    basic_score = 0
    
    try:
        # 只请求一次接口
        df = ak.stock_individual_info_em(symbol=str(symbol))
        
        if df is not None and not df.empty:
            # 标准化列名，处理不同版本的返回值
            if len(df.columns) >= 2:
                df.columns = ['key', 'val']
                
                # --- 1. 提取名称 ---
                name_mask = df['key'].astype(str).str.contains("简称", na=False)
                if name_mask.any():
                    val = df.loc[name_mask, 'val'].values[0]
                    if pd.notna(val) and val != '':
                        name = str(val)
                
                # --- 2. 提取基本面评分 (模糊匹配修复版) ---
                def get_val(keyword):
                    try:
                        mask = df['key'].astype(str).str.contains(keyword, na=False)
                        if not mask.any(): return 0
                        v_str = str(df.loc[mask, 'val'].values[0])
                        if v_str in ['-', 'null', 'None', '', 'nan'] or '亏损' in v_str:
                            return 0
                        return float(v_str)
                    except:
                        return 0
                
                pe = get_val('市盈率')
                pb = get_val('市净率')
                roe = get_val('净资产收益')
                
                if pe > 0 and pe < 30: basic_score += 8
                if pb > 0 and pb < 3: basic_score += 6
                if roe > 10: basic_score += 6
                basic_score = min(20, basic_score)
                
    except Exception:
        # 如果获取失败，就用默认值，保证程序不崩
        pass
        
    return name, basic_score

def get_news_face(symbol):
    """获取消息面评分，带极短超时保护"""
    try:
        # 新闻接口如果不重要，可以设个陷阱，失败就跳过
        news = ak.stock_news_em(symbol=str(symbol))
        if news is not None and not news.empty and '标题' in news.columns:
            # 只取最近10条，提高处理速度
            recent_news = news.head(10)
            title_text = ' '.join(recent_news['标题'].astype(str).tolist())
            
            pos_k = ['好', '利好', '上涨', '爆拉', '涨停', '增长', '业绩', '增持', '回购', '突破']
            neg_k = ['坏', '利空', '下跌', '暴跌', '跌停', '亏损', '风险', '减持', '被查', '立案']
            
            pos = sum(1 for k in pos_k if k in title_text)
            neg = sum(1 for k in neg_k if k in title_text)
            
            score = (pos - neg) * 3
            return min(15, max(0, score))
    except Exception:
        pass
    return 0

def get_emotion_face(df):
    if df is None or len(df) < 1: return 0
    try:
        last_vol = df['volume'].iloc[-1]
        # 简单估算：假设大致市值换算，仅作参考
        # 更好的方式是用换手率列，如果akshare返回的数据里有换手率直接用
        # 这里维持原逻辑，但增加安全性
        if last_vol > 0:
            # 盲猜逻辑：如果成交量巨大可能是高换手，这里仅做示例逻辑保护
            # 实战中建议直接看量比或缩量逻辑
            return 5 
    except:
        pass
    return 5 # 默认给个中间分

@st.cache_data(ttl=1800)
def get_money_flow(symbol):
    try:
        market = "sh" if str(symbol).startswith('6') else "sz"
        flow = ak.stock_individual_fund_flow(stock=str(symbol), market=market)
        if flow is not None and not flow.empty:
            val = flow.iloc[0]['主力净流入-净额']
            if isinstance(val, str):
                val = val.replace('万', '').replace('亿', '')
            return float(val) / 100000000
    except:
        pass
    return 0.0

@st.cache_data(ttl=3600)
def fetch_history_data(symbol):
    # 腾讯接口通常比较快且稳
    symbol = str(symbol).strip()
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    try:
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
        # 强制设置 headers connection close
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

@st.cache_data(ttl=60)
def get_real_time_price(symbol, df_hist):
    # 优先取实时接口，取不到就用历史收盘
    try:
        prefix = 'sh' if str(symbol).startswith(('6', '9')) else 'sz'
        url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
        r = requests.get(url, headers={'Referer': 'http://sina.com.cn'}, timeout=3)
        if 'var hq_str_' in r.text:
            parts = r.text.split('"')[1].split(',')
            if len(parts) > 3:
                price = float(parts[3])
                if price > 0: return price, "实时"
    except:
        pass
    
    if df_hist is not None and not df_hist.empty:
        return df_hist['close'].iloc[-1], "收盘"
    return 0, "无"

# ==========================================
# 3. 指标计算 (保持浩哥原版逻辑)
# ==========================================
def calculate_indicators(df):
    if df is None or len(df) < 5: return df
    df = df.copy()
    # 均线
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    # 浩哥专用线
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
    
    # 成交量逻辑
    df['vol_max20'] = df['volume'].rolling(20).max()
    df['vol_max30'] = df['volume'].rolling(30).max()
    df['vol_max50'] = df['volume'].rolling(50).max()
    
    df['缩量'] = (df['volume'] < df['vol_max20'] * 0.416) | (df['volume'] < df['vol_max50'] / 3)
    df['回踩缩量'] = (df['volume'] < df['vol_max20'] * 0.45) | (df['volume'] < df['vol_max50'] / 3)
    df['适当缩量'] = (df['volume'] < df['vol_max20'] * 0.618) | (df['volume'] < df['vol_max50'] / 3)
    df['超缩量'] = (df['volume'] < df['vol_max30'] / 4) | (df['volume'] < df['vol_max50'] / 6)
    
    df['当日振幅'] = (df['high'] - df['low']) / df['low'] * 100
    df['当日涨跌幅'] = (df['close'] - df.shift(1)['close']) / df.shift(1)['close'] * 100
    df['收阳线'] = df['close'] > df['open']
    
    # 战法前置条件
    df['做上涨趋势'] = (df['趋势白线'] >= df['大哥黄线'] * 0.999) & \
                      (df['close'] >= df['大哥黄线'] * 0.975)
    
    dist_white = abs(df['close'] - df['趋势白线']) / df['close'] * 100
    df['回踩白线'] = ((df['close'] >= df['趋势白线']) & (dist_white <= 2)) | \
                   ((df['close'] < df['趋势白线']) & (dist_white < 0.8))

    dist_yellow = abs(df['close'] - df['大哥黄线']) / df['大哥黄线'] * 100
    df['回踩黄线'] = ((df['close'] >= df['大哥黄线']) & (dist_yellow <= 2)) | \
                   ((df['close'] < df['大哥黄线']) & (dist_yellow <= 0.8))

    # 战法判定 (简化版逻辑，保持核心准确)
    df['浩哥缩量战法'] = df['做上涨趋势'] & (df['J'] < 14) & df['缩量'] & (df['当日振幅'] < 8)
    df['浩哥极缩战法'] = df['做上涨趋势'] & (df['J'] < 14) & df['超缩量']
    df['浩哥拐头战法'] = df['做上涨趋势'] & (df['RSI'] > df.shift(1)['RSI']) & (df.shift(1)['RSI'] < 25) & df['缩量']
    df['浩哥白线战法'] = df['做上涨趋势'] & df['回踩白线'] & df['回踩缩量'] & (df['J'] < 30)
    df['浩哥超级战法'] = (df['close'] > df['MA20']) & df['适当缩量'] & (df['J'] < 35) & df['回踩白线']
    
    df = df.ffill().bfill()
    return df

def analyze_stock_logic(df, name, current, symbol, money_flow, basic_score, news_score):
    if df is None or len(df) < 20: return 0, "数据不足", "观望"
    last = df.iloc[-1]
    
    price = current if current > 0 else last['close']
    tier = 'mid'
    if price <= 12: tier = 'low'
    elif price > 50: tier = 'high'
    
    # 战法 & 胜率
    wins = {
        '浩哥缩量战法': {'low': 54, 'mid': 58, 'high': 56},
        '浩哥极缩战法': {'low': 59, 'mid': 63, 'high': 61},
        '浩哥拐头战法': {'low': 57, 'mid': 60, 'high': 59},
        '浩哥白线战法': {'low': 61, 'mid': 65, 'high': 63},
        '浩哥超级战法': {'low': 64, 'mid': 68, 'high': 66}
    }
    
    triggered = []
    for sig, w in wins.items():
        if sig in df.columns and last[sig]:
            triggered.append((sig, w[tier]))
            
    tech_score = 0
    sig_name = "无特定战法信号"
    
    if triggered:
        sig_name, rate = max(triggered, key=lambda x: x[1])
        if rate > 60: tech_score = 45
        elif rate > 55: tech_score = 35
        else: tech_score = 25
    
    total = tech_score + basic_score + news_score + 5 # 情绪给保底5分
    
    obs = []
    if last['macd'] > 0: obs.append("MACD红柱")
    else: obs.append("MACD绿柱")
    if last['volume'] < last['vol_max20'] * 0.5: obs.append("量能萎缩")
    
    comment = f"**{name}** ({symbol}): 现价 {price}。\n"
    comment += f"📊 信号：{sig_name}\n"
    comment += f"🔍 状态：{'，'.join(obs)}\n"
    comment += f"💰 主力：{money_flow}亿\n"
    comment += f"🏆 总分：{total:.0f} (技术{tech_score}+基本{basic_score:.0f}+消息{news_score:.0f})"
    
    advice = "建议关注" if total > 70 else "建议观望"
    return total, comment, advice

# ==========================================
# 4. 主程序
# ==========================================
st.title("浩哥战法量化终端 v4.2 (防卡死极速版)")
codes_input = st.text_area("输入代码 (例如: 002446, 600868)", height=100)

if st.button("🚀 开始分析"):
    codes = re.findall(r'\d{6}', codes_input)
    codes = list(set(codes))[:50]
    
    if not codes:
        st.error("请输入代码")
    else:
        results = []
        bar = st.progress(0)
        status = st.empty()
        
        for i, code in enumerate(codes):
            status.text(f"正在分析 {code} ({i+1}/{len(codes)})...")
            
            # 1. 获取K线 (最快)
            df = fetch_history_data(code)
            
            if df is not None:
                # 2. 获取名称和基本面 (合并请求，防卡)
                name, basic_score = process_stock_info(code)
                
                # 3. 其他数据
                curr_price, _ = get_real_time_price(code, df)
                money = get_money_flow(code)
                news_score = get_news_face(code)
                
                # 4. 综合分析
                score, cmt, adv = analyze_stock_logic(df, name, curr_price, code, money, basic_score, news_score)
                
                results.append({
                    'code': code, 'name': name, 'score': score, 
                    'cmt': cmt, 'adv': adv, 'df': df
                })
            
            bar.progress((i+1)/len(codes))
            # 必须有的间隔，防止接口封IP
            time.sleep(0.5) 
            
        status.success("完成！")
        results.sort(key=lambda x: x['score'], reverse=True)
        
        for res in results:
            with st.expander(f"{res['name']} ({res['code']}) - {res['score']:.0f}分", expanded=True):
                c1, c2 = st.columns([3, 1])
                with c1: st.markdown(res['cmt'])
                with c2: st.info(res['adv'])
                
                # 绘图
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                df_p = res['df'].iloc[-100:]
                fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['open'], high=df_p['high'], 
                                           low=df_p['low'], close=df_p['close'], name='K线'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p['趋势白线'], line=dict(color='white'), name='白线'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p['大哥黄线'], line=dict(color='yellow'), name='黄线'), row=1, col=1)
                fig.add_trace(go.Bar(x=df_p.index, y=df_p['volume'], name='成交量'), row=2, col=1)
                fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), plot_bgcolor='#111')
                st.plotly_chart(fig, use_container_width=True)
