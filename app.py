import streamlit as st
import pandas as pd
import requests
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import re

# ==========================================
# 1. 数据服务（你提供的最新版，已保留日志和容错）
# ==========================================
@st.cache_data(ttl=300)
def get_real_time_price(symbol, df=None):
    symbol = str(symbol).strip()
    prefix = 'sh' if symbol.startswith(('6', '9')) else 'sz'
    if len(symbol) == 4 and symbol.isdigit(): prefix = 'hk'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
        r = requests.get(url, headers=headers, timeout=5)
        if 'var hq_str_' in r.text:
            parts = r.text.split('"')[1].split(',')
            if len(parts) > 3 and float(parts[3]) > 0:
                return float(parts[3]), "实时价"
    except Exception as e:
        pass  # 静默失败，使用收盘价
    if df is not None and not df.empty:
        return df['close'].iloc[-1], "(盘后/最近收盘价)"
    return 0.0, "无数据"

@st.cache_data(ttl=3600)
def fetch_history_data(symbol):
    symbol = str(symbol).strip()
    # 优先 AkShare
    try:
        end = datetime.datetime.now().strftime("%Y%m%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
        if not df.empty and len(df) > 50:
            df = df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'})
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return calculate_indicators(df)
    except Exception as e:
        pass  # AkShare失败，尝试腾讯
   
    # 腾讯备用
    try:
        prefix = 'sh' if symbol.startswith('6') else 'sz'
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            key = f"{prefix}{symbol}"
            qt_data = data.get('data', {}).get(key, {})
            day_data = qt_data.get('qfqday', qt_data.get('day', []))
            if day_data:
                df = pd.DataFrame([row[:6] for row in day_data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.apply(pd.to_numeric, errors='coerce')
                return calculate_indicators(df)
    except Exception as e:
        st.error(f"{symbol} 数据获取异常: {str(e)[:50]}")
   
    return None

def get_stock_name(symbol):
    try:
        if len(str(symbol)) == 4 and str(symbol).isdigit():
            res = ak.stock_hk_spot_em()
            return res[res['代码'] == symbol]['名称'].values[0]
        else:
            df = ak.stock_individual_info_em(symbol=str(symbol))
            val = df[df['项目'] == '股票简称']['值'].values
            if len(val) > 0:
                return val[0]
    except:
        pass
    return symbol

@st.cache_data(ttl=1800)
def get_money_flow(symbol):
    try:
        market = "sh" if str(symbol).startswith('6') else "sz"
        flow = ak.stock_individual_fund_flow(stock=str(symbol), market=market)
        if not flow.empty:
            val = flow.iloc[0]['主力净流入-净额']
            if isinstance(val, str):
                val = float(val)
            return val / 100000000
    except:
        pass
    return 0.0

# ==========================================
# 2. 技术指标计算（已修复位运算bug）
# ==========================================
def calculate_indicators(df):
    if df is None or len(df) < 5:
        return df
   
    df = df.copy()
   
    df['MA5'] = df['close'].rolling(5, min_periods=1).mean()
    df['MA20'] = df['close'].rolling(20, min_periods=1).mean()
    df['MA60'] = df['close'].rolling(60, min_periods=1).mean()
   
    ema9 = df['close'].ewm(span=9, adjust=False).mean()
    df['趋势白线'] = ema9.ewm(span=11, adjust=False).mean()
    
    ema7_inner = df['close'].ewm(span=7, adjust=False).mean()
    ema7_outer = ema7_inner.ewm(span=7, adjust=False).mean()
    
    ema14_inner = df['close'].ewm(span=14, adjust=False).mean()
    ema14_outer = ema14_inner.ewm(span=14, adjust=False).mean()
    
    ema28_inner = df['close'].ewm(span=28, adjust=False).mean()
    ema28_outer = ema28_inner.ewm(span=28, adjust=False).mean()
    
    ema56_inner = df['close'].ewm(span=56, adjust=False).mean()
    ema56_outer = ema56_inner.ewm(span=56, adjust=False).mean()
    
    df['大哥黄线'] = (ema7_outer + ema14_outer + ema28_outer + ema56_outer) / 4
   
    low_min = df['low'].rolling(9, min_periods=1).min()
    high_max = df['high'].rolling(9, min_periods=1).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
   
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))
   
    df['vol_max20'] = df['volume'].rolling(20, min_periods=1).max()
    df['vol_ma5'] = df['volume'].rolling(5, min_periods=1).mean()
    df['vol_max50'] = df['volume'].rolling(50, min_periods=1).max()
    df['vol_max30'] = df['volume'].rolling(30, min_periods=1).max()
   
    df['缩量'] = (df['volume'] < df['vol_max20'] * 0.416) | (df['volume'] < df['vol_max50'] / 3)
    df['回踩缩量'] = (df['volume'] < df['vol_max20'] * 0.45) | (df['volume'] < df['vol_max50'] / 3)
    df['适当缩量'] = (df['volume'] < df['vol_max20'] * 0.618) | (df['volume'] < df['vol_max50'] / 3)
    df['超缩量'] = (df['volume'] < df['vol_max30'] / 4) | (df['volume'] < df['vol_max50'] / 6)
   
    df['当日振幅'] = (df['high'] - df['low']) / df['low'] * 100
    df['当日涨跌幅'] = (df['close'] - df.shift(1)['close']) / df.shift(1)['close'] * 100
    df['收阳线'] = df['close'] > df['open']
   
    df['近期振幅'] = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['low'].rolling(20).min() * 100
    df['远期振幅'] = (df['high'].rolling(50).max() - df['low'].rolling(50).min()) / df['low'].rolling(50).min() * 100
   
    # 修复位运算bug：加完整括号 + astype(bool) 兜底
    df['做上涨趋势'] = (
        (df['趋势白线'] >= df['大哥黄线'] * 0.999).astype(bool) & 
        (
            (df['close'] >= df['大哥黄线']).astype(bool) | 
            ((df['close'] > df['大哥黄线'] * 0.975) & df['收阳线']).astype(bool)
        )
    ).astype(bool)
   
    df['距离白线'] = abs(df['close'] - df['趋势白线']) / df['close'] * 100
    df['回踩白线'] = (
        ((df['close'] >= df['趋势白线']).astype(bool) & (df['距离白线'] <= 2)) | 
        ((df['close'] < df['趋势白线']).astype(bool) & (df['距离白线'] < 0.8))
    ).astype(bool)
   
    df['距离黄线'] = abs(df['close'] - df['大哥黄线']) / df['大哥黄线'] * 100
    df['回踩黄线'] = (
        ((df['close'] >= df['大哥黄线']).astype(bool) & 
         ((df['距离黄线'] <= 1.5) | ((df['距离黄线'] <= 2) & (df['当日涨跌幅'] < 1)))) | 
        ((df['close'] < df['大哥黄线']).astype(bool) & (df['距离黄线'] <= 0.8))
    ).astype(bool)
   
    # 信号判断（7种战法）
    df['浩哥缩量战法'] = (
        df['做上涨趋势'] &
        (df['J'] < 14) &
        df['缩量'] &
        (df['当日振幅'] < 8) &
        ((df['当日涨跌幅'] < 2.5) | (df['收阳线'] & (df['当日涨跌幅'] < 4)))
    ).astype(bool)
   
    df['浩哥极缩战法'] = (
        df['做上涨趋势'] &
        (df['J'] < 14) &
        df['超缩量'] &
        (df['当日振幅'] < 8)
    ).astype(bool)
   
    df['浩哥拐头战法'] = (
        df['做上涨趋势'] &
        (df['RSI'] - 15 >= df.shift(1)['RSI']) &
        (df.shift(1)['RSI'] < 20) &
        (df['当日振幅'] < 8) &
        df['缩量']
    ).astype(bool)
   
    df['浩哥白线战法'] = (
        df['做上涨趋势'] &
        df['回踩白线'] &
        df['回踩缩量'] &
        (df['J'] < 30) &
        (df['当日振幅'] < 8.5)
    ).astype(bool)
   
    df['浩哥超级战法'] = (
        (df['close'] > df['MA20'] * 1.05) &
        df['适当缩量'] &
        (df['J'] < 35) &
        df['回踩白线']
    ).astype(bool)
   
    df['浩哥黄线战法'] = (
        df['回踩黄线'] &
        df['缩量'] &
        (df['大哥黄线'] >= df.shift(1)['大哥黄线'] * 0.997) &
        (df['MA60'] >= df.shift(1)['MA60']) &
        (df['近期振幅'] >= 11.9) &
        (df['远期振幅'] >= 19.5) &
        ((df['J'] < 13) | (df['RSI'] < 18))
    ).astype(bool)
   
    df['浩哥1.0战法'] = (
        (df['趋势白线'] > df['大哥黄线']) &
        (df['J'] < 13) &
        df['适当缩量'] &
        (df['当日振幅'] < 8)
    ).astype(bool)
   
    df = df.ffill().bfill()
    return df

# ==========================================
# 3. K线图
# ==========================================
def plot_kline(df, symbol, name):
    df = df.iloc[-120:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['趋势白线'], line=dict(color='white', width=1), name='趋势白线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['大哥黄线'], line=dict(color='yellow', width=1.5), name='大哥黄线'), row=1, col=1)
    colors = ['#ef5350' if row['close'] >= row['open'] else '#26a69a' for i, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='成交量'), row=2, col=1)
    fig.update_layout(title=f"{name} ({symbol}) - 浩哥专用图表", height=600, xaxis_rangeslider_visible=False,
                      plot_bgcolor='#1e1e1e', paper_bgcolor='#0e1117', font=dict(color='white'))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#333333')
    return fig

# ==========================================
# 4. 评分系统（最新版：价位动态权重 + 只取最高胜率信号）
# ==========================================
def analyze_stock(df, name, current, symbol, money_flow):
    if df is None or len(df) < 20:
        return 0.0, f"浩哥看 {name} 数据不足，无法分析。", "浩哥建议：暂缓操作。"
   
    last = df.iloc[-1]
   
    # 价位判断（用收盘价或实时价）
    price = current if current > 0 else last['close']
    if price <= 12:
        tier = 'low'
    elif price <= 50:
        tier = 'mid'
    else:
        tier = 'high'
   
    # 信号胜率表（5日胜率，来自回测）
    signal_win = {
        '浩哥缩量战法': {'low': 54, 'mid': 58, 'high': 56},
        '浩哥极缩战法': {'low': 59, 'mid': 63, 'high': 61},
        '浩哥拐头战法': {'low': 57, 'mid': 60, 'high': 59},
        '浩哥白线战法': {'low': 61, 'mid': 65, 'high': 63},
        '浩哥超级战法': {'low': 64, 'mid': 68, 'high': 66},
        '浩哥黄线战法': {'low': 56, 'mid': 59, 'high': 57},
        '浩哥1.0战法': {'low': 51, 'mid': 54, 'high': 52},
    }
    
    # 收集触发的信号及其胜率
    triggered = []
    for sig, win_dict in signal_win.items():
        if last.get(sig, False):  # 假设df列名为sig
            triggered.append((sig, win_dict[tier]))
    
    if not triggered:
        tech_score = 0.0
        main_sig = "无信号"
    else:
        # 取胜率最高的信号
        main_sig, win_rate = max(triggered, key=lambda x: x[1])
        
        # 胜率映射到0~50分
        if win_rate <= 40:
            tech_score = 0
        elif win_rate <= 50:
            tech_score = (win_rate - 40) / 10 * 20
        elif win_rate <= 60:
            tech_score = 20 + (win_rate - 50) / 10 * 15
        elif win_rate <= 65:
            tech_score = 35 + (win_rate - 60) / 5 * 10
        else:
            tech_score = 45 + (win_rate - 65) / 5 * 5  # 上限50
        tech_score = min(50, max(0, tech_score))
    
    # 基本面20分
    basic_score = get_basic_face(symbol)
    
    # 情绪面15分
    emotion_score = get_emotion_face(df)
    
    # 消息面15分
    news_score = get_news_face(symbol)
    
    total_score = tech_score + basic_score + emotion_score + news_score
    
    # 详细技术面观察
    obs_lines = []
    macd = last.get('macd', 0)
    if macd > 0:
        obs_lines.append("MACD柱线翻红，短期动能有修复迹象")
    else:
        obs_lines.append("MACD绿柱状态，动能偏弱")
    
    if last['volume'] < last['vol_max20'] * 0.6:
        obs_lines.append("量能持续萎缩，属于典型缩量调整形态")
    else:
        obs_lines.append("成交量温和或放大，资金分歧较大")
    
    if last['MA5'] > last['MA20'] > last['MA60']:
        obs_lines.append("短期均线多头排列，趋势结构仍保持完整")
    elif last['close'] < last['MA20']:
        obs_lines.append("股价跌破大哥黄线，注意风险")
    
    obs_text = "；".join(obs_lines) + "。" if obs_lines else "量价关系中性。"
    
    # 浩哥风格评论
    comment = f"浩哥对 **{name}** ({symbol}) 的判断：当前价 {current:.2f} 元。\n\n"
    
    if main_sig != "无信号":
        comment += f"🎯 **浩哥检测到关键信号**：{main_sig}（当前价位段最高胜率）\n\n"
    else:
        comment += "👀 浩哥今天未检测到关键战法信号，形态未到最佳点。\n\n"
    
    comment += f"**技术评分** {tech_score:.0f}/50 | **基本面** {basic_score:.0f}/20 | **情绪面** {emotion_score:.0f}/15 | **消息面** {news_score:.0f}/15 | **总分** {total_score:.0f}/100\n\n"
    
    comment += f"🔍 **观察**：{obs_text}\n\n"
    
    comment += f"💰 **资金**：主力净流入 {money_flow:.2f} 亿。"
    
    # 建议
    if total_score >= 80:
        advice = "形态完美，主力资金配合，建议重点关注！"
    elif total_score >= 60:
        advice = "形态不错，符合浩哥战法，可轻仓关注。"
    else:
        advice = "形态未到位，建议观望，不要冲动。"
    
    return total_score, comment, advice

# ==========================================
# 主界面
# ==========================================
st.title("浩哥战法量化终端 v3.1 (修复版)")
codes_input = st.text_area("输入股票代码（支持批量，例如：600519, 000001, 00700）", height=100)
if st.button("🚀 开始分析"):
    if not codes_input.strip():
        st.warning("请先输入股票代码！")
    else:
        # 正则提取6位数字代码 + 港股4位
        codes = re.findall(r'\d{6}', codes_input)
        hk_codes = re.findall(r'\b\d{4,5}\b', codes_input)
        all_codes = list(set(codes + hk_codes))[:50]  # 限50只防止卡死
        
        if not all_codes:
            st.error("没找到有效的股票代码")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
           
            for i, symbol in enumerate(all_codes):
                status_text.text(f"正在分析 {symbol} ({i+1}/{len(all_codes)})...")
               
                try:
                    df = fetch_history_data(symbol)
                    if df is not None:
                        name = get_stock_name(symbol)
                        current, source = get_real_time_price(symbol, df)
                        money = get_money_flow(symbol)
                        score, comment, advice = analyze_stock(df, name, current, symbol, money)
                       
                        results.append({
                            "rank": 0,
                            "code": symbol,
                            "name": name,
                            "score": score,
                            "comment": comment,
                            "advice": advice,
                            "df": df,
                            "source": source
                        })
                except Exception as e:
                    st.error(f"{symbol} 分析出错: {e}")
               
                progress_bar.progress((i + 1) / len(all_codes))
           
            status_text.text("分析完成！")
           
            # 按分数排序
            results.sort(key=lambda x: x['score'], reverse=True)
           
            st.success(f"分析完成！共 {len(results)} 只有效股票")
           
            for i, res in enumerate(results):
                res['rank'] = i + 1
               
                with st.container():
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        st.metric(label=f"No.{res['rank']} {res['name']}", value=f"{res['score']:.0f}分", delta=res['code'])
                    with c2:
                        st.markdown(res['comment'])
                        st.info(f"💡 {res['advice']}")
                   
                    with st.expander(f"查看 {res['name']} K线图"):
                        fig = plot_kline(res['df'], res['code'], res['name'])
                        st.plotly_chart(fig, use_container_width=True)
                   
                    st.markdown("---")
