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
# 1. 数据服务（稳定版，支持A股/港股）
# ==========================================
@st.cache_data(ttl=300)
def get_real_time_price(symbol, df=None):
    symbol = str(symbol)
    prefix = 'sh' if symbol.startswith(('6', '9')) else 'sz'
    if len(symbol) == 4 and symbol.isdigit(): prefix = 'hk'  # 港股
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
        r = requests.get(url, headers=headers, timeout=3)
        if 'var hq_str_' in r.text:
            parts = r.text.split('"')[1].split(',')
            if len(parts) > 3 and float(parts[3]) > 0:
                return float(parts[3]), "实时价"
    except:
        pass
    if df is not None and not df.empty:
        return df['close'].iloc[-1], "(盘后/最近收盘价)"
    return 0.0, "无数据"

@st.cache_data(ttl=3600)
def fetch_history_data(symbol):
    symbol = str(symbol)
    is_hk = len(symbol) == 4 and symbol.isdigit()
    
    # A股腾讯接口
    if not is_hk:
        prefix = 'sh' if symbol.startswith('6') else 'sz'
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            r = requests.get(url, headers=headers, timeout=5)
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
        except:
            pass
    
    # AkShare 兜底（A股/港股）
    try:
        if is_hk:
            df = ak.stock_hk_daily(symbol=symbol, adjust="qfq")
        else:
            end = datetime.datetime.now().strftime("%Y%m%d")
            start = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y%m%d")
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
        
        if not df.empty:
            rename_map = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return calculate_indicators(df)
    except:
        pass
    
    return None

def get_stock_name(symbol):
    try:
        if len(str(symbol)) == 4 and str(symbol).isdigit():
            return ak.stock_hk_spot_em().query(f"代码 == '{symbol}'")['名称'].values[0]
        else:
            df = ak.stock_individual_info_em(symbol=str(symbol))
            return df[df['项目'] == '股票简称']['值'].values[0]
    except:
        return symbol

@st.cache_data(ttl=1800)
def get_money_flow(symbol):
    try:
        market = "sh" if str(symbol).startswith('6') else "sz"
        flow = ak.stock_individual_fund_flow(stock=str(symbol), market=market)
        if not flow.empty:
            return flow.iloc[0]['主力净流入-净额'] / 100000000
    except:
        pass
    return 0.0

# ==========================================
# 2. 技术指标计算（已修正趋势白线 & 大哥黄线）
# ==========================================
def calculate_indicators(df):
    if df is None or len(df) < 5:
        return df
   
    df = df.copy()
   
    # 均线
    df['MA5'] = df['close'].rolling(5, min_periods=1).mean()
    df['MA20'] = df['close'].rolling(20, min_periods=1).mean()
    df['MA60'] = df['close'].rolling(60, min_periods=1).mean()
   
    # 趋势白线: EMA(EMA(C,9),11)
    ema9 = df['close'].ewm(span=9, adjust=False).mean()
    df['趋势白线'] = ema9.ewm(span=11, adjust=False).mean()
    
    # 大哥黄线: (EMA(EMA(C,7),7) + EMA(EMA(C,14),14) + EMA(EMA(C,28),28) + EMA(EMA(C,56),56))/4
    ema7_inner = df['close'].ewm(span=7, adjust=False).mean()
    ema7_outer = ema7_inner.ewm(span=7, adjust=False).mean()
    
    ema14_inner = df['close'].ewm(span=14, adjust=False).mean()
    ema14_outer = ema14_inner.ewm(span=14, adjust=False).mean()
    
    ema28_inner = df['close'].ewm(span=28, adjust=False).mean()
    ema28_outer = ema28_inner.ewm(span=28, adjust=False).mean()
    
    ema56_inner = df['close'].ewm(span=56, adjust=False).mean()
    ema56_outer = ema56_inner.ewm(span=56, adjust=False).mean()
    
    df['大哥黄线'] = (ema7_outer + ema14_outer + ema28_outer + ema56_outer) / 4
   
    # KDJ
    low_min = df['low'].rolling(9, min_periods=1).min()
    high_max = df['high'].rolling(9, min_periods=1).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
   
    # 量能
    df['vol_max20'] = df['volume'].rolling(20, min_periods=1).max()
    df['vol_ma5'] = df['volume'].rolling(5, min_periods=1).mean()
   
    df = df.ffill().bfill()
    return df

# ==========================================
# 3. K线图（大块显示）
# ==========================================
def plot_kline(df, symbol, name):
    df = df.iloc[-120:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='white', width=1), name='白线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='yellow', width=1.5), name='大哥线'), row=1, col=1)
    colors = ['red' if row['open'] < row['close'] else 'green' for i, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='成交量'), row=2, col=1)
    fig.update_layout(title=f"{name} ({symbol}) - 浩哥专用图表", height=600, xaxis_rangeslider_visible=True, 
                      plot_bgcolor='#1e1e1e', paper_bgcolor='#0e1117', font=dict(color='white'))
    return fig

# ==========================================
# 4. 评分系统（详细技术面观察 + 浩哥风格评论）
# ==========================================
def analyze_stock(df, name, current, symbol, money_flow):
    if df is None or len(df) < 20:
        return 0.0, f"浩哥看 {name} 数据不足，无法分析。", "浩哥建议：暂缓操作。", "#888"
   
    last = df.iloc[-1]
   
    triggered = []
    tech_score = 0.0
   
    j_val = last['J']
    dist_white = abs(last['close'] - last['趋势白线']) / last['close'] * 100 if last['close'] > 0 else 999
    dist_yellow = abs(last['close'] - last['大哥黄线']) / last['大哥黄线'] * 100 if last['大哥黄线'] > 0 else 999
   
    if j_val < 0 and last['volume'] < last['vol_max20'] * 0.6:
        triggered.append("浩哥缩量战法")
        tech_score += 20.0
    if dist_white < 2.0 and last['close'] > last['大哥黄线']:
        triggered.append("浩哥白线战法")
        tech_score += 25.0
   
    hao_score = 0.0
    if money_flow > 0.5: hao_score = 15.0
    elif money_flow > 0: hao_score = 5.0
   
    total_score = min(100, tech_score + hao_score)
   
    # 详细技术面观察
    obs_lines = []
    macd = last['macd'] if 'macd' in last else 0
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
   
    # 浩哥风格评论（详细、专业）
    comment = f"浩哥对 {name} 的综合判断：当前价 {current:.2f} 元。\n\n"
   
    if triggered:
        comment += f"浩哥检测到关键信号：{' + '.join(triggered)}\n\n"
    else:
        comment += "浩哥今天未检测到关键信号，形态未到最佳点。\n\n"
   
    comment += f"【技术面评分】{tech_score:.1f}/70 【浩哥评分】{hao_score:.1f}/30 【浩哥综合打分】{total_score:.1f}/100\n\n"
   
    comment += f"技术面观察：{obs_text}\n\n"
   
    comment += f"资金面：主力净流入 {money_flow:.2f} 亿。浩哥认为当前风险大于机会，形态和情绪均未到位，短期不宜重仓。"
   
    if total_score >= 80:
        advice = "浩哥喊单：机会显著，重仓干！"
    elif total_score >= 60:
        advice = "浩哥建议：形态还行，可以轻仓试试。"
    else:
        advice = "浩哥建议：暂时回避，保护本金，等更清晰信号。"
   
    color = "#d32f2f" if total_score >= 80 else "#ff5722" if total_score >= 60 else "#757575"
   
    return total_score, comment, advice, color

# ==========================================
# 主界面（简洁版 + 批量支持 + 排序）
# ==========================================
st.set_page_config(page_title="浩哥战法", layout="wide")
st.title("浩哥战法量化终端 v3.0")

codes_input = st.text_area("输入股票代码（逗号或换行分隔，支持批量，最多300只）", height=150)
if st.button("开始分析"):
    codes = re.findall(r'\d{6}', codes_input)
    codes = list(set(codes))[:300]  # 去重 + 限300只
    if not codes:
        st.error("没找到有效代码")
    else:
        results = []
        progress = st.progress(0)
        status = st.empty()
       
        for i, symbol in enumerate(codes):
            status.text(f"分析中 {i+1}/{len(codes)}: {symbol}")
            df = fetch_history_data(symbol)
            if df is not None:
                name = get_stock_name(symbol)
                current, source = get_real_time_price(symbol, df)
                money = get_money_flow(symbol)
                score, comment, advice, color = analyze_stock(df, name, current, symbol, money)
                results.append({
                    "rank": 0,
                    "code": symbol,
                    "name": name,
                    "score": score,
                    "comment": comment,
                    "advice": advice,
                    "color": color,
                    "df": df,
                    "source": source
                })
            progress.progress((i + 1) / len(codes))
       
        # 排序
        results.sort(key=lambda x: x['score'], reverse=True)
        for i, res in enumerate(results):
            res['rank'] = i + 1
       
        st.success(f"分析完成！共 {len(results)} 只票")
       
        for res in results:
            c1, c2 = st.columns([1, 5])
            with c1:
                st.markdown(f"**第 {res['rank']} 名**")
                st.markdown(f"<h2 style='color: {res['color']}'>{res['score']:.1f}/100</h2>", unsafe_allow_html=True)
            with c2:
                st.markdown(res['comment'])
                st.markdown(f"**浩哥建议**：{res['advice']}")
                st.caption(f"价格来源：{res['source']}")
            
            with st.expander("K线图"):
                fig = plot_kline(res['df'], res['code'], res['name'])
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
