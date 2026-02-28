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
st.set_page_config(
    page_title="浩哥战法量化终端 v7.2 (智能点评版)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 核心数据引擎 (腾讯直连 - 稳)
# ==========================================
def get_realtime_data(symbol):
    symbol = str(symbol).strip()
    prefix = 'sh' if symbol.startswith(('6', '9')) else 'sz'
    code = f"{prefix}{symbol}"
    try:
        url = f"http://qt.gtimg.cn/q={code}"
        r = requests.get(url, timeout=3)
        r.encoding = 'gbk'
        text = r.text
        if f'v_{code}="' in text:
            data_str = text.split('"')[1]
            parts = data_str.split('~')
            if len(parts) > 45:
                return {
                    'name': parts[1],
                    'price': float(parts[3]),
                    'turnover': float(parts[38]) if parts[38] else 0,
                    'pe': float(parts[39]) if parts[39] else 0,
                    'pb': float(parts[46]) if parts[46] else 0,
                    'change': float(parts[32]) if parts[32] else 0 # 涨跌幅
                }
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600)
def fetch_kline_data(symbol):
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
    except Exception:
        pass
    return None

# ==========================================
# 3. 指标计算 (浩哥战法核心)
# ==========================================
def calculate_indicators(df):
    if df is None or len(df) < 5: return df
    df = df.copy()
    
    # 均线
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    # 浩哥双线
    ema9 = df['close'].ewm(span=9, adjust=False).mean()
    df['趋势白线'] = ema9.ewm(span=11, adjust=False).mean()
    
    ema_vals = [df['close'].ewm(span=x, adjust=False).mean().ewm(span=x, adjust=False).mean() for x in [7,14,28,56]]
    df['大哥黄线'] = sum(ema_vals) / 4
    
    # KDJ & RSI
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(com=13).mean() / down.ewm(com=13).mean()
    df['RSI'] = 100 - (100 / (1 + rs))

    # 量能 (极缩 = 20日最大量 * 0.25)
    df['vol_max20'] = df['volume'].rolling(20).max()
    df['极缩'] = (df['volume'] < df['vol_max20'] * 0.25)
    df['普通缩量'] = (df['volume'] < df['vol_max20'] * 0.45)
    
    # 回踩判定
    dist_white = abs(df['close'] - df['趋势白线']) / df['close'] * 100
    dist_yellow = abs(df['close'] - df['大哥黄线']) / df['大哥黄线'] * 100
    
    df['贴近白线'] = (dist_white < 2.0)
    df['贴近黄线'] = (dist_yellow < 2.0)
    df['回踩支撑'] = df['贴近白线'] | df['贴近黄线']
    
    # 趋势
    df['趋势向上'] = (df['close'] > df['MA60']) 
    
    # 信号生成
    df['浩哥王炸'] = df['趋势向上'] & df['极缩'] & df['回踩支撑'] & (df['J'] < 35)
    df['极缩待涨'] = df['趋势向上'] & df['极缩']
    
    df = df.ffill().bfill()
    return df

# ==========================================
# 4. 智能文案生成系统 (核心升级点)
# ==========================================
def generate_smart_comment(info, last):
    """
    根据各项指标，生成独一无二的分析文案
    """
    sentences = []
    
    # 1. 量能分析
    if last['极缩']:
        sentences.append("👀 **量能状态**：成交量已缩至极致（地量），主力锁仓迹象明显，变盘在即。")
    elif last['普通缩量']:
        sentences.append("👀 **量能状态**：量能温和萎缩，市场分歧减小。")
    else:
        sentences.append("👀 **量能状态**：成交量正常，需等待进一步缩量确认。")
        
    # 2. 形态与支撑分析
    if last['贴近黄线']:
        sentences.append("🦵 **形态位置**：股价精准回踩 **'大哥黄线'**，这里通常是强支撑位，只要不破就是好买点。")
    elif last['贴近白线']:
        if last['close'] > last['趋势白线']:
            sentences.append("🦵 **形态位置**：回踩 **'趋势白线'** 不破，属于强势调整（空中加油），上方空间依然存在。")
        else:
            sentences.append("🦵 **形态位置**：股价在白线附近震荡，正在考验短期支撑。")
    elif last['close'] < last['MA60']:
        sentences.append("⚠️ **形态位置**：目前处于均线下方，趋势偏弱，属于左侧博弈，风险较高。")
    else:
        sentences.append("🦵 **形态位置**：股价运行在均线上方，趋势保持良好。")
        
    # 3. 指标分析 (KDJ/RSI)
    if last['J'] < 0:
        sentences.append("📈 **指标信号**：KDJ的J值已进入负值区，存在强烈的 **超卖反弹** 需求。")
    elif last['J'] < 20:
        sentences.append("📈 **指标信号**：J值处于低位，随时可能金叉拐头向上。")
    
    # 4. 基本面与价位分析
    pe = info['pe']
    price = info['price']
    
    if price > 50 and pe > 60:
        sentences.append("💰 **价值评估**：高价高估值标的，属于纯情绪博弈，只适合短线快进快出。")
    elif price < 10 and pe < 20:
        sentences.append("💰 **价值评估**：低价绩优股，安全边际较高，适合潜伏。")
    elif info['pb'] < 1.5:
        sentences.append("💰 **价值评估**：市净率极低，属于 **股权财政/核心资产** 范畴，具备防守属性。")
        
    # 5. 总结性话术
    if last['浩哥王炸']:
        summary = "🔥 **浩哥结论**：完美符合 **'极缩+回踩'** 王炸模型！B1买点特征显著，值得重点出击！"
    elif last['极缩待涨']:
        summary = "💎 **浩哥结论**：极致缩量通常是底部特征，建议列入自选，等待第一根放量阳线确认。"
    else:
        summary = "📝 **浩哥结论**：形态尚可，但未到最佳击球点，建议继续观察。"
        
    return "\n\n".join(sentences + [summary])

def analyze_stock_logic(code, info, df):
    if not info or df is None: return None
    
    last = df.iloc[-1]
    
    # 评分逻辑
    score = 0
    if last['浩哥王炸']: score = 90
    elif last['极缩待涨']: score = 75
    else: score = 60
    
    # 微调分数
    if last['贴近黄线']: score += 5
    if last['J'] < 20: score += 5
    if info['pe'] > 0 and info['pe'] < 30: score += 5
    score = min(99, score)
    
    # 建议标签
    advice = "观望"
    if score >= 85: advice = "B1 买点 (重点)"
    elif score >= 70: advice = "适当关注"
    
    # 生成智能点评
    smart_comment = generate_smart_comment(info, last)
    
    header = f"**{info['name']}** ({code}) 现价: **{info['price']}** (涨幅 {info['change']}%)"
    
    return {
        'code': code, 
        'name': info['name'], 
        'score': score, 
        'header': header,
        'comment': smart_comment, # 这里是生成的长文案
        'advice': advice, 
        'df': df, 
        'is_king': last['浩哥王炸']
    }

# ==========================================
# 5. 主程序界面
# ==========================================
st.title("浩哥战法量化终端 v7.2 (智能点评版)")
st.caption("🚀 特性：移除止损位显示，针对每只股票生成独一无二的浩哥风格点评。")

codes_input = st.text_area("请输入股票代码", height=100)

if st.button("🚀 开始智能分析"):
    codes = re.findall(r'\d{6}', codes_input)
    codes = list(set(codes))[:50]
    
    if not codes:
        st.warning("请先输入股票代码！")
    else:
        results = []
        bar = st.progress(0)
        
        for i, code in enumerate(codes):
            info = get_realtime_data(code)
            if info:
                df = fetch_kline_data(code)
                res = analyze_stock_logic(code, info, df)
                if res: results.append(res)
            bar.progress((i+1)/len(codes))
            time.sleep(0.05)
            
        results.sort(key=lambda x: (x['is_king'], x['score']), reverse=True)
        
        st.success(f"分析完成！共 {len(results)} 只标的")
        
        for res in results:
            prefix = "👑 " if res['is_king'] else ""
            with st.expander(f"{prefix}{res['header']} - {res['score']:.0f}分", expanded=res['is_king']):
                c1, c2 = st.columns([3, 1])
                with c1:
                    # 显示智能点评
                    st.markdown(res['comment'])
                with c2:
                    if res['score'] >= 85: st.error(res['advice'])
                    elif res['score'] >= 70: st.success(res['advice'])
                    else: st.info(res['advice'])
                
                if res['df'] is not None:
                    df_p = res['df'].iloc[-100:]
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                    
                    fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['open'], high=df_p['high'],
                                               low=df_p['low'], close=df_p['close'], name='K线'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['趋势白线'], line=dict(color='white', width=1), name='白线'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['大哥黄线'], line=dict(color='yellow', width=1), name='黄线'), row=1, col=1)
                    
                    # 极缩显示蓝色，王炸显示紫色
                    colors = ['#9c27b0' if r['浩哥王炸'] else '#2196f3' if r['极缩'] else '#ef5350' if r['close']>=r['open'] else '#26a69a' for _,r in df_p.iterrows()]
                    fig.add_trace(go.Bar(x=df_p.index, y=df_p['volume'], marker_color=colors, name='成交量'), row=2, col=1)
                    
                    fig.update_layout(height=450, margin=dict(l=0,r=0,t=0,b=0), plot_bgcolor='#131722', paper_bgcolor='#131722', font=dict(color='#d1d4dc'), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
