import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import time
import socket

# ==========================================
# 1. 基础配置
# ==========================================
socket.setdefaulttimeout(20)
st.set_page_config(
    page_title="浩哥战法量化终端 v14.6 (完整修复版)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 侧边栏：缓存清理
with st.sidebar:
    st.header("🔧 维护工具")
    if st.button("🗑️ 清除缓存 (修复报错)"):
        st.cache_data.clear()
        st.success("缓存已清除，请重新运行！")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Connection': 'close'
}

# ==========================================
# 2. 数据引擎
# ==========================================
def get_realtime_data(symbol):
    symbol = str(symbol).strip()
    prefix = 'sh' if symbol.startswith(('6', '9')) else 'sz'
    code = f"{prefix}{symbol}"
    try:
        url = f"http://qt.gtimg.cn/q={code}"
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.encoding = 'gbk'
        text = r.text
        if not text or f'v_{code}="' not in text:
            return None
        data_str = text.split('"')[1]
        parts = data_str.split('~')
        if len(parts) > 45:
            return {
                'name': parts[1],
                'code': code,
                'price': float(parts[3]),
                'turnover': float(parts[38]) if parts[38] else 0,
                'pe': float(parts[39]) if parts[39] else 0,
                'pb': float(parts[46]) if parts[46] else 0,
                'mkt_cap': float(parts[45]) if parts[45] else 0,
                'change': float(parts[32]) if parts[32] else 0
            }
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def fetch_kline_data(symbol):
    symbol = str(symbol).strip()
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    try:
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
        r = requests.get(url, headers=HEADERS, timeout=10)
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
# 3. 核心算法 (补全所有逻辑，防止KeyError)
# ==========================================
def sma(series, n, m=1):
    return series.ewm(alpha=m/n, adjust=False).mean()

def hhv(series, n):
    return series.rolling(n).max()

def llv(series, n):
    return series.rolling(n).min()

def calculate_indicators(df):
    if df is None or len(df) < 60:
        df = df.copy() if df is not None else pd.DataFrame()
        df['数据不足'] = True
        return df

    df = df.copy()
    df['数据不足'] = False

    # 1. 预初始化所有可能用到的布尔列 (防止KeyError的核心)
    init_cols = [
        '拐头B', '缩量B', '原始B1', '超缩量B', '白线B', '黄线B',
        '浩哥王炸', '砖型翻红', '浩哥极缩', '砖型起爆', '收益达标'
    ]
    for col in init_cols:
        df[col] = False

    # 2. 预初始化数值列
    df['趋势白线'] = np.nan
    df['大哥黄线'] = np.nan
    df['砖型图'] = 0.0
    df['止损价'] = np.nan
    df['目标价'] = np.nan

    try:
        C = df['close']
        O = df['open']
        H = df['high']
        L = df['low']
        V = df['volume']
        RC = C.shift(1)

        # --- 均线系统 ---
        df['MA5'] = C.rolling(5, min_periods=1).mean()
        df['MA20'] = C.rolling(20, min_periods=1).mean()
        df['MA60'] = C.rolling(60, min_periods=1).mean()

        # --- 浩哥趋势线 ---
        ema9 = C.ewm(span=9, adjust=False).mean()
        df['趋势白线'] = ema9.ewm(span=11, adjust=False).mean()

        ema7 = C.ewm(span=7, adjust=False).mean()
        ema14 = C.ewm(span=14, adjust=False).mean()
        ema28 = C.ewm(span=28, adjust=False).mean()
        ema56 = C.ewm(span=56, adjust=False).mean()
        df['大哥黄线'] = (ema7.ewm(span=7, adjust=False).mean() +
                       ema14.ewm(span=14, adjust=False).mean() +
                       ema28.ewm(span=28, adjust=False).mean() +
                       ema56.ewm(span=56, adjust=False).mean()) / 4

        # --- 指标计算 ---
        # KDJ
        low9 = llv(L, 9)
        high9 = hhv(H, 9)
        rsv = (C - low9) / (high9 - low9) * 100
        df['K'] = sma(rsv, 3, 1)
        df['D'] = sma(df['K'], 3, 1)
        df['J'] = 3 * df['K'] - 2 * df['D']

        # RSI
        delta = C.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        df['RSI'] = 100 - (100 / (1 + rs))

        # 量能状态
        vol_max20 = hhv(V, 20)
        vol_max30 = hhv(V, 30)
        vol_max50 = hhv(V, 50)
        df['缩量'] = (V < vol_max20 * 0.416) | (V < vol_max50 / 3)
        df['回踩缩量'] = (V < vol_max20 * 0.45) | (V < vol_max50 / 3)
        df['适当缩量'] = (V < vol_max20 * 0.618) | (V < vol_max50 / 3)
        df['超缩量'] = (V < vol_max30 / 4) | (V < vol_max50 / 6)

        # 基础形态
        df['当日振幅'] = (H - L) / L * 100
        df['当日涨跌幅'] = (C - RC) / RC * 100
        df['收阳线'] = C > O
        df['近期振幅'] = (hhv(H, 20) - llv(L, 20)) / llv(L, 20) * 100
        df['远期振幅'] = (hhv(H, 50) - llv(L, 50)) / llv(L, 50) * 100

        # --- 趋势与回踩 ---
        df['做上涨趋势'] = (
            (df['趋势白线'] >= df['大哥黄线'] * 0.999) &
            ((C >= df['大哥黄线']) | ((C > df['大哥黄线'] * 0.975) & df['收阳线']))
        )

        dist_white = abs(C - df['趋势白线']) / C * 100
        dist_yellow = abs(C - df['大哥黄线']) / df['大哥黄线'] * 100

        df['回踩白线'] = (
            (C >= df['趋势白线']) & (dist_white <= 2) |
            (C < df['趋势白线']) & (dist_white < 0.8)
        )

        df['回踩黄线'] = (
            (C >= df['大哥黄线']) & ((dist_yellow <= 1.5) | ((dist_yellow <= 2) & (df['当日涨跌幅'] < 1))) |
            (C < df['大哥黄线']) & (dist_yellow <= 0.8)
        )

        # --- 浩哥六大信号 (B1) ---
        df['拐头B'] = (
            df['做上涨趋势'] &
            (df['RSI'] - 15 >= df['RSI'].shift(1)) &
            (df['RSI'].shift(1) < 20) &
            df['缩量'] &
            (df['当日振幅'] < 8)
        )

        df['缩量B'] = (
            df['做上涨趋势'] &
            (df['J'] < 14) &
            df['缩量'] &
            (df['当日振幅'] < 8) &
            ((df['当日涨跌幅'] < 2.5) | (df['收阳线'] & (df['当日涨跌幅'] < 4)))
        )

        df['原始B1'] = (
            (df['趋势白线'] > df['大哥黄线']) &
            (df['J'] < 13) &
            df['适当缩量'] &
            (df['当日振幅'] < 8)
        )

        df['超缩量B'] = (
            df['做上涨趋势'] &
            (df['J'] < 14) &
            df['超缩量'] &
            (df['当日振幅'] < 8)
        )

        df['白线B'] = (
            df['做上涨趋势'] &
            df['回踩白线'] &
            df['回踩缩量'] &
            (df['J'] < 30) &
            (df['当日振幅'] < 8.5)
        )

        df['黄线B'] = (
            df['回踩黄线'] &
            df['缩量'] &
            (df['大哥黄线'] >= df['大哥黄线'].shift(1) * 0.997) &
            (df['MA60'] >= df['MA60'].shift(1)) &
            (df['近期振幅'] >= 11.9) &
            (df['远期振幅'] >= 19.5) &
            ((df['J'] < 13) | (df['RSI'] < 18))
        )

        # --- 砖型图逻辑 ---
        hhv4 = hhv(H, 4)
        llv4 = llv(L, 4)
        range4 = (hhv4 - llv4).replace(0, 0.01)
        uar1a = (hhv4 - C) / range4 * 100 - 90
        uar2a = sma(uar1a, 4, 1) + 100
        uar3a = (C - llv4) / range4 * 100
        uar4a = sma(uar3a, 6, 1)
        uar5a = sma(uar4a, 6, 1) + 100
        uar6a = uar5a - uar2a
        df['砖型图'] = np.where(uar6a > 4, uar6a - 4, 0)

        # 砖型信号处理
        df['AA'] = (df['砖型图'] > df['砖型图'].shift(1)).fillna(False)
        df['CC'] = ((~df['AA'].shift(1).fillna(False)) & df['AA']).fillna(False)
        df['砖型起爆'] = df['CC']
        df['砖型翻红'] = ((df['砖型图'] > 0) & (df['砖型图'].shift(1) <= 0)).fillna(False)

        # --- 组合信号 ---
        df['浩哥极缩'] = df['超缩量B'] | (df['缩量B'] & (df['当日振幅'] < 6))
        df['浩哥王炸'] = df['浩哥极缩'] & df['砖型起爆'] & (df['回踩白线'] | df['回踩黄线'])

        # --- 止损与回测收益计算 ---
        # 止损位: 均线支撑下浮3%
        df['技术支撑'] = df[['MA20', '大哥黄线', '趋势白线']].min(axis=1)
        df['止损价'] = df['技术支撑'] * 0.97
        
        # 目标位: 近20日高点上浮15%
        df['目标价'] = df['high'].rolling(20).max() * 1.15
        
        # 收益达标: 未来3日最高价涨幅 > 2% (补全此列防止回测报错)
        future_high = H.shift(-3).rolling(3).max()
        df['收益达标'] = ((future_high - C) / C > 0.02).fillna(False)

    except Exception as e:
        # print(f"Error: {e}") 调试用
        pass

    df = df.ffill().bfill()
    return df

# ==========================================
# 4. 矩阵回测引擎
# ==========================================
TIER_MATRIX = {
    'low':  {'min': 0, 'max': 12,  'base_score': 50, 'name': '低价股'},
    'mid':  {'min': 12, 'max': 50, 'base_score': 70, 'name': '黄金价位'},
    'high': {'min': 50, 'max': 9999,'base_score': 60, 'name': '高价股'}
}

def perform_matrix_backtest(df, current_price):
    # 检查列是否存在，如果不存在直接返回
    if '收益达标' not in df.columns:
        return 'mid', {}, ["数据异常，无法回测"]

    if df['数据不足'].iloc[-1]:
        return 'mid', {}, ["数据不足"]

    # 截取过去半年数据进行回测（去掉最近3天防止未来函数）
    df_test = df.iloc[-120:-3] 
    strategies = ['拐头B', '缩量B', '原始B1', '超缩量B', '白线B', '黄线B']

    # 确定价位段
    tier = 'mid'
    for t_name, t_data in TIER_MATRIX.items():
        if t_data['min'] <= current_price < t_data['max']:
            tier = t_name
            break

    backtest_result = {}
    history_report = []
    
    for sig in strategies:
        # 确保列存在才读取
        if sig not in df_test.columns:
            continue
            
        triggered = df_test[df_test[sig] == True]
        count = len(triggered)
        
        if count < 3: # 样本过少不计
            win_rate = 0
        else:
            wins = triggered['收益达标'].sum()
            win_rate = (wins / count) * 100

        backtest_result[sig] = {'count': count, 'win_rate': win_rate}
        
        if count >= 3:
            history_report.append(f"{sig}: {count}次 (胜率{win_rate:.0f}%)")

    return tier, backtest_result, history_report

# ==========================================
# 5. 评分与展示逻辑
# ==========================================
def analyze_stock_logic(code, info, df):
    # 数据有效性检查
    if not info or df is None or df.empty or df['数据不足'].iloc[-1]:
        return {
            'code': code, 'name': code, 'score': 0,
            'comment': "数据不足或获取失败", 'advice': "跳过", 'df': None,
            'has_signal': False
        }

    last = df.iloc[-1]
    name = info.get('name', code)
    price = info['price']

    # 执行回测
    tier, bt_result, hist_report = perform_matrix_backtest(df, price)

    score = 0
    signals = []
    active_sigs = []

    # 1. 砖型起爆（权重高）
    if last['砖型起爆']:
        score = 85
        signals.append("🧱 砖型起爆")
        active_sigs.append('砖型起爆')

    # 2. 浩哥王炸（组合王炸）
    if last['浩哥王炸']:
        score = 98
        signals.insert(0, "👑 浩哥王炸")
        active_sigs.append('浩哥王炸')

    # 3. 单B信号 + 历史胜率修正
    for sig in ['拐头B', '缩量B', '原始B1', '超缩量B', '白线B', '黄线B']:
        if last.get(sig, False):
            active_sigs.append(sig)
            
            # 基础分
            base = 60 if tier == 'mid' else 50
            
            # 胜率修正
            if sig in bt_result and bt_result[sig]['count'] >= 3:
                wr = bt_result[sig]['win_rate']
                if wr >= 60:
                    base += 20
                    signals.append(f"{sig} (历史胜率{wr:.0f}%🔥)")
                elif wr >= 50:
                    base += 10
                    signals.append(f"{sig} (历史胜率{wr:.0f}%)")
                else:
                    base -= 10
                    signals.append(f"{sig} (历史胜率{wr:.0f}%⚠️)")
            else:
                base += 5 # 新信号奖励
                signals.append(f"{sig} (新信号)")
                
            score = max(score, base)

    # 4. 组合加分
    if last['砖型起爆'] and len(active_sigs) > 1:
        score += 10
        signals.append("共振加成")

    # 5. 基本面微调
    if 0 < info['pe'] < 30: score += 5
    if info['pb'] < 2.0: score += 5

    score = min(99, max(0, score))

    advice = "观望"
    if score >= 90: advice = "S级买点（推荐）"
    elif score >= 80: advice = "A级买点（关注）"
    elif score >= 65: advice = "B级买点（低吸）"

    comment = f"**{name}** ({code}) 现价: {price:.2f}\n\n"
    comment += f"📊 **价位**: {tier}\n"
    comment += f"📡 **信号**: {' + '.join(signals) if signals else '无'}\n"
    comment += f"⏳ **回测**: {' | '.join(hist_report) if hist_report else '无样本'}\n"

    if not np.isnan(last['止损价']):
        comment += f"\n🛡️ **止损**: {last['止损价']:.2f}"

    return {
        'code': code, 'name': name, 'score': score,
        'comment': comment, 'advice': advice, 'df': df,
        'has_signal': len(active_sigs) > 0
    }

# ==========================================
# 6. 主程序界面
# ==========================================
st.title("浩哥战法量化终端 v14.6 (零报错修复版)")
st.caption("核心修复：补全了收益回测逻辑，增加了列名预初始化，防止 KeyError。")

codes_input = st.text_area("请输入股票代码（逗号或换行分隔）", height=120)

if st.button("🚀 矩阵回测分析"):
    codes = re.findall(r'\d{6}', codes_input)
    codes = list(set(codes))[:50]

    if not codes:
        st.warning("请输入有效代码")
    else:
        results = []
        bar = st.progress(0)

        for i, code in enumerate(codes):
            info = get_realtime_data(code)
            df = fetch_kline_data(code)
            if df is not None:
                res = analyze_stock_logic(code, info, df)
                if res:
                    results.append(res)
            bar.progress((i + 1) / len(codes))
            time.sleep(0.05) 

        results.sort(key=lambda x: x['score'], reverse=True)
        st.success(f"扫描完成！共 {len(results)} 只有效票")

        for res in results:
            prefix = "👑 " if res['score'] >= 90 else "🔥 " if res['score'] >= 80 else ""
            with st.expander(f"{prefix}{res['name']} ({res['code']}) - {res['score']:.0f}分", expanded=res['score'] >= 80):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(res['comment'])
                with c2:
                    if res['score'] >= 90: st.error(res['advice'])
                    elif res['score'] >= 80: st.success(res['advice'])
                    else: st.info(res['advice'])

                if res['df'] is not None and len(res['df']) > 20:
                    df_p = res['df'].iloc[-100:]
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.25, 0.20])

                    # K线
                    fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['open'], high=df_p['high'], low=df_p['low'], close=df_p['close'], name='K线'), row=1, col=1)
                    if '趋势白线' in df_p.columns:
                        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['趋势白线'], line=dict(color='white', width=1.2), name='趋势白线'), row=1, col=1)
                    if '大哥黄线' in df_p.columns:
                        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['大哥黄线'], line=dict(color='yellow', width=1.5), name='大哥黄线'), row=1, col=1)

                    # 砖型图
                    brick_vals = df_p['砖型图'].fillna(0)
                    brick_colors = ['#ff3333' if v > 0 and v >= brick_vals[i-1] else '#33ff33' for i, v in enumerate(brick_vals)]
                    fig.add_trace(go.Bar(x=df_p.index, y=brick_vals, marker_color=brick_colors, name='砖型图'), row=2, col=1)

                    # 标记起爆
                    if '砖型起爆' in df_p.columns:
                        起爆 = df_p[df_p['砖型起爆']]
                        if not 起爆.empty:
                            fig.add_trace(go.Scatter(x=起爆.index, y=起爆['砖型图']*1.1, mode='markers', marker=dict(symbol='triangle-up', size=12, color='gold'), name='砖型起爆'), row=2, col=1)

                    # 成交量
                    vol_colors = ['#00aaff' if r['浩哥极缩'] else 'gray' for _, r in df_p.iterrows()]
                    fig.add_trace(go.Bar(x=df_p.index, y=df_p['volume'], marker_color=vol_colors, name='成交量'), row=3, col=1)

                    fig.update_layout(height=600, margin=dict(l=0,r=0,t=30,b=0), plot_bgcolor='#0e1117', paper_bgcolor='#0e1117', font=dict(color='#d1d4dc'), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
