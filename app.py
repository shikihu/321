import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import re
import time
import socket

# ==========================================
# 基础配置
# ==========================================
socket.setdefaulttimeout(20)
st.set_page_config(
    page_title="浩哥战法量化终端 v14.5 (无懈可击版)",
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
# 数据引擎
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
# 核心算法（严格对齐浩哥原意）
# ==========================================
def sma(series, n, m=1):
    return series.ewm(alpha=m/n, adjust=False).mean()

def hhv(series, n):
    return series.rolling(n).max()

def llv(series, n):
    return series.rolling(n).min()

def calculate_indicators(df):
    if df is None or len(df) < 60:
        df['数据不足'] = True
        return df

    df = df.copy()
    df['数据不足'] = False

    # 预初始化所有关键列，防KeyError
    init_cols = [
        '拐头B', '缩量B', '原始B1', '超缩量B', '白线B', '黄线B',
        '浩哥王炸', '砖型翻红', '浩哥极缩', '收益达标', '砖型起爆',
        '趋势白线', '大哥黄线', '止损价', '目标价'
    ]
    for col in init_cols:
        df[col] = False if 'B' in col or '起爆' in col else np.nan

    try:
        C, O, H, L, V = df['close'], df['open'], df['high'], df['low'], df['volume']
        RC = C.shift(1)

        # 基础均线 & 趋势线
        df['MA5'] = C.rolling(5, min_periods=1).mean()
        df['MA20'] = C.rolling(20, min_periods=1).mean()
        df['MA60'] = C.rolling(60, min_periods=1).mean()

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

        # 量能
        vol_max20 = hhv(V, 20)
        vol_max30 = hhv(V, 30)
        vol_max50 = hhv(V, 50)
        df['缩量'] = (V < vol_max20 * 0.416) | (V < vol_max50 / 3)
        df['回踩缩量'] = (V < vol_max20 * 0.45) | (V < vol_max50 / 3)
        df['适当缩量'] = (V < vol_max20 * 0.618) | (V < vol_max50 / 3)
        df['超缩量'] = (V < vol_max30 / 4) | (V < vol_max50 / 6)

        df['当日振幅'] = (H - L) / L * 100
        df['当日涨跌幅'] = (C - RC) / RC * 100
        df['收阳线'] = C > O
        df['不是大绿棒'] = ~((C < O) & (C < RC * 0.96) & (V > V.shift(1) * 1.1))

        df['近期振幅'] = (hhv(H, 20) - llv(L, 20)) / llv(L, 20) * 100
        df['远期振幅'] = (hhv(H, 50) - llv(L, 50)) / llv(L, 50) * 100

        # 趋势 & 回踩（浩哥原意阈值）
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

        # 浩哥六大B信号（严格原意阈值）
        df['拐头B'] = df['做上涨趋势'] & (df['RSI'] - 15 >= df['RSI'].shift(1)) & \
                       (df['RSI'].shift(1) < 20) & df['缩量'] & (df['当日振幅'] < 8)

        df['缩量B'] = df['做上涨趋势'] & (df['J'] < 14) & df['缩量'] & \
                      (df['当日振幅'] < 8) & ((df['当日涨跌幅'] < 2.5) | (df['收阳线'] & (df['当日涨跌幅'] < 4)))

        df['原始B1'] = (df['趋势白线'] > df['大哥黄线']) & (df['J'] < 13) & df['适当缩量'] & (df['当日振幅'] < 8)

        df['超缩量B'] = df['做上涨趋势'] & (df['J'] < 14) & df['超缩量'] & (df['当日振幅'] < 8)

        df['白线B'] = df['做上涨趋势'] & df['回踩白线'] & df['回踩缩量'] & (df['J'] < 30) & (df['当日振幅'] < 8.5)

        df['黄线B'] = df['回踩黄线'] & df['缩量'] & \
                       (df['大哥黄线'] >= df['大哥黄线'].shift(1) * 0.997) & \
                       (df['MA60'] >= df['MA60'].shift(1)) & \
                       (df['近期振幅'] >= 11.9) & (df['远期振幅'] >= 19.5) & \
                       ((df['J'] < 13) | (df['RSI'] < 18))

        # 砖型图（严格CC判断）
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

        df['AA'] = df['砖型图'] > df['砖型图'].shift(1)
        df['BB'] = df['砖型图'] < df['砖型图'].shift(1)
        df['CC'] = (~df['AA'].shift(1)) & df['AA']  # 前日非上涨 & 当前上涨
        df['砖型起爆'] = df['CC']

        df['砖型翻红'] = (df['砖型图'] > 0) & (df['砖型图'].shift(1) == 0)

        # 组合信号
        df['浩哥极缩'] = df['超缩量B'] | (df['缩量B'] & (df['当日振幅'] < 6))
        df['浩哥王炸'] = df['浩哥极缩'] & df['砖型起爆'] & (df['回踩白线'] | df['回踩黄线'])

        # 止损 & 目标（基于技术支撑）
        df['技术支撑'] = df[['MA20', '大哥黄线', '趋势白线']].min(axis=1)
        df['止损价'] = df['技术支撑'] * 0.97
        df['目标价'] = df['high'].rolling(20).max() * 1.15  # 保守15%目标

    except Exception as e:
        st.warning(f"计算异常，但继续运行: {str(e)[:100]}")
        pass

    df = df.ffill().bfill()
    return df

# ==========================================
# 矩阵回测（样本量过滤）
# ==========================================
TIER_MATRIX = {
    'low': {'min': 0, 'max': 12, 'base_score': 50},
    'mid': {'min': 12, 'max': 50, 'base_score': 70},
    'high': {'min': 50, 'max': 9999, 'base_score': 60}
}

def perform_matrix_backtest(df, current_price):
    if '数据不足' in df.columns and df['数据不足'].iloc[-1]:
        return None, {}, ["数据不足，无法回测"]

    df_test = df.iloc[-120:-3]  # 避免未来数据泄露
    strategies = ['拐头B', '缩量B', '原始B1', '超缩量B', '白线B', '黄线B']

    tier = 'mid'
    for t_name, t_data in TIER_MATRIX.items():
        if t_data['min'] <= current_price < t_data['max']:
            tier = t_name
            break

    backtest_result = {}
    history_report = []
    for sig in strategies:
        if sig not in df_test.columns:
            continue
        triggered = df_test[df_test[sig] == True]
        count = len(triggered)
        if count < 5:  # 样本量过滤
            win_rate = 0
        else:
            wins = triggered['收益达标'].sum() if '收益达标' in triggered.columns else 0
            win_rate = (wins / count) * 100

        backtest_result[sig] = {'count': count, 'win_rate': win_rate}
        if count >= 5:
            history_report.append(f"{sig}: {count}次 (胜率{win_rate:.0f}%)")

    return tier, backtest_result, history_report

# ==========================================
# 评分逻辑（组合加分）
# ==========================================
def analyze_stock_logic(code, info, df):
    if not info or df is None or df['数据不足'].iloc[-1]:
        return {
            'code': code, 'name': code, 'score': 0,
            'comment': "数据不足或获取失败", 'advice': "跳过", 'df': None,
            'has_signal': False
        }

    last = df.iloc[-1]
    name = info.get('name', code)
    price = info['price']

    tier, bt_result, hist_report = perform_matrix_backtest(df, price)

    score = 0
    signals = []
    active_sigs = []

    # 砖型起爆（权重最高）
    if last['砖型起爆']:
        score = 88
        signals.append("🧱 砖型起爆（主升确认）")
        active_sigs.append('砖型起爆')

    # 浩哥王炸（组合王炸）
    if last['浩哥王炸']:
        score = 95
        signals.insert(0, "👑 浩哥王炸（极缩+砖型+回踩）")
        active_sigs.append('浩哥王炸')

    # 单B信号
    for sig in ['拐头B', '缩量B', '原始B1', '超缩量B', '白线B', '黄线B']:
        if last.get(sig, False):
            active_sigs.append(sig)
            base = 60 if tier == 'mid' else 50 if tier == 'low' else 55
            if sig in bt_result and bt_result[sig]['count'] >= 5:
                wr = bt_result[sig]['win_rate']
                if wr >= 65:
                    base += 20
                    signals.append(f"{sig} (历史胜率{wr:.0f}%🔥)")
                elif wr >= 55:
                    base += 10
                    signals.append(f"{sig} (历史胜率{wr:.0f}%)")
                else:
                    base -= 10
                    signals.append(f"{sig} (历史胜率{wr:.0f}%⚠️)")
            else:
                base += 5
                signals.append(f"{sig} (新信号)")
            score = max(score, base)

    # 组合加分
    if last['砖型起爆'] and any(last.get(s, False) for s in ['白线B', '黄线B', '超缩量B']):
        score += 15
        signals.append("组合共振 +15分")

    # 基本面修正（腾讯数据）
    if 0 < info['pe'] < 35: score += 5
    if info['pb'] < 2.0: score += 5

    score = min(99, max(0, score))

    advice = "观望"
    if score >= 90: advice = "S级买点（强烈推荐）"
    elif score >= 80: advice = "A级买点（重点关注）"
    elif score >= 65: advice = "B级买点（谨慎布局）"

    comment = f"**{name}** ({code}) 现价: {price:.2f}\n\n"
    comment += f"📊 **价位档**: {tier}\n"
    comment += f"📡 **触发信号**: {' + '.join(signals) if signals else '无明显信号'}\n"
    comment += f"⏳ **历史回测**: {' | '.join(hist_report) if hist_report else '样本不足'}\n"

    if not np.isnan(last['止损价']):
        comment += f"\n🛡️ **建议止损**: {last['止损价']:.2f}（技术支撑-3%）"
    if not np.isnan(last['目标价']):
        comment += f"\n🎯 **参考目标**: {last['目标价']:.2f}（20日高点+15%）"

    return {
        'code': code,
        'name': name,
        'score': score,
        'comment': comment,
        'advice': advice,
        'df': df,
        'has_signal': len(active_sigs) > 0
    }

# ==========================================
# 主程序
# ==========================================
st.title("浩哥战法量化终端 v14.5 (无懈可击版)")
st.caption("核心升级：信号严格对齐浩哥原意 + 砖型起爆精确判断 + 组合加分 + 防崩溃")

codes_input = st.text_area("请输入股票代码（逗号或换行分隔，最多50只）", height=120)
if st.button("🚀 开始矩阵扫描"):
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
            time.sleep(0.1)  # 防限流

        results.sort(key=lambda x: x['score'], reverse=True)
        st.success(f"扫描完成！共 {len(results)} 只有效票")

        for res in results:
            prefix = "👑 " if res['score'] >= 90 else "🔥 " if res['score'] >= 80 else ""
            with st.expander(f"{prefix}{res['name']} ({res['code']}) - {res['score']:.0f}分", expanded=res['score'] >= 80):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(res['comment'])
                with c2:
                    if res['score'] >= 90:
                        st.error(res['advice'])
                    elif res['score'] >= 80:
                        st.success(res['advice'])
                    else:
                        st.info(res['advice'])

                if res['df'] is not None and len(res['df']) > 20:
                    df_p = res['df'].iloc[-100:]
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.25, 0.20])

                    # K线 + 趋势线
                    fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['open'], high=df_p['high'], low=df_p['low'], close=df_p['close'], name='K线'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['趋势白线'], line=dict(color='white', width=1.2), name='趋势白线'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['大哥黄线'], line=dict(color='yellow', width=1.5), name='大哥黄线'), row=1, col=1)

                    # 砖型图
                    brick_vals = df_p['砖型图'].fillna(0)
                    brick_colors = []
                    for i in range(len(brick_vals)):
                        if i == 0:
                            brick_colors.append('gray')
                        elif brick_vals[i] > 0 and brick_vals[i] >= brick_vals[i-1]:
                            brick_colors.append('#ff3333')  # 红持
                        else:
                            brick_colors.append('#33ff33')  # 绿空

                    fig.add_trace(go.Bar(x=df_p.index, y=brick_vals, marker_color=brick_colors, name='浩哥砖型图'), row=2, col=1)

                    # 标记起爆点
                    if '砖型起爆' in df_p.columns:
                       起爆 = df_p[df_p['砖型起爆']]
                        if not 起爆.empty:
                            fig.add_trace(go.Scatter(x=起爆.index, y=起爆['砖型图']*1.1, mode='markers', marker=dict(symbol='triangle-up', size=12, color='gold'), name='砖型起爆'), row=2, col=1)

                    # 成交量（极缩变蓝）
                    vol_colors = ['#00aaff' if r['浩哥极缩'] else 'gray' for _, r in df_p.iterrows()]
                    fig.add_trace(go.Bar(x=df_p.index, y=df_p['volume'], marker_color=vol_colors, name='成交量'), row=3, col=1)

                    fig.update_layout(
                        height=700,
                        margin=dict(l=0,r=0,t=30,b=0),
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(color='#d1d4dc'),
                        xaxis_rangeslider_visible=False,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
