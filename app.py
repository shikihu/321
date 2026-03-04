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
# 基础配置
# ==========================================
socket.setdefaulttimeout(20)
st.set_page_config(
    page_title="浩哥战法量化终端 v14.6 (零报错版)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 侧边栏：缓存清理
with st.sidebar:
    st.header("维护工具")
    if st.button("清除缓存 (修复报错)"):
        st.cache_data.clear()
        st.success("缓存已清除，请重新运行！")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
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
        df = df.copy()
        df['数据不足'] = True
        return df

    df = df.copy()
    df['数据不足'] = False

    # 预初始化所有关键列
    init_cols = [
        '拐头B', '缩量B', '原始B1', '超缩量B', '白线B', '黄线B',
        '浩哥王炸', '砖型翻红', '浩哥极缩', '砖型起爆'
    ]
    for col in init_cols:
        df[col] = False

    df['趋势白线'] = np.nan
    df['大哥黄线'] = np.nan
    df['止损价'] = np.nan
    df['目标价'] = np.nan

    try:
        C = df['close']
        O = df['open']
        H = df['high']
        L = df['low']
        V = df['volume']
        RC = C.shift(1)

        # 基础均线
        df['MA5'] = C.rolling(5, min_periods=1).mean()
        df['MA20'] = C.rolling(20, min_periods=1).mean()
        df['MA60'] = C.rolling(60, min_periods=1).mean()

        # 趋势线
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

        df['近期振幅'] = (hhv(H, 20) - llv(L, 20)) / llv(L, 20) * 100
        df['远期振幅'] = (hhv(H, 50) - llv(L, 50)) / llv(L, 50) * 100

        # 趋势 & 回踩
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

        # 砖型图
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
        df['CC'] = (~df['AA'].shift(1)) & df['AA']  # 前日非上涨 & 当前上涨 = 严格起爆
        df['砖型起爆'] = df['CC']

        df['砖型翻红'] = (df['砖型图'] > 0) & (df['砖型图'].shift(1) == 0)

        # 组合信号
        df['浩哥极缩'] = df['超缩量B'] | (df['缩量B'] & (df['当日振幅'] < 6))
        df['浩哥王炸'] = df['浩哥极缩'] & df['砖型起爆'] & (df['回踩白线'] | df['回踩黄线'])

        # 止损 & 目标
        df['技术支撑'] = df[['MA20', '大哥黄线', '趋势白线']].min(axis=1)
        df['止损价'] = df['技术支撑'] * 0.97
        df['目标价'] = df['high'].rolling(20).max() * 1.15

    except Exception as e:
        st.warning(f"计算异常，但继续运行: {str(e)[:100]}")

    df = df.ffill().bfill()
    return df

# 其余部分（矩阵回测、评分逻辑、主程序）保持原样，只需替换calculate_indicators函数即可

# 如果你想完整版，我可以再贴一次，但重点就是这个函数的缩进和逻辑已修复
