import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import numpy as np
import akshare as ak
from datetime import datetime

# ======================
# 【1】技术指标计算（必须放在最前！】
# ======================
def calculate_indicators(df):
    """计算MA20/MA60/MACD/KDJ等核心指标"""
    df = df.copy()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = ema12 - ema26
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    
    # KDJ
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    denominator = (high_max - low_min).replace(0, 1)  # 避免除零
    rsv = (df['close'] - low_min) / denominator * 100
    df['k'] = rsv.ewm(span=3, adjust=False).mean()
    df['d'] = df['k'].ewm(span=3, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    
    return df

# ======================
# 【2】数据获取函数
# ======================
def get_real_time_price(symbol):
    """优先腾讯实时价"""
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
    
    # 备用 yfinance
    try:
        ticker = f"{symbol}.SS" if symbol.startswith('6') else f"{symbol}.SZ"
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('currentPrice', info.get('regularMarketPrice', 0.0))
    except:
        return 0.0

def fetch_stock_history(symbol):
