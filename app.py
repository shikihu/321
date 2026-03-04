import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import time
import socket
import warnings
from bs4 import BeautifulSoup

# 压制Pandas未来警告和弃用警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==========================================
# 基础配置
# ==========================================
socket.setdefaulttimeout(20)

st.set_page_config(
    page_title="浩哥战法量化终端 v14.8 (信号胜率版)",
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
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Connection': 'close'
}

# ==========================================
# 多维度数据引擎（稳定+准确）
# ==========================================
def get_realtime_data(symbol):
    symbol = str(symbol).strip()
    prefix = 'sh' if symbol.startswith(('6', '9')) else 'sz'
    code = f"{prefix}{symbol}"
    try:
        # 1. 从东方财富网获取基本面数据
        url = f"https://emweb.securities.eastmoney.com/PC_HSF10/CompanySurvey/Index?type=web&code={symbol}"
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.encoding = 'utf-8'
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # 提取基本面数据
        roe = 0
        gross_margin = 0
        revenue_growth = 0
        try:
            roe_td = soup.find('td', text='净资产收益率')
            if roe_td:
                roe = float(roe_td.find_next_sibling('td').text.replace('%', ''))
            
            gross_margin_td = soup.find('td', text='毛利率')
            if gross_margin_td:
                gross_margin = float(gross_margin_td.find_next_sibling('td').text.replace('%', ''))
            
            revenue_growth_td = soup.find('td', text='营业收入同比增长')
            if revenue_growth_td:
                revenue_growth = float(revenue_growth_td.find_next_sibling('td').text.replace('%', ''))
        except Exception as e:
            st.warning(f"获取基本面数据失败 {symbol}: {str(e)[:50]}")

        # 2. 从腾讯财经获取实时行情数据
        url = f"http://qt.gtimg.cn/q={code}"
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.encoding = 'gbk'
        text = r.text
        if not text or f'v_{code}="' not in text:
            return None
        data_str = text.split('"')[1]
        parts = data_str.split('~')
        if len(parts) > 50:
            return {
                'name': parts[1],
                'code': code,
                'price': float(parts[3]) if parts[3] and parts[3] != '' else 0,
                'turnover': float(parts[38]) if parts[38] and parts[38] != '' else 0,
                'pe': float(parts[39]) if parts[39] and parts[39] != '' else 0,
                'pb': float(parts[46]) if parts[46] and parts[46] != '' else 0,
                'roe': roe,
                'gross_margin': gross_margin,
                'revenue_growth': revenue_growth,
                'mkt_cap': float(parts[45]) if parts[45] and parts[45] != '' else 0,
                'change': float(parts[32]) if parts[32] and parts[32] != '' else 0,
                'sector': parts[51] if len(parts) > 51 else '',
                'sector_code': parts[52] if len(parts) > 52 else ''
            }
    except Exception as e:
        st.warning(f"获取实时数据失败 {symbol}: {str(e)[:50]}")
    return None

@st.cache_data(ttl=3600)
def fetch_kline_data(symbol):
    symbol = str(symbol).strip()
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    try:
        # 使用前复权数据确保回测准确性，获取最近300天数据
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,300,qfq"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # 处理不同的数据结构
            if isinstance(data.get('data'), list):
                data = data['data'][0] if data['data'] else {}
            
            day_data = data.get('data', {}).get(f"{prefix}{symbol}", {}).get('qfqday', [])
            if not day_data:
                day_data = data.get('data', {}).get(f"{prefix}{symbol}", {}).get('day', [])
            if day_data and len(day_data) > 0:
                df = pd.DataFrame([row[:6] for row in day_data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                # 安全转换数值类型，处理空值
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                return calculate_indicators(df)
    except Exception as e:
        st.warning(f"获取K线数据失败 {symbol}: {str(e)[:50]}")
    return None

@st.cache_data(ttl=3600)
def get_market_data():
    try:
        # 获取上证指数最近30天数据
        url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000001,day,,,30,qfq"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # 处理不同的数据结构
            if isinstance(data.get('data'), list):
                data = data['data'][0] if data['data'] else {}
            
            day_data = data.get('data', {}).get('sh000001', {}).get('qfqday', [])
            if not day_data:
                day_data = data.get('data', {}).get('sh000001', {}).get('day', [])
            if day_data and len(day_data) > 0:
                df = pd.DataFrame([row[:6] for row in day_data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                # 计算大盘最近5天涨幅
                recent_change = df['close'].pct_change().tail(5).sum() * 100
                # 计算大盘成交量变化（最近5天vs前5天）
                recent_volume = df['volume'].tail(5).mean()
                prev_volume = df['volume'].tail(10).head(5).mean()
                volume_growth = (recent_volume - prev_volume) / prev_volume * 100 if prev_volume != 0 else 0
                return {
                    'recent_change': recent_change,
                    'volume_growth': volume_growth
                }
    except Exception as e:
        st.warning(f"获取大盘数据失败: {str(e)[:50]}")
    return None

@st.cache_data(ttl=3600)
def get_sector_data(symbol, sector_code):
    try:
        if not sector_code:
            return None
            
        # 腾讯财经的板块代码需要加concept_前缀
        if not sector_code.startswith('concept_'):
            sector_code = f'concept_{sector_code}'
            
        # 获取板块最近10天数据
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sector_code},day,,,10,qfq"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        
        data = r.json()
        # 处理腾讯财经返回的列表格式数据
        if isinstance(data.get('data'), list):
            data = data['data'][0] if data['data'] else {}
        
        # 获取板块K线数据，兼容不同的接口结构
        sector_data = data.get('data', {}).get(sector_code, {})
        day_data = sector_data.get('qfqday', [])
        if not day_data:
            day_data = sector_data.get('day', [])
        if not day_data:
            return None
        
        df = pd.DataFrame([row[:6] for row in day_data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        for col in ['open', 'close', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 计算板块最近5天涨幅
        sector_change = df['close'].pct_change().tail(5).sum() * 100
        # 计算板块资金流入（简化计算）
        capital_inflow = (df['close'].iloc[-1] - df['close'].iloc[-6]) * df['volume'].tail(5).mean() / 100000000
        
        return {
            'sector_change': sector_change,
            'capital_inflow': capital_inflow
        }
    except Exception as e:
        st.warning(f"获取板块数据失败 {symbol}: {str(e)[:100]}")
    return None

# ==========================================
# 核心算法（准确回测+精细指标）
# ==========================================
def sma(series, n, m=1):
    return series.ewm(alpha=m/n, adjust=False).mean()

def hhv(series, n):
    return series.rolling(n, min_periods=1).max()

def llv(series, n):
    return series.rolling(n, min_periods=1).min()

def calculate_indicators(df):
    if df is None or len(df) < 60:
        df = df.copy() if df is not None else pd.DataFrame()
        df['数据不足'] = True
        return df

    df = df.copy()
    df['数据不足'] = False

    # 预初始化所有关键列
    init_cols = [
        '拐头B', '缩量B', '原始B1', '超缩量B', '白线B', '黄线B',
        '浩哥王炸', '砖型翻红', '浩哥极缩', '砖型起爆', 'AA', 'CC',
        '收益达标'
    ]
    for col in init_cols:
        df[col] = False

    df['趋势白线'] = np.nan
    df['大哥黄线'] = np.nan
    df['止损价'] = np.nan
    df['目标价'] = np.nan
    df['砖型图'] = 0

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
        high9 = np.where(high9 == low9, low9 + 0.001, high9)
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
        ema_down = np.where(ema_down == 0, 0.001, ema_down)
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

        # 浩哥六大B信号
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

        # 先计算 AA
        df['AA'] = (df['砖型图'] > df['砖型图'].shift(1)).fillna(False).astype(bool)

        # 再计算 CC
        aa_shift = df['AA'].shift(1).fillna(False).astype(bool)
        df['CC'] = (~aa_shift) & df['AA']
        df['砖型起爆'] = df['CC']

        df['砖型翻红'] = (df['砖型图'] > 0) & (df['砖型图'].shift(1) <= 0)

        # 组合信号
        df['浩哥极缩'] = df['超缩量B'] | (df['缩量B'] & (df['当日振幅'] < 6))
        df['浩哥王炸'] = df['浩哥极缩'] & df['砖型起爆'] & (df['回踩白线'] | df['回踩黄线'])

        # 止损 & 目标价
        df['技术支撑'] = df[['MA20', '大哥黄线', '趋势白线']].min(axis=1)
        df['止损价'] = df['技术支撑'] * 0.97
        df['目标价'] = df['high'].rolling(20).max() * 1.15

        # 准确回测：向量化计算收益达标（提高速度）
        # 信号出现的位置
        signal_mask = df[['拐头B', '缩量B', '原始B1', '超缩量B', '白线B', '黄线B']].any(axis=1)
        # 未来10天的最高价
        future_high = df['high'].rolling(10).max().shift(-10)
        # 收益达标：未来10天最高价 >= 目标价
        df['收益达标'] = signal_mask & (future_high >= df['目标价'])

    except Exception as e:
        st.warning(f"计算异常，但继续运行: {str(e)[:100]}")

    df = df.ffill().bfill().infer_objects(copy=False)
    return df

# ==========================================
# 矩阵回测（准确+信号胜率分析）
# ==========================================
def perform_matrix_backtest(df, current_price):
    if '数据不足' in df.columns and df['数据不足'].iloc[-1]:
        return None, {}, ["数据不足，无法回测"]

    # 准确回测：使用最近300天数据，排除最近5天未完成数据
    if len(df) < 300:
        df_test = df.iloc[:-5] if len(df) > 5 else df
    else:
        df_test = df.iloc[-300:-5]
    
    if len(df_test) < 20:
        return None, {}, ["样本不足（少于20条），无法回测"]

    strategies = ['拐头B', '缩量B', '原始B1', '超缩量B', '白线B', '黄线B']

    backtest_result = {}
    history_report = []
    
    # 分析每个信号的胜率
    for sig in strategies:
        if sig not in df_test.columns:
            continue
        triggered = df_test[df_test[sig] == True]
        count = len(triggered)
        if count < 5:
            win_rate = 0
            history_report.append(f"{sig}: {count}次 (样本不足，胜率0%)")
        else:
            wins = triggered['收益达标'].sum() if '收益达标' in triggered.columns else 0
            win_rate = (wins / count) * 100
            history_report.append(f"{sig}: {count}次 (胜率{win_rate:.1f}%)")
        
        backtest_result[sig] = {'count': count, 'win_rate': win_rate}

    # 分析不同价位档的信号胜率
    price_tiers = [
        ('低价股（0-12元）', df_test[df_test['close'] < 12]),
        ('中价股（12-50元）', df_test[(df_test['close'] >= 12) & (df_test['close'] < 50)]),
        ('高价股（50元以上）', df_test[df_test['close'] >= 50])
    ]
    
    tier_report = []
    for tier_name, tier_df in price_tiers:
        if len(tier_df) < 10:
            continue
        tier_report.append(f"\n{tier_name}信号胜率：")
        for sig in strategies:
            if sig not in tier_df.columns:
                continue
            triggered = tier_df[tier_df[sig] == True]
            count = len(triggered)
            if count < 3:
                tier_report.append(f"  {sig}: {count}次 (样本不足)")
            else:
                wins = triggered['收益达标'].sum() if '收益达标' in triggered.columns else 0
                win_rate = (wins / count) * 100
                tier_report.append(f"  {sig}: {count}次 (胜率{win_rate:.1f}%)")

    history_report.extend(tier_report)
    return backtest_result, history_report

# ==========================================
# 信号胜率为主的评分逻辑
# ==========================================
def analyze_stock_logic(code, info, df, market_data, sector_data):
    if not info or df is None or df['数据不足'].iloc[-1]:
        return {
            'code': code, 'name': code, 'score': 0,
            'comment': "数据不足或获取失败", 'advice': "跳过", 'df': None,
            'has_signal': False, 'score_reason': []
        }

    last = df.iloc[-1]
    name = info.get('name', code)
    price = info['price']

    bt_result, hist_report = perform_matrix_backtest(df, price)

    score = 0
    score_reason = []
    signals = []
    active_sigs = []

    # 1. 信号胜率分（70分，核心评分项）
    signal_score = 0
    best_signal = None
    best_win_rate = 0
    for sig in ['拐头B', '缩量B', '原始B1', '超缩量B', '白线B', '黄线B']:
        if last.get(sig, False):
            active_sigs.append(sig)
            if sig in bt_result and bt_result[sig]['count'] >= 5:
                wr = bt_result[sig]['win_rate']
                if wr > best_win_rate:
                    best_win_rate = wr
                    best_signal = sig
    if best_signal:
        signal_score = (best_win_rate / 100) * 70
        score += signal_score
        score_reason.append(f"信号胜率：{best_signal}信号历史胜率{best_win_rate:.1f}%，加{signal_score:.1f}分")
    else:
        score_reason.append(f"无有效信号或样本不足，信号胜率分0分")

    # 2. 组合共振分（15分）
    resonance_score = 0
    if last['砖型起爆']:
        active_resonance = [s for s in ['白线B', '黄线B', '超缩量B'] if last.get(s, False)]
        if len(active_resonance) >= 2:
            resonance_score = 15
            score += resonance_score
            score_reason.append(f"组合共振：砖型起爆+{'+'.join(active_resonance)}信号，加15分")
        elif len(active_resonance) == 1:
            resonance_score = 10
            score += resonance_score
            score_reason.append(f"组合共振：砖型起爆+{active_resonance[0]}信号，加10分")
        else:
            resonance_score = 5
            score += resonance_score
            score_reason.append(f"组合共振：仅砖型起爆信号，加5分")

    # 3. 基本面分（5分）
    fundamental_score = 0
    if info.get('roe', 0) > 15:
        fundamental_score += 2
        score_reason.append(f"基本面：ROE{info['roe']:.1f}%>15%，加2分")
    if info.get('gross_margin', 0) > 30:
        fundamental_score += 2
        score_reason.append(f"基本面：毛利率{info['gross_margin']:.1f}%>30%，加2分")
    if info.get('revenue_growth', 0) > 20:
        fundamental_score += 1
        score_reason.append(f"基本面：营收增长率{info['revenue_growth']:.1f}%>20%，加1分")
    score += fundamental_score

    # 4. 大盘盘面分（5分）
    market_score = 0
    if market_data and market_data['recent_change'] > 0:
        market_score += 3
        score_reason.append(f"大盘盘面：最近5天上涨{market_data['recent_change']:.1f}%，加3分")
    if market_data and market_data['volume_growth'] > 10:
        market_score += 2
        score_reason.append(f"大盘盘面：最近5天成交量放大{market_data['volume_growth']:.1f}%，加2分")
    score += market_score

    # 5. 板块信息分（3分）
    sector_score = 0
    if sector_data and info.get('sector'):
        if sector_data['sector_change'] > 0:
            sector_score += 2
            score_reason.append(f"板块信息：所属{info['sector']}板块最近5天上涨{sector_data['sector_change']:.1f}%，加2分")
        if sector_data['capital_inflow'] > 0:
            sector_score += 1
            score_reason.append(f"板块信息：所属{info['sector']}板块资金流入{sector_data['capital_inflow']:.1f}亿，加1分")
    else:
        score_reason.append("板块信息：无法获取板块数据，加0分")
    score += sector_score

    # 6. 活跃资金分（2分）
    active_capital_score = 0
    if info.get('turnover', 0) > 5:
        active_capital_score += 1
        score_reason.append(f"活跃资金：换手率{info['turnover']:.1f}%>5%，加1分")
    if info.get('change', 0) > 0:
        active_capital_score += 1
        score_reason.append(f"活跃资金：今日上涨{info['change']:.1f}%，加1分")
    score += active_capital_score

    # 总分归一化到0-99分
    score = min(99, max(0, score))

    # 建议等级
    advice = "观望"
    if score >= 90:
        advice = "S级买点（强烈推荐）"
    elif score >= 80:
        advice = "A级买点（重点关注）"
    elif score >= 65:
        advice = "B级买点（谨慎布局）"

    # 详细评论
    comment = f"**{name}** ({code}) 现价: {price:.2f}\n\n"
    comment += f"📊 **总评分**: {score:.1f}分\n"
    comment += f"📡 **触发信号**: {' + '.join(active_sigs) if active_sigs else '无明显信号'}\n"
    comment += f"⏳ **历史回测**:\n"
    for reason in hist_report:
        comment += f"- {reason}\n"
    comment += f"📝 **评分理由**:\n"
    for reason in score_reason:
        comment += f"- {reason}\n"

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
        'has_signal': len(active_sigs) > 0,
        'score_reason': score_reason
    }

# ==========================================
# 主程序
# ==========================================
st.title("浩哥战法量化终端 v14.8 (信号胜率版)")
st.caption("信号胜率为主的评分系统+6种B1信号胜率分析+准确回测")

# 获取大盘数据（只获取一次，提高速度）
market_data = get_market_data()

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
            sector_data = None
            if info and info.get('sector_code'):
                sector_data = get_sector_data(code, info['sector_code'])
            
            if df is not None and info is not None:
                res = analyze_stock_logic(code, info, df, market_data, sector_data)
                if res:
                    results.append(res)
            bar.progress((i + 1) / len(codes))
            time.sleep(0.1)  # 防限流

        # 按评分排序
        results.sort(key=lambda x: x['score'], reverse=True)
        st.success(f"扫描完成！共 {len(results)} 只有效票")

        # 显示结果
        for res in results:
            prefix = "👑 " if res['score'] >= 90 else "🔥 " if res['score'] >= 80 else ""
            with st.expander(f"{prefix}{res['name']} ({res['code']}) - {res['score']:.1f}分", expanded=res['score'] >= 80):
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

                # 绘制图表
                if res['df'] is not None and len(res['df']) > 20:
                    df_p = res['df'].iloc[-100:]
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.25, 0.20])

                    # K线图
                    fig.add_trace(go.Candlestick(
                        x=df_p.index, 
                        open=df_p['open'], 
                        high=df_p['high'], 
                        low=df_p['low'], 
                        close=df_p['close'], 
                        name='K线'
                    ), row=1, col=1)
                    
                    # 趋势线
                    if '趋势白线' in df_p.columns and not df_p['趋势白线'].isna().all():
                        fig.add_trace(go.Scatter(
                            x=df_p.index, 
                            y=df_p['趋势白线'], 
                            line=dict(color='white', width=1.2), 
                            name='趋势白线'
                        ), row=1, col=1)
                    if '大哥黄线' in df_p.columns and not df_p['大哥黄线'].isna().all():
                        fig.add_trace(go.Scatter(
                            x=df_p.index, 
                            y=df_p['大哥黄线'], 
                            line=dict(color='yellow', width=1.5), 
                            name='大哥黄线'
                        ), row=1, col=1)

                    # 砖型图
                    brick_vals = df_p['砖型图'].fillna(0).values
                    brick_colors = []
                    for i in range(len(brick_vals)):
                        if i == 0:
                            brick_colors.append('gray')
                        else:
                            if brick_vals[i] > 0 and brick_vals[i] >= brick_vals[i-1]:
                                brick_colors.append('#ff3333')  # 红
                            else:
                                brick_colors.append('#33ff33')  # 绿

                    fig.add_trace(go.Bar(
                        x=df_p.index, 
                        y=brick_vals, 
                        marker_color=brick_colors, 
                        name='浩哥砖型图'
                    ), row=2, col=1)

                    # 标记起爆点
                    if '砖型起爆' in df_p.columns:
                        起爆 = df_p[df_p['砖型起爆']]
                        if not 起爆.empty:
                            fig.add_trace(go.Scatter(
                                x=起爆.index, 
                                y=起爆['砖型图']*1.1, 
                                mode='markers', 
                                marker=dict(symbol='triangle-up', size=12, color='gold'), 
                                name='砖型起爆'
                            ), row=2, col=1)

                    # 成交量
                    vol_colors = ['#00aaff' if r['浩哥极缩'] else 'gray' for _, r in df_p.iterrows()]
                    fig.add_trace(go.Bar(
                        x=df_p.index, 
                        y=df_p['volume'], 
                        marker_color=vol_colors, 
                        name='成交量'
                    ), row=3, col=1)

                    # 图表样式配置
                    fig.update_layout(
                        height=700,
                        margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(color='#d1d4dc'),
                        xaxis_rangeslider_visible=False,
                        showlegend=True
                    )
                    
                    fig.update_yaxes(range=[0, max(df_p['砖型图'].max() * 1.2, 1)], row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
