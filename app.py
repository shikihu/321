import streamlit as st
import pandas as pd
import requests
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import time
import socket

# ==========================================
# 1. 基础配置
# ==========================================
# 设置超时，防止死循环
socket.setdefaulttimeout(15)
st.set_page_config(page_title="浩哥战法量化终端 v5.0 (全市场快照版)", layout="wide")

# ==========================================
# 2. 核心数据引擎 (全市场快照)
# ==========================================
@st.cache_data(ttl=60)
def get_market_snapshot():
    """
    一次性拉取全市场所有股票的实时数据 (PE, PB, 现价, 资金, 换手)
    解决了单只股票请求失败的问题。
    """
    try:
        # 使用 akshare 的实时行情接口
        df = ak.stock_zh_a_spot_em()
        
        # 重命名列以方便索引
        # 通常列名：代码, 名称, 最新价, 涨跌幅, ..., 换手率, 市盈率-动态, 市净率, 主力净流入
        df = df.rename(columns={
            '代码': 'code',
            '名称': 'name',
            '最新价': 'price',
            '市盈率-动态': 'pe',
            '市净率': 'pb',
            '换手率': 'turnover',
            '主力净流入': 'money_flow'
        })
        
        # 将代码作为索引，方便极速查询
        df = df.set_index('code')
        
        # 简单清洗数据，非数字转为 0
        cols_to_fix = ['pe', 'pb', 'turnover', 'money_flow', 'price']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        return df
    except Exception as e:
        st.error(f"全市场数据拉取失败，请检查 akshare 是否更新: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_history_data(symbol):
    """
    获取K线历史数据 (腾讯接口)
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
            # 兼容不同返回格式
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

def get_news_score(symbol):
    """消息面评分 (轻量级)"""
    try:
        # 简单请求，失败即跳过
        news = ak.stock_news_em(symbol=str(symbol))
        if news is not None and not news.empty:
            titles = "".join(news.head(10)['标题'].astype(str).tolist())
            pos = sum(titles.count(w) for w in ['增长','利好','突破','涨停','回购'])
            neg = sum(titles.count(w) for w in ['下跌','利空','亏损','减持','被查'])
            score = (pos - neg) * 2
            return min(15, max(0, score + 5)) # 基础分5分
    except:
        pass
    return 5

# ==========================================
# 3. 指标计算 & 战法逻辑
# ==========================================
def calculate_indicators(df):
    if df is None or len(df) < 5: return df
    df = df.copy()
    
    # 均线
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    # 浩哥均线
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
    
    # 成交量
    df['vol_max20'] = df['volume'].rolling(20).max()
    df['vol_max50'] = df['volume'].rolling(50).max()
    df['缩量'] = (df['volume'] < df['vol_max20'] * 0.45)
    
    # 战法判定
    df['当日振幅'] = (df['high'] - df['low']) / df['low'] * 100
    
    # 上涨趋势定义
    df['做上涨趋势'] = (df['趋势白线'] >= df['大哥黄线'] * 0.999)
    
    dist = abs(df['close'] - df['趋势白线']) / df['close'] * 100
    df['回踩白线'] = (dist <= 2.5)
    
    # 信号
    df['浩哥缩量战法'] = df['做上涨趋势'] & (df['J'] < 20) & df['缩量']
    df['浩哥超级战法'] = (df['close'] > df['MA20']) & df['缩量'] & (df['J'] < 40) & df['回踩白线']
    
    df = df.ffill().bfill()
    return df

# ==========================================
# 4. 分析逻辑 (结合快照数据)
# ==========================================
def analyze_stock(code, snapshot_df, history_df):
    # 1. 从全市场快照中获取数据 (决不为0)
    if code in snapshot_df.index:
        info = snapshot_df.loc[code]
        name = str(info['name'])
        price = float(info['price'])
        pe = float(info['pe'])
        pb = float(info['pb'])
        turnover = float(info['turnover'])
        money = float(info['money_flow']) / 100000000 # 转亿
    else:
        # 如果快照里都没有，说明可能是停牌或代码错误
        return None 
    
    # 2. 计算基本面 (满分20)
    # 茅台逻辑：PE~28, PB~8。 PB较高，得分会少，这是算法逻辑决定的，不是bug
    basic_score = 0
    if pe > 0 and pe < 35: basic_score += 10 # 放宽PE标准
    if pb > 0 and pb < 4: basic_score += 10
    elif pb >= 4 and pe < 20: basic_score += 5 # 只有低估值高PB才给分
    
    # 3. 计算情绪面 (满分15)
    emotion_score = 0
    if turnover > 0.5 and turnover < 5: emotion_score += 15 # 健康换手
    elif turnover >= 5 and turnover < 10: emotion_score += 10 # 活跃
    elif turnover >= 10: emotion_score += 5 # 过热
    else: emotion_score += 5 # 极低换手
    
    # 4. 消息面 (满分15)
    news_score = get_news_score(code)
    
    # 5. 技术面 (满分50)
    tech_score = 0
    sig_msg = "无信号"
    advice = "观望"
    
    if history_df is not None:
        last = history_df.iloc[-1]
        
        # 简单有效的加分逻辑
        if last['做上涨趋势']: tech_score += 15
        if last['macd'] > 0: tech_score += 10
        if last['浩哥缩量战法']: 
            tech_score += 25
            sig_msg = "浩哥缩量战法"
        elif last['浩哥超级战法']: 
            tech_score += 20
            sig_msg = "浩哥超级战法"
        elif last['J'] < 10:
            tech_score += 15
            sig_msg = "超卖反弹预期"
            
    # 汇总
    total = basic_score + emotion_score + news_score + tech_score
    if total > 75: advice = "重点关注"
    elif total > 60: advice = "适当关注"
    
    comment = f"**{name}** ({code}) 现价:{price}\n"
    comment += f"PE(动):{pe:.1f} | PB:{pb:.2f} | 换手:{turnover}%\n"
    comment += f"主力资金: {money:.2f}亿\n"
    comment += f"信号: {sig_msg}\n"
    comment += f"评分细节: 基本{basic_score}+情绪{emotion_score}+消息{news_score}+技术{tech_score}"
    
    return {
        'code': code, 'name': name, 'score': total, 
        'comment': comment, 'advice': advice, 
        'history': history_df
    }

# ==========================================
# 5. 主程序界面
# ==========================================
st.title("浩哥战法量化终端 v5.0 (全市场快照版)")
st.caption("🚀 核心升级：不再逐个查询基本面，直接加载全市场实时数据，彻底解决茅台等股票0分问题。")

codes_input = st.text_area("输入代码 (支持批量，例如: 600519, 002446)", height=100)

if st.button("开始分析"):
    codes = re.findall(r'\d{6}', codes_input)
    codes = list(set(codes))[:50]
    
    if not codes:
        st.warning("请输入代码")
    else:
        # 1. 先拉取全市场数据 (只做一次)
        with st.spinner("正在加载全市场实时数据 (约2-3秒)..."):
            market_snapshot = get_market_snapshot()
            
        if market_snapshot.empty:
            st.error("行情接口连接失败，请稍后重试。")
        else:
            results = []
            progress = st.progress(0)
            
            for i, code in enumerate(codes):
                # 2. 拉K线
                hist_df = fetch_history_data(code)
                
                # 3. 综合计算
                res = analyze_stock(code, market_snapshot, hist_df)
                
                if res:
                    results.append(res)
                else:
                    st.warning(f"{code} 停牌或未找到实时数据")
                    
                progress.progress((i+1)/len(codes))
                time.sleep(0.1) # 极快模式，只需微小间隔
            
            results.sort(key=lambda x: x['score'], reverse=True)
            
            st.success(f"分析完成！共 {len(results)} 只股票")
            
            for res in results:
                with st.expander(f"{res['name']} ({res['code']}) - 总分 {res['score']}", expanded=True):
                    c1, c2 = st.columns([3,1])
                    with c1: st.markdown(res['comment'])
                    with c2: 
                        if res['score'] > 70: st.success(res['advice'])
                        else: st.info(res['advice'])
                    
                    if res['history'] is not None:
                        df_p = res['history'].iloc[-120:]
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
                        fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['open'], high=df_p['high'],
                                                   low=df_p['low'], close=df_p['close'], name='K线'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['趋势白线'], line=dict(color='white'), name='白线'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['大哥黄线'], line=dict(color='yellow'), name='黄线'), row=1, col=1)
                        fig.add_trace(go.Bar(x=df_p.index, y=df_p['volume'], name='成交量'), row=2, col=1)
                        fig.update_layout(height=450, margin=dict(l=0,r=0,t=0,b=0), plot_bgcolor='#111')
                        st.plotly_chart(fig, use_container_width=True)
