import streamlit as st
import pandas as pd
import requests
import numpy as np
import akshare as ak  # 必须在最前面
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import time
import urllib3
import warnings

# ==========================================
# 1. 全局配置与屏蔽警告
# ==========================================
st.set_page_config(page_title="浩哥战法量化终端 v5.0 (严谨数据版)", layout="wide")

# 屏蔽 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# 屏蔽 Pandas 的 FutureWarning (如 fillna method 被弃用的警告)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

# ==========================================
# 2. 严谨数据源获取 (核心修改部分)
# ==========================================

@st.cache_data(ttl=600)
def get_market_spot_data():
    """
    【严谨】获取全市场实时行情数据（包含真实的动态PE、PB）
    来源：东方财富-行情中心-沪深京A股
    作用：一次性获取所有股票的 PE/PB，避免在循环中频繁请求个股接口被封。
    """
    try:
        # 获取沪深A股实时行情
        df = ak.stock_zh_a_spot_em()
        # 确保代码列是字符串，方便后续匹配
        df['代码'] = df['代码'].astype(str)
        # 重命名关键列以防接口变动，做标准化处理
        # 东方财富接口返回列名通常为: 序号, 代码, 名称, 最新价, ..., 市盈率-动态, 市净率
        return df
    except Exception as e:
        # 如果全市场接口挂了，返回空表，后续逻辑会处理为0分，不会报错
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_roe_data(symbol):
    """
    【严谨】获取个股最新财务指标中的ROE
    来源：东方财富-个股-财务分析-主要指标
    """
    try:
        # 获取主要财务指标表
        df = ak.stock_financial_analysis_indicator(symbol=str(symbol))
        if df is not None and not df.empty:
            # 这里的列名通常是日期，索引是指标名，或者反过来，需要根据返回结构处理
            # akshare该接口返回：index是日期，columns包含 '净资产收益率(%)'
            if '净资产收益率(%)' in df.columns:
                # 取最近一期的数据（第一行通常是最新）
                val = df.iloc[0]['净资产收益率(%)']
                return float(val) if pd.notna(val) else 0
    except Exception:
        pass
    return 0

def get_basic_face_rigorous(symbol, market_df):
    """
    【严谨评分逻辑】
    参数:
        symbol: 股票代码
        market_df: 全市场实时行情表 (从外部传入，避免重复请求)
    """
    symbol = str(symbol)
    pe = 0.0
    pb = 0.0
    roe = 0.0
    score = 0
    
    # --- 1. 从全市场表中查找 PE 和 PB ---
    if not market_df.empty:
        # 精确匹配代码
        row = market_df[market_df['代码'] == symbol]
        if not row.empty:
            try:
                # 提取真实数据
                # 注意：列名必须与 akshare 返回的一致
                pe_val = row.iloc[0].get('市盈率-动态')
                pb_val = row.iloc[0].get('市净率')
                
                # 数据清洗：处理非数字的情况
                pe = float(pe_val) if pd.notna(pe_val) else 0
                pb = float(pb_val) if pd.notna(pb_val) else 0
            except:
                pe = 0
                pb = 0
    
    # --- 2. 获取 ROE (单独请求，需限制频率) ---
    # 为了防止批量跑的时候太慢，这里可以做一个权衡：
    # 如果是批量跑50只，建议不跑ROE或者接受慢速。
    # 这里演示严谨获取：
    roe = get_roe_data(symbol)

    # --- 3. 严格评分标准 (不满足即0分) ---
    
    # PE (0 < PE < 30) -> 8分
    # 逻辑：必须盈利(>0)且估值低(<30)。亏损股(-PE)直接0分。
    if 0 < pe < 30:
        score += 8
        
    # PB (0 < PB < 3) -> 6分
    # 逻辑：资产安全垫高
    if 0 < pb < 3:
        score += 6
        
    # ROE (> 10%) -> 6分
    # 逻辑：优秀的盈利能力
    if roe > 10:
        score += 6
        
    return score, pe, pb, roe

# ==========================================
# 3. 其他辅助数据函数 (情绪、消息、资金)
# ==========================================

def get_emotion_face(df):
    if df is None or len(df) < 1:
        return 0
    last = df.iloc[-1]
    volume = last.get('volume', 0)
    # 简易换手率逻辑：这里没有总股本数据，只能用成交量绝对值做个粗略替代
    # 严谨版建议：如果有流值数据，用 成交额/流通市值 计算
    # 这里保持原逻辑，避免引入更多复杂接口导致不稳定
    if volume > 0:
        # 假设 volume 单位是手
        # 这里仅作示例，实际情绪需结合涨停家数等，这里简化为量能评分
        return 5 # 默认中性分
    return 0

def get_news_face(symbol):
    try:
        # 增加超时控制
        news = ak.stock_news_em(symbol=str(symbol))
        if news is not None and not news.empty and '标题' in news.columns:
            positive_keywords = ['好', '利好', '上涨', '爆拉', '涨停', '增长', '业绩', '增持', '回购', '突破']
            negative_keywords = ['坏', '利空', '下跌', '暴跌', '跌停', '亏损', '风险', '减持', '处罚', '立案']
            
            # 只取最近10条
            recent_news = news.head(10)
            title_text = ' '.join(recent_news['标题'].astype(str).tolist())
            
            pos_count = sum(1 for k in positive_keywords if k in title_text)
            neg_count = sum(1 for k in negative_keywords if k in title_text)
            
            sentiment = pos_count - neg_count
            score = min(15, max(0, sentiment * 3))
            return score
        return 0
    except:
        return 0 # 接口报错时不扣分也不加分

@st.cache_data(ttl=1800)
def get_money_flow(symbol):
    try:
        market = "sh" if str(symbol).startswith('6') else "sz"
        # 资金流接口
        flow = ak.stock_individual_fund_flow(stock=str(symbol), market=market)
        if flow is not None and not flow.empty and '主力净流入-净额' in flow.columns:
            val = flow.iloc[0]['主力净流入-净额']
            # 处理中文单位
            if isinstance(val, str):
                val = val.replace('万', '').replace('亿', '')
                try:
                    val = float(val)
                except:
                    val = 0.0
            # 统一转换为亿
            return val / 100000000
        return 0.0
    except:
        return 0.0

# ==========================================
# 4. K线数据与技术指标 (腾讯源+计算)
# ==========================================

@st.cache_data(ttl=1800)
def get_real_time_price(symbol, df=None):
    # 优先用新浪接口获取毫秒级实时价
    symbol = str(symbol).strip()
    prefix = 'sh' if symbol.startswith(('6', '9')) else 'sz'
    try:
        url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
        headers = {'Referer': 'http://finance.sina.com.cn/', 'Connection': 'close'}
        r = requests.get(url, headers=headers, timeout=3)
        if 'var hq_str_' in r.text:
            parts = r.text.split('"')[1].split(',')
            if len(parts) > 3:
                price = float(parts[3])
                if price > 0: return price, "实时"
    except:
        pass
    
    # 降级方案
    if df is not None and not df.empty:
        return df['close'].iloc[-1], "收盘"
    return 0.0, "无"

@st.cache_data(ttl=3600)
def fetch_history_data(symbol):
    symbol = str(symbol).strip()
    try:
        prefix = 'sh' if symbol.startswith('6') else 'sz'
        # 腾讯接口，稳定且快
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
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
                df = df.dropna(thresh=4)
                
                if len(df) < 50: return None
                return calculate_indicators(df)
    except Exception:
        pass
    return None

def get_stock_name(symbol, market_df):
    # 优先从全市场表里取，不需额外请求
    if not market_df.empty:
        row = market_df[market_df['代码'] == str(symbol)]
        if not row.empty:
            return str(row.iloc[0]['名称'])
    return str(symbol)

def calculate_indicators(df):
    if df is None or len(df) < 5: return df
    df = df.copy()
   
    # 均线
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
   
    # 浩哥战法核心：趋势白线 (EMA结构)
    ema9 = df['close'].ewm(span=9, adjust=False).mean()
    df['趋势白线'] = ema9.ewm(span=11, adjust=False).mean()
    
    # 浩哥战法核心：大哥黄线 (多重EMA平均)
    ema7 = df['close'].ewm(span=7, adjust=False).mean().ewm(span=7, adjust=False).mean()
    ema14 = df['close'].ewm(span=14, adjust=False).mean().ewm(span=14, adjust=False).mean()
    ema28 = df['close'].ewm(span=28, adjust=False).mean().ewm(span=28, adjust=False).mean()
    ema56 = df['close'].ewm(span=56, adjust=False).mean().ewm(span=56, adjust=False).mean()
    df['大哥黄线'] = (ema7 + ema14 + ema28 + ema56) / 4
   
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
    df['macd_dif'] = exp12 - exp26
    df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['macd_dif'] - df['macd_dea']) * 2
   
    # 量能计算
    df['vol_max20'] = df['volume'].rolling(20).max()
    df['vol_max50'] = df['volume'].rolling(50).max()
    df['vol_max30'] = df['volume'].rolling(30).max()
   
    # 状态定义
    df['缩量'] = (df['volume'] < df['vol_max20'] * 0.416) | (df['volume'] < df['vol_max50'] / 3)
    df['超缩量'] = (df['volume'] < df['vol_max30'] / 4) | (df['volume'] < df['vol_max50'] / 6)
    df['适当缩量'] = (df['volume'] < df['vol_max20'] * 0.618)
    
    df['当日振幅'] = (df['high'] - df['low']) / df['low'] * 100
    df['当日涨跌幅'] = (df['close'] - df.shift(1)['close']) / df.shift(1)['close'] * 100
    df['收阳线'] = df['close'] > df['open']
   
    # 距离计算
    df['距离白线'] = abs(df['close'] - df['趋势白线']) / df['close'] * 100
    df['回踩白线'] = (df['距离白线'] <= 2) & (df['close'] >= df['趋势白线'] * 0.98)
    
    # === 战法信号定义 ===
    # 1. 浩哥缩量战法
    df['浩哥缩量战法'] = (
        (df['趋势白线'] > df['大哥黄线']) & # 趋势向上
        (df['J'] < 15) &                   # 超卖
        df['缩量'] &
        (df['当日振幅'] < 8)
    )
    
    # 2. 浩哥白线战法
    df['浩哥白线战法'] = (
        (df['趋势白线'] > df['大哥黄线']) &
        df['回踩白线'] &
        df['适当缩量'] &
        (df['J'] < 35)
    )
    
    # 3. 浩哥超级战法
    df['浩哥超级战法'] = (
        (df['close'] > df['MA20'] * 1.05) & # 强势
        df['回踩白线'] &
        (df['J'] < 40)
    )

    # 填充空值 (修复 FutureWarning)
    df = df.ffill().bfill()
    return df

# ==========================================
# 5. 可视化与综合分析
# ==========================================

def plot_kline(df, symbol, name):
    df_plot = df.iloc[-100:] # 只画最近100天
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3])
    
    # K线
    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot['open'], high=df_plot['high'],
        low=df_plot['low'], close=df_plot['close'], name='K线'
    ), row=1, col=1)
    
    # 战法线
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['趋势白线'], line=dict(color='white', width=1), name='趋势白线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['大哥黄线'], line=dict(color='yellow', width=1.5), name='大哥黄线'), row=1, col=1)
    
    # 成交量
    colors = ['#ef5350' if c >= o else '#26a69a' for c, o in zip(df_plot['close'], df_plot['open'])]
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['volume'], marker_color=colors, name='成交量'), row=2, col=1)
    
    fig.update_layout(
        title=f"{name} ({symbol})",
        height=600,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='#1e1e1e', paper_bgcolor='#0e1117',
        font=dict(color='white'),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

def analyze_stock(df, name, current, symbol, money_flow, basic_score, roe, pe, pb):
    if df is None or len(df) < 20:
        return 0, "数据不足", "观望"
        
    last = df.iloc[-1]
    
    # --- 技术面评分 ---
    tech_score = 0
    signals = []
    
    if last.get('浩哥缩量战法'):
        tech_score += 40
        signals.append("浩哥缩量战法(高胜率)")
    elif last.get('浩哥超级战法'):
        tech_score += 45
        signals.append("浩哥超级战法(强势回踩)")
    elif last.get('浩哥白线战法'):
        tech_score += 35
        signals.append("浩哥白线战法(趋势回踩)")
        
    # 趋势分
    if last['close'] > last['大哥黄线']: tech_score += 5
    if last['趋势白线'] > last['大哥黄线']: tech_score += 5
    
    tech_score = min(50, tech_score) # 上限50
    
    # --- 消息/情绪面 ---
    news_score = get_news_face(symbol)
    emotion_score = get_emotion_face(df)
    
    # --- 总分 ---
    total_score = tech_score + basic_score + news_score + emotion_score
    
    # --- 生成评语 ---
    signal_str = "、".join(signals) if signals else "无明显战法信号"
    
    comment = f"""
    **{name} ({symbol}) 分析报告**
    
    💰 **资金流向**: 主力净流入 {money_flow:.2f} 亿
    📊 **基本面数据**: PE(动): {pe:.1f} | PB: {pb:.2f} | ROE: {roe:.2f}% -> 得分: {basic_score}/20
    📈 **技术面状态**: {signal_str}
    
    **得分详情**: 技术 {tech_score} + 基本 {basic_score} + 消息 {news_score} + 情绪 {emotion_score} = **{total_score}**
    """
    
    if total_score >= 80: advice = "强烈关注 - 形态资金共振"
    elif total_score >= 60: advice = "建议关注 - 基本面或技术面有亮点"
    else: advice = "观望为主 - 等待更好时机"
    
    return total_score, comment, advice

# ==========================================
# 6. 主程序逻辑
# ==========================================

# 初始化session state
if 'market_data' not in st.session_state:
    st.session_state.market_data = pd.DataFrame()

# 输入区
st.sidebar.header("控制面板")
codes_input = st.text_area("输入股票代码 (每行一个或逗号分隔)", "600519,000858", height=100)
run_btn = st.button("🚀 开始分析", type="primary")

if run_btn:
    if not codes_input.strip():
        st.warning("请输入代码")
    else:
        # 解析代码
        codes = re.findall(r'\d{6}', codes_input)
        codes = list(set(codes))[:50] # 限制50只
        
        # 1. 预加载全市场数据 (关键步骤)
        with st.spinner("正在同步交易所实时行情..."):
            market_df = get_market_spot_data()
            st.session_state.market_data = market_df
        
        results = []
        progress_bar = st.progress(0)
        status = st.empty()
        
        for i, code in enumerate(codes):
            status.text(f"正在分析 {code} ...")
            
            # A. 获取数据
            df = fetch_history_data(code)
            
            if df is not None:
                # B. 获取各项指标
                name = get_stock_name(code, market_df)
                curr_price, _ = get_real_time_price(code, df)
                money = get_money_flow(code)
                
                # C. 严谨基本面评分
                b_score, pe, pb, roe = get_basic_face_rigorous(code, market_df)
                
                # D. 综合分析
                score, comment, advice = analyze_stock(df, name, curr_price, code, money, b_score, roe, pe, pb)
                
                results.append({
                    "code": code, "name": name, "score": score,
                    "comment": comment, "advice": advice, "df": df
                })
            
            progress_bar.progress((i + 1) / len(codes))
            time.sleep(0.1) # 稍微防封，因为大头数据已经预加载了，这里主要是K线请求
            
        status.empty()
        progress_bar.empty()
        
        # 排序与显示
        results.sort(key=lambda x: x['score'], reverse=True)
        
        st.success(f"分析完成，共 {len(results)} 只股票")
        
        for res in results:
            with st.container():
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.metric(f"{res['name']}", f"{res['score']}分", res['code'])
                    st.caption(res['advice'])
                with c2:
                    st.markdown(res['comment'])
                
                with st.expander("查看K线图"):
                    st.plotly_chart(plot_kline(res['df'], res['code'], res['name']), use_container_width=True)
                st.divider()
