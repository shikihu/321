import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import time
import socket

# ==========================================
# 1. 基础配置 (防卡死设置)
# ==========================================
# 设置全局超时，防止网络波动卡死
socket.setdefaulttimeout(10)

# 页面设置
st.set_page_config(
    page_title="浩哥战法量化终端 v7.0 (王炸策略版)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 核心数据引擎 (腾讯直连 - 极速稳定)
# ==========================================
def get_realtime_data(symbol):
    """
    获取实时行情 (包含现价、PE、PB、市值、换手率)
    """
    symbol = str(symbol).strip()
    # 判断市场前缀
    prefix = 'sh' if symbol.startswith(('6', '9')) else 'sz'
    code = f"{prefix}{symbol}"
    
    try:
        # 腾讯极速接口
        url = f"http://qt.gtimg.cn/q={code}"
        r = requests.get(url, timeout=3)
        # 腾讯返回的是GBK编码
        r.encoding = 'gbk'
        text = r.text
        
        # 解析数据 v_sh600519="1~贵州茅台~..."
        if f'v_{code}="' in text:
            data_str = text.split('"')[1]
            parts = data_str.split('~')
            
            # 确保数据完整
            if len(parts) > 45:
                return {
                    'name': parts[1],
                    'price': float(parts[3]),
                    'turnover': float(parts[38]) if parts[38] else 0, # 换手率
                    'pe': float(parts[39]) if parts[39] else 0,       # 市盈率(动)
                    'pb': float(parts[46]) if parts[46] else 0,       # 市净率
                    'mv': float(parts[45]) if parts[45] else 0        # 总市值
                }
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600)
def fetch_kline_data(symbol):
    """
    获取日K线历史数据
    """
    symbol = str(symbol).strip()
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    try:
        # 腾讯K线接口
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
        r = requests.get(url, headers={'Connection': 'close'}, timeout=5)
        
        if r.status_code == 200:
            data = r.json()
            key = f"{prefix}{symbol}"
            
            # 兼容不同返回格式
            day_data = data.get('data', {}).get(key, {}).get('qfqday', [])
            if not day_data:
                day_data = data.get('data', {}).get(key, {}).get('day', [])
            
            if day_data:
                # 提取前6列: Date, Open, Close, High, Low, Volume
                df = pd.DataFrame([row[:6] for row in day_data], columns=['date', 'open', 'close', 'high', 'low', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.apply(pd.to_numeric, errors='coerce')
                return calculate_indicators(df)
    except Exception:
        pass
    return None

# ==========================================
# 3. 浩哥战法核心算法 (v7.0 极缩+回踩)
# ==========================================
def calculate_indicators(df):
    if df is None or len(df) < 5: return df
    df = df.copy()
    
    # --- 基础均线 ---
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    # --- 浩哥专用趋势线 ---
    # 白线: EMA9的EMA11
    ema9 = df['close'].ewm(span=9, adjust=False).mean()
    df['趋势白线'] = ema9.ewm(span=11, adjust=False).mean()
    
    # 黄线: 多周期EMA加权
    ema_vals = [df['close'].ewm(span=x, adjust=False).mean().ewm(span=x, adjust=False).mean() for x in [7,14,28,56]]
    df['大哥黄线'] = sum(ema_vals) / 4
    
    # --- 震荡指标 (KDJ & RSI) ---
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

    # --- 1. 量能判定 (王炸核心：极度缩量) ---
    # 定义：成交量小于过去20天最大量的 25% (非常严格)
    df['vol_max20'] = df['volume'].rolling(20).max()
    df['极缩'] = (df['volume'] < df['vol_max20'] * 0.25)
    # 普通缩量：45%
    df['普通缩量'] = (df['volume'] < df['vol_max20'] * 0.45)
    
    # --- 2. 回踩判定 (B1买点) ---
    # 计算收盘价距离各均线的幅度
    dist_white = abs(df['close'] - df['趋势白线']) / df['close'] * 100
    dist_yellow = abs(df['close'] - df['大哥黄线']) / df['大哥黄线'] * 100
    dist_ma20 = abs(df['close'] - df['MA20']) / df['close'] * 100
    
    # 回踩定义：距离均线 < 2%，且收盘价没有跌破均线太多(>3%视为有效跌破)
    # 这里逻辑稍微放宽，只要贴近就算回踩，具体止损由止损位控制
    df['贴近白线'] = (dist_white < 2.0)
    df['贴近黄线'] = (dist_yellow < 2.0)
    df['贴近MA20'] = (dist_ma20 < 2.0)
    df['回踩支撑'] = df['贴近白线'] | df['贴近黄线'] | df['贴近MA20']
    
    # --- 3. 拐头信号 (J值或RSI低位勾头) ---
    df['J_拐头'] = (df['J'] < 25) & (df['J'] > df.shift(1)['J'])
    df['RSI_拐头'] = (df['RSI'] < 30) & (df['RSI'] > df.shift(1)['RSI'])
    df['拐头信号'] = df['J_拐头'] | df['RSI_拐头']
    
    # --- 4. 趋势判断 ---
    # 简单的多头排列过滤
    df['趋势向上'] = (df['close'] > df['MA60']) 
    
    # --- 5. 信号生成 ---
    
    # 【浩哥王炸】：趋势向上 + 极缩 + 回踩 + J值低位
    df['浩哥王炸'] = df['趋势向上'] & df['极缩'] & df['回踩支撑'] & (df['J'] < 35)
    
    # 【极缩等待】：仅满足极缩和趋势，等待变盘
    df['极缩待涨'] = df['趋势向上'] & df['极缩']
    
    # 【普通战法】：普通缩量 + 拐头
    df['浩哥缩量'] = df['趋势向上'] & df['普通缩量'] & df['拐头信号']
    
    # --- 6. 止损位计算 (关键风控) ---
    # 取 MA20 和 黄线 中较高的一个作为支撑，下浮 3% 作为硬止损
    df['支撑位'] = df[['MA20', '大哥黄线']].max(axis=1)
    df['建议止损'] = df['支撑位'] * 0.97
    
    # 填充空值
    df = df.ffill().bfill()
    return df

# ==========================================
# 4. 综合评分与分析系统
# ==========================================
def analyze_stock_logic(code, info, df):
    if not info or df is None: return None
    
    name = info['name']
    price = info['price']
    pe = info['pe']
    pb = info['pb']
    
    last = df.iloc[-1]
    
    # 初始化
    tech_score = 0
    signals = []
    king_signal = False
    
    # --- 1. 技术面打分 (满分60) ---
    
    if last['浩哥王炸']:
        tech_score = 60 # 满分
        signals.append("👑 浩哥王炸 (极缩+回踩)")
        king_signal = True
    elif last['极缩待涨']:
        tech_score = 45
        signals.append("💎 极致缩量 (主力锁仓)")
    elif last['浩哥缩量']:
        tech_score = 35
        signals.append("🔹 普通缩量回踩")
        
    # 叠加分
    if last['拐头信号'] and not king_signal:
        tech_score += 10
        signals.append("⤴️ 指标拐头")
    if last['回踩支撑'] and not king_signal:
        tech_score += 5
        signals.append("🦶 回踩稳固")
        
    # --- 2. 价位段权重修正 (浩哥逻辑) ---
    tier_msg = ""
    multiplier = 1.0
    
    if price < 8:
        # 低价股降权，防止退市风险阴跌
        multiplier = 0.8 
        tier_msg = "⚠️ 低价风险区 (<8元)"
    elif 8 <= price <= 50:
        # 机构游资最爱舒适区
        multiplier = 1.1 
        tier_msg = "✅ 黄金交易区 (8-50元)"
    else:
        # 高价股看业绩
        if pe > 0 and pe < 40:
            multiplier = 1.05
            tier_msg = "✅ 绩优白马区"
        else:
            multiplier = 0.9
            tier_msg = "⚠️ 高价题材区 (需谨慎)"
            
    tech_score = min(70, tech_score * multiplier) # 技术分上限放宽到70
    
    # --- 3. 基本面防守 (满分20) ---
    basic_score = 0
    if pe > 0 and pe < 35: basic_score += 10
    if pb > 0 and pb < 4: basic_score += 10
    elif pb >= 4 and pe < 25: basic_score += 5 # 容忍高PB低PE
    
    # --- 4. 情绪/资金面 (满分10) ---
    # 简单用换手率辅助
    emotion_score = 5
    if 3 < info['turnover'] < 10: emotion_score = 10 # 活跃
    
    # --- 总分 ---
    total = tech_score + basic_score + emotion_score
    total = min(100, total)
    
    # 建议文案
    advice = "观望"
    if total >= 80: advice = "重点出击 (B1买点)"
    elif total >= 65: advice = "适当关注"
    
    sig_str = " + ".join(signals) if signals else "趋势暂不明朗"
    stop_loss_price = last['建议止损']
    
    # 构造返回对象
    return {
        'code': code,
        'name': name,
        'score': total,
        'price': price,
        'comment': (
            f"**{name}** ({code}) 现价: **{price}**\n\n"
            f"🎯 **核心信号**: {sig_str}\n"
            f"📊 **价位属性**: {tier_msg}\n"
            f"🛡️ **浩哥锦囊**: 建议止损位 **{stop_loss_price:.2f}元** (破位坚决离场)\n"
            f"📝 **综合评分**: {total:.0f} (技术{tech_score:.0f} + 基本{basic_score} + 情绪{emotion_score})"
        ),
        'advice': advice,
        'df': df,
        'is_king': king_signal, # 是否王炸
        'is_extreme': last['极缩'] # 是否极缩
    }

# ==========================================
# 5. 主程序界面
# ==========================================
st.title("浩哥战法量化终端 v7.0 (王炸策略版)")
st.caption("🚀 核心升级：基于浩哥最新逻辑，植入'极缩+回踩'王炸组合，增加止损位计算与价位分层。")

# 输入区
codes_input = st.text_area("请输入股票代码 (支持批量，如: 600519, 002446, 600418)", height=100)

if st.button("🚀 扫描 B1 买点"):
    # 提取代码
    codes = re.findall(r'\d{6}', codes_input)
    codes = list(set(codes))[:50] # 限制50只
    
    if not codes:
        st.warning("请先输入股票代码！")
    else:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 开始循环分析
        for i, code in enumerate(codes):
            status_text.text(f"正在分析 {code} ...")
            
            # 1. 获取实时
            info = get_realtime_data(code)
            
            if info:
                # 2. 获取K线
                df = fetch_kline_data(code)
                # 3. 综合逻辑
                res = analyze_stock_logic(code, info, df)
                if res:
                    results.append(res)
            else:
                # 获取失败跳过
                pass
                
            # 进度条
            progress_bar.progress((i + 1) / len(codes))
            # 极速模式，微小延时即可
            time.sleep(0.05)
            
        status_text.success(f"分析完成！共 {len(results)} 只标的")
        
        # 排序逻辑：王炸置顶 > 分数高低
        results.sort(key=lambda x: (x['is_king'], x['score']), reverse=True)
        
        # 展示结果
        for res in results:
            # 王炸特殊标题
            prefix = "👑 [浩哥王炸] " if res['is_king'] else ""
            
            # 展开框
            with st.expander(f"{prefix}{res['name']} ({res['code']}) - {res['score']:.0f}分", expanded=res['is_king']):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(res['comment'])
                with c2:
                    if res['score'] >= 80:
                        st.error(f"🔥 {res['advice']}") # 红色高亮
                    elif res['score'] >= 65:
                        st.success(f"✅ {res['advice']}")
                    else:
                        st.info(f"👀 {res['advice']}")
                        
                # 绘制K线图
                if res['df'] is not None:
                    df_p = res['df'].iloc[-100:] # 只看最近100天
                    
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                    
                    # 1. K线
                    fig.add_trace(go.Candlestick(
                        x=df_p.index, open=df_p['open'], high=df_p['high'],
                        low=df_p['low'], close=df_p['close'], name='K线'
                    ), row=1, col=1)
                    
                    # 2. 均线系统
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['趋势白线'], line=dict(color='white', width=1), name='白线'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['大哥黄线'], line=dict(color='yellow', width=1), name='黄线'), row=1, col=1)
                    
                    # 3. 止损线 (虚线显示)
                    fig.add_trace(go.Scatter(x=df_p.index, y=df_p['建议止损'], line=dict(color='red', width=1, dash='dot'), name='止损线'), row=1, col=1)

                    # 4. 成交量 (王炸变色)
                    # 颜色逻辑：王炸=紫色，极缩=蓝色，涨=红，跌=绿
                    colors = []
                    for idx, row in df_p.iterrows():
                        if row['浩哥王炸']:
                            colors.append('#9c27b0') # 紫色
                        elif row['极缩']:
                            colors.append('#2196f3') # 蓝色
                        elif row['close'] >= row['open']:
                            colors.append('#ef5350') # 红色
                        else:
                            colors.append('#26a69a') # 绿色
                            
                    fig.add_trace(go.Bar(x=df_p.index, y=df_p['volume'], marker_color=colors, name='成交量'), row=2, col=1)
                    
                    # 图表布局
                    fig.update_layout(
                        height=450, 
                        margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor='#131722',
                        paper_bgcolor='#131722',
                        font=dict(color='#d1d4dc'),
                        xaxis_rangeslider_visible=False
                    )
                    fig.update_xaxes(showgrid=False)
                    fig.update_yaxes(showgrid=True, gridcolor='#363c4e')
                    
                    st.plotly_chart(fig, use_container_width=True)
