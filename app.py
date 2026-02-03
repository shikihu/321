def fetch_stock_history(symbol):
    if not symbol.isdigit() or len(symbol) != 6:
        return None
        
    # 函数：用 yfinance 拉 A股数据（Yahoo Finance 对香港友好）
def fetch_stock_data(symbol):
    try:
        # A股在 Yahoo 用 .SS (上证) 或 .SZ (深证)
        ticker = f"{symbol}.SS" if symbol.startswith('6') else f"{symbol}.SZ"
        stock = yf.Ticker(ticker)
        
        # 实时信息
        info = stock.info
        name = info.get('shortName', '未知股票')
        current = info.get('currentPrice', info.get('regularMarketPrice', 0.0))
        
        # 历史 K线（最近 365 天）
        hist = stock.history(period="1y", interval="1d")
        if hist.empty:
            raise ValueError("无历史数据")
        
        df = hist[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']

        # 计算技术指标
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
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        df['k'] = rsv.ewm(span=3, adjust=False).mean()
        df['d'] = df['k'].ewm(span=3, adjust=False).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']

        # BBI
        ma3 = df['close'].rolling(3).mean()
        ma6 = df['close'].rolling(6).mean()
        ma12 = df['close'].rolling(12).mean()
        ma24 = df['close'].rolling(24).mean()
        df['bbi'] = (ma3 + ma6 + ma12 + ma24) / 4

        return df

    except Exception as e:
        st.error(f"❌ 获取 {symbol} 数据时出错: {str(e)}")
        return None
