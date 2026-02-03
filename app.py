def fetch_stock_history(symbol):
    if not symbol.isdigit() or len(symbol) != 6:
        return None
        
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # 检查 HTTP 错误
        raw = response.json()
        
        # 安全提取数据：腾讯接口有时 data 是 { "sh600519": { "qfqday": [...] } }
        data_section = raw.get('data', {})
        stock_data = data_section.get(f"{prefix}{symbol}", {}) \
                                 .get('qfqday', []) \
                      or data_section.get(f"{prefix}{symbol}", {}) \
                                 .get('day', [])
        
        if not stock_data:
            # 尝试另一种可能的嵌套结构（有些版本返回在 'data' 下还有一次嵌套）
            inner = data_section.get('', {})  # 注意：有时 key 是空字符串！
            if isinstance(inner, dict):
                stock_data = inner.get('qfqday', []) or inner.get('day', [])
        
        if not stock_data:
            st.warning(f"⚠️ {symbol}：接口返回空数据，可能是股票代码错误或接口限制。")
            return None

        df = pd.DataFrame(stock_data, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        for col in ['open', 'close', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        if df.empty:
            return None

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
