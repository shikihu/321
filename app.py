def fetch_stock_history(symbol):
    if not (isinstance(symbol, str) and symbol.isdigit() and len(symbol) == 6):
        return None

    prefix = 'sh' if symbol.startswith('6') else 'sz'
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,360,qfq"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        raw = response.json()

        # 安全提取数据：支持 "" 或 "sh600519" 作为 key
        data_section = raw.get('data', {})
        stock_data = []

        # 尝试正常 key
        if f"{prefix}{symbol}" in data_section:
            inner = data_section[f"{prefix}{symbol}"]
            stock_data = inner.get('qfqday', []) or inner.get('day', [])
        # 尝试空字符串 key（常见于某些股票）
        elif "" in data_section:
            inner = data_section[""]
            if isinstance(inner, dict):
                stock_data = inner.get('qfqday', []) or inner.get('day', [])

        if not stock_data or not isinstance(stock_data, list):
            st.warning(f"⚠️ {symbol}：未获取到有效K线数据。")
            return None

        # ✅ 核心修复：只取每行的前6个元素（date, open, close, high, low, volume）
        cleaned_data = []
        for row in stock_data:
            if isinstance(row, list) and len(row) >= 6:
                # 强制取前6个，并转为字符串防止类型错乱
                cleaned_data.append([str(x) for x in row[:6]])
            # 忽略长度不足的行

        if not cleaned_data:
            st.error(f"❌ {symbol}：所有K线数据行长度不足6。")
            return None

        # 创建 DataFrame，手动指定列名
        df = pd.DataFrame(cleaned_data, columns=['date', 'open', 'close', 'high', 'low', 'volume']).copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df.set_index('date', inplace=True)

        # 转换数值列
        for col in ['open', 'close', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        if len(df) < 20:
            st.warning(f"⚠️ {symbol}：有效数据少于20天，无法分析。")
            return None

        # 计算指标
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()

        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['dif'] = ema12 - ema26
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        df['macd'] = (df['dif'] - df['dea']) * 2

        low_min = df['low'].rolling(9).min()
        high_max = df['high'].rolling(9).max()
        rsv = (df['close'] - low_min) / (high_max - low_min).replace(0, 1) * 100
        df['k'] = rsv.ewm(span=3, adjust=False).mean()
        df['d'] = df['k'].ewm(span=3, adjust=False).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']

        ma3 = df['close'].rolling(3).mean()
        ma6 = df['close'].rolling(6).mean()
        ma12 = df['close'].rolling(12).mean()
        ma24 = df['close'].rolling(24).mean()
        df['bbi'] = (ma3 + ma6 + ma12 + ma24) / 4

        return df

    except Exception as e:
        st.error(f"❌ 获取 {symbol} 数据失败: {str(e)}")
        return None
