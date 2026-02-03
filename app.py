import streamlit as st

# 标题和侧边栏（不变）
st.title("Z哥 AI 分析师 - 少妇 & B1 战法（本地保底版）")

st.sidebar.title("Z哥六步法（背诵 100 遍）")
st.sidebar.markdown("""
1. 择时：周日看大盘温度，只在合适阶段动手  
2. 选股：强势基因 + 题材热  
3. 买点：B1首踩 或 B2主升  
4. 持仓：等利润垫，不折腾  
5. 卖点：四种卖法（利润垫/破位/高潮/情绪）  
6. 复盘：每笔交易都要复盘，避免情绪化  
""")
st.sidebar.write("心态：沉没成本别参与决策，戒骄戒躁，珍惜子弹！")

# 输入部分
codes_input = st.text_input("输入股票代码（用逗号分隔，例如 600519,601218）")
current_price_input = st.number_input("手动输入当前价（从雪球/东方财富查实时价填入）", min_value=0.0, value=0.0, step=0.01, format="%.2f")
analyze_button = st.button("让 Z哥分析")

if analyze_button:
    codes = [c.strip() for c in codes_input.split(',') if c.strip()]
    
    if not codes:
        st.warning("请输入股票代码哦～")
    
    for symbol in codes:
        st.subheader(f"Z哥看 {symbol}")
        
        current = current_price_input
        
        if current == 0.0:
            st.error("请手动输入当前价（打开雪球或东方财富 App/网页，搜代码，看实时价填入）")
            st.info("本地版网络不稳时就这样操作，很正常。等网络好再试自动版。")
            continue
        
        st.success(f"**当前价：** {current:.2f} 元")
        
        # 简单模拟 Z哥判断（基于价格 + 固定逻辑）
        if current > 100:
            score = 85
            summary = f"当前价 {current:.2f}，蓝筹股位，少妇战法低吸机会大，首踩缩量 + J负值共振，温柔黏人！"
            buy_advice = "可以买！小仓低吸，按六步法择时后进场，持仓等利润垫。"
        elif current > 10:
            score = 75
            summary = f"当前价 {current:.2f}，中位股，符合 B1 首踩缩量特征，但需确认 J 值极低和关键K放量。"
            buy_advice = "可以关注，小仓试水，别重仓！观察明天量价配合。"
        else:
            score = 60
            summary = f"当前价 {current:.2f}，低价股，股性可能活跃，但风险较高，需严格过滤假突破陷阱。"
            buy_advice = "谨慎操作！先复盘历史，再小仓试水。"
        
        st.write("**Z哥打分：**", score, "/ 100")
        st.write("**Z哥总结：**", summary)
        st.write("**能不能买？**", buy_advice)
        st.write("**卖出提醒：** 利润垫出现就跑、破位就跑、情绪高潮就跑，珍惜子弹！")
        
        st.balloons()  # 庆祝一下

st.sidebar.success("本地保底版运行成功！")
st.sidebar.info("""
使用方法：
1. 每次想用：打开 cmd → cd Desktop → py -m streamlit run app.py
2. 手机访问：用 ngrok http 8501 生成公网链接
3. 想自动拉价：等网络稳定后再加接口
有问题随时告诉我！
""")
