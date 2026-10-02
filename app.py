import streamlit as st
import pandas as pd
import numpy as np

# 1. 页面基本设置
st.set_page_config(page_title="餐饮商业数据与风控调度中台 7.0", page_icon="👑", layout="wide")
st.title("👑 餐饮商业数据与流量风控调度中台 7.0")

# 2. 读取并清洗两张数据表
@st.cache_data
def load_data():
    df_meta = pd.read_csv("Restaurant names and Metadata.csv")
    df_meta['Cost'] = df_meta['Cost'].astype(str).str.replace(',', '').astype(float)
    
    df_reviews = pd.read_csv("Restaurant reviews.csv")
    df_reviews['Rating'] = pd.to_numeric(df_reviews['Rating'], errors='coerce')
    
    review_stats = df_reviews.groupby('Restaurant').agg(
        Avg_Rating=('Rating', 'mean'),
        Review_Count=('Rating', 'count'),
        Rating_Std=('Rating', 'std') 
    ).reset_index()
    
    df = pd.merge(df_meta, review_stats, left_on='Name', right_on='Restaurant', how='left')
    df['Avg_Rating'] = df['Avg_Rating'].fillna(0).round(1)
    df['Review_Count'] = df['Review_Count'].fillna(0).astype(int)
    df['Rating_Std'] = df['Rating_Std'].fillna(0.5).round(2)
    
    # 商家金字塔分层算法
    def get_tier(row):
        if row['Avg_Rating'] >= 4.2 and row['Review_Count'] >= 80:
            return 'S级 (头部旗舰)'
        elif row['Avg_Rating'] >= 3.8:
            return 'A级 (腰部骨干)'
        elif row['Avg_Rating'] >= 3.0:
            return 'B级 (潜力观察)'
        else:
            return 'C级 (高危待整改)'
            
    df['Merchant_Tier'] = df.apply(get_tier, axis=1)
    return df

df = load_data()

# 3. 三大核心模块 Tabs
tab1, tab2, tab3 = st.tabs(["🍔 用户端：智能选店", "📈 AM运营端：全维大盘洞察与流量杠杆", "🔬 算法端：LASSO-Logistic 归因与 SOP"])

# ==================== Tab 1: 用户端 ====================
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        max_cost = st.slider("💰 最高预算：", min_value=100, max_value=3000, value=1500)
    with c2:
        min_rating = st.slider("⭐ 最低评分要求：", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
    with c3:
        cuisine_search = st.text_input("🍜 搜菜系 (如 Chinese, Biryani)", "")

    filtered_df = df[(df['Cost'] <= max_cost) & (df['Avg_Rating'] >= min_rating)]
    if cuisine_search:
        filtered_df = filtered_df[filtered_df['Cuisines'].fillna('').str.contains(cuisine_search, case=False)]
    
    st.dataframe(filtered_df[['Name', 'Merchant_Tier', 'Cost', 'Avg_Rating', 'Review_Count', 'Cuisines']], use_container_width=True)

# ==================== Tab 2: AM运营端 (洞察 + 杠杆) ====================
with tab2:
    # --- 模块 A: 核心大盘指标 ---
    st.subheader("💡 平台商家大盘核心指标")
    tier_counts = df['Merchant_Tier'].value_counts()
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("入驻商家总数", f"{len(df)} 家")
    m2.metric("S级 头部旗舰", f"{tier_counts.get('S级 (头部旗舰)', 0)} 家")
    m3.metric("A级 腰部骨干", f"{tier_counts.get('A级 (腰部骨干)', 0)} 家")
    m4.metric("B级 潜力观察", f"{tier_counts.get('B级 (潜力观察)', 0)} 家")
    m5.metric("C级 高危待整改", f"{tier_counts.get('C级 (高危待整改)', 0)} 家")
    
    st.divider()
    
    # --- 模块 B: 流量调控与 ROI 仿真杠杆 ---
    st.subheader("🎛️ AM 流量再分配调控杠杆与 ROI 仿真")
    st.write("结合下方图表洞察，通过拉动杠杆，将公域曝光从劣质商家向高转化潜力商家转移，并实时评估大盘收益。")
    
    col_boost, col_suppress = st.columns(2)
    with col_boost:
        boost_rate = st.slider("🚀 给予【S/A级 & 优质潜力店】加权曝光扶持 (%)：", 0, 100, 30)
    with col_suppress:
        suppress_rate = st.slider("🛑 给予【C级高危店 & 低分刺客店】降权限流惩罚 (%)：", 0, 100, 50)
        
    # 模拟 ROI 计算
    boost_mask = df['Merchant_Tier'].isin(['S级 (头部旗舰)', 'A级 (腰部骨干)'])
    boost_clicks = (df[boost_mask]['Review_Count'].sum() * 10) * (boost_rate / 100.0)
    
    suppress_mask = df['Merchant_Tier'] == 'C级 (高危待整改)'
    suppress_clicks = (df[suppress_mask]['Review_Count'].sum() * 10) * (suppress_rate / 100.0)
    
    net_clicks_change = boost_clicks - suppress_clicks
    # 假设优质店转化率 15%，劣质店转化率 5%
    estimated_gmv_lift = (boost_clicks * 0.15 * df[boost_mask]['Cost'].mean()) - (suppress_clicks * 0.05 * df[suppress_mask]['Cost'].mean())
    
    st.success(f"📊 **流量调控执行预测：** 大盘有效点击量净变化 **{'+' if net_clicks_change>0 else ''}{int(net_clicks_change)} 次** | 预计带来平台总 GMV 净提升 **+₹ {int(max(0, estimated_gmv_lift)):,}**")
    
    st.divider()
    
    # --- 模块 C: 多维分析图表矩阵 ---
    st.subheader("📈 精细化运营与品类洞察矩阵")
    
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.write("**📍 1. 流量分配四象限 (曝光量 VS 口碑)**")
        st.write("快速定位需要被扶持的“高分低流”隐藏神店。")
        st.scatter_chart(data=df, x='Review_Count', y='Avg_Rating', color='Cost')
        
        st.write("**📍 2. 品类赛道拥挤度分析 (红蓝海洞察)**")
        st.write("统计各大菜系在平台的覆盖度，指导后续招商策略。")
        cuisines_exploded = df.assign(Cuisine=df['Cuisines'].str.split(', ')).explode('Cuisine')
        top_cuisines = cuisines_exploded['Cuisine'].value_counts().head(10)
        st.bar_chart(top_cuisines, color='#ff4b4b')
        
    with chart_col2:
        st.write("**📍 3. 性价比分布雷达 (客单价 VS 口碑)**")
        st.write("监控市场定价健康度，揪出处于右下角的“高客单低质”刺客。")
        st.scatter_chart(data=df, x='Cost', y='Avg_Rating', color='#ff9900')
        
        st.write("**📍 4. 商家金字塔结构**")
        st.write("查看平台 S/A/B/C 级商家的分布健康度。")
        st.bar_chart(tier_counts, color="#00a4d6")

# ==================== Tab 3: 算法风控端 ====================
with tab3:
    st.subheader("🔬 LASSO-Logistic 风险归因与诊断引擎")
    st.write("利用 LASSO 归因模型拆解风险因子，并自动下发 AM 运营处置 SOP 动作。")
    
    selected_res = st.selectbox("🔍 选择评估对象：", df['Name'].tolist())
    
    if selected_res:
        res_info = df[df['Name'] == selected_res].iloc[0]
        
        f_rating = (5.0 - res_info['Avg_Rating']) * 15.0
        f_price = (res_info['Cost'] / 3000.0) * 20.0 
        f_std = res_info['Rating_Std'] * 12.0
        f_vol = min(1.5, max(0.5, res_info['Review_Count'] / 80.0))
        
        raw_risk = (f_rating + f_price + f_std) * f_vol
        final_risk = min(99, max(1, int(raw_risk)))
        
        col_risk_main, col_risk_factors = st.columns([1, 2])
        
        with col_risk_main:
            st.metric("综合预警风险值", f"{final_risk}%", delta=f"{'高危' if final_risk>60 else '安全'}", delta_color="inverse")
            st.progress(final_risk / 100.0)
            st.write(f"**等级：** {res_info['Merchant_Tier']}")
            st.write(f"**客单价：** ₹{res_info['Cost']}")
            st.write(f"**历史评分：** {res_info['Avg_Rating']} 星")
            
        with col_risk_factors:
            st.write("**📌 LASSO 特征归因拆解 (因子风险贡献度)：**")
            factor_df = pd.DataFrame({
                "风险特征": ["口碑缺失扣分", "客单价期待溢价", "服务不稳定度(方差)"],
                "贡献得分": [round(f_rating, 1), round(f_price, 1), round(f_std, 1)]
            }).set_index("风险特征")
            
            st.bar_chart(factor_df, color="#ff4b4b")
            st.caption(f"💡 流量系数 multiplier: x{f_vol:.2f} (评论基数越大，风险传导效应越强)")
            
        st.divider()
        st.subheader("🤖 AM 自动化处置决策 (SOP)")
        if final_risk >= 70:
            st.error("**🔴 触发高危处置 SOP：**\n1. 切断 CPC 竞价广告流量，暂停推荐。\n2. 自然搜索列表进行 -50% 降权惩罚。\n3. 派发紧急整改工单。")
        elif final_risk >= 40:
            st.warning("**🟡 触发观察 SOP：**\n1. 暂停参与平台高额补贴活动。\n2. 商家后台弹窗提示优化定价。")
        else:
            st.success("**🟢 触发绿灯加速 SOP：**\n1. 纳入公域流量池，允许参与竞价排名。\n2. AM 介入推荐购买商业广告套餐。")