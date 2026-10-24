"""Streamlit Web Application: RL-Based Dynamic Pricing Optimization for Vintage Fashion on Shopee.
Interactive UI for Supply Chain exploration, MDP environment simulation, and RL policy benchmarking.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project modules are discoverable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.utils.helpers import load_config, set_seed, format_currency_idr
from src.data.loader import ShopeeDatasetLoader
from src.features.feature_pipeline import PricingFeatureEngineer
from src.environment.shopee_pricing_env import ShopeeDynamicPricingEnv
from src.agents.baselines import (
    FixedCatalogPricingAgent,
    RuleBasedMarkdownAgent,
    CampaignAwarePricingAgent,
)
from src.agents.q_learning import QLearningPricingAgent
from src.agents.bandit_agents import LinUCBPricingAgent
from src.evaluation.backtester import PricingBacktester
from src.evaluation.metrics import compute_financial_metrics

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RL Dynamic Pricing | Vintage Fashion Shopee",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS STYLING ---
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #EE4D2D, #FF7337, #FF9B26);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E222D;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #EE4D2D;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .badge-shopee {
        background-color: #EE4D2D;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-rl {
        background-color: #00897B;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_and_preprocess_data():
    """Loads dataset and prepares SKU features with caching."""
    config = load_config("configs/config.yaml")
    loader = ShopeeDatasetLoader(config["data"]["raw_file"])
    sheets = loader.load_all_sheets()

    engineer = PricingFeatureEngineer(
        platform_fee_pct=config["platform"]["fee_rates"]["total_effective_fee_pct"],
        daily_holding_cost_rp=config["inventory_economics"]["daily_holding_cost_rp"],
    )
    sku_features = engineer.build_sku_features(sheets)
    return config, sheets, sku_features


config, sheets, sku_features = load_and_preprocess_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 15px;'>
        <h2 style='color: #EE4D2D; margin: 0;'>🛍️ Shopee Vintage RL</h2>
        <p style='font-size: 0.85rem; color: #888;'>Dynamic Pricing Optimization</p>
    </div>
    """,
    unsafe_allow_html=True,
)

menu = st.sidebar.radio(
    "Navigasi Modul:",
    [
        "📊 Ringkasan & Rantai Pasok",
        "🎮 Simulator Dynamic Pricing",
        "🏆 Evaluasi & Benchmark Kebijakan",
        "🧮 Kalkulator Margin & Fee Shopee",
    ],
    index=1,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Karakteristik Fesyen Vintage 90an**:
    - *Kelangkaan*: Deadstock & thrift kurasi
    - *Holding Cost*: Beban gudang per hari
    - *Platform Fee*: ~9.0% (Star + Gratis Ongkir)
    - *Promo Shocks*: Payday (25-27) & Double Date
    """
)

# ==============================================================================
# 1. RINGKASAN & RANTAI PASOK
# ==============================================================================
if menu == "📊 Ringkasan & Rantai Pasok":
    st.markdown('<div class="main-header">Katalog Pakaian Vintage 90an & Rantai Pasok Shopee</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Analisis terintegrasi 6 lembar kerja: Produk Master, Pemasok, Pembelian, Pesanan, Retur, & Persediaan.</div>',
        unsafe_allow_html=True,
    )

    # Top KPI Metrics
    total_skus = len(sku_features)
    total_orders = len(sheets["orders"])
    avg_gross_margin = sku_features["Margin_Kotor_%"].mean() * 100
    avg_net_margin = sku_features["Net_Margin_%"].mean()
    total_inventory = sku_features["Stok_Akhir"].sum()
    avg_return_rate = sku_features["Return_Rate_%"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total SKU Vintage", f"{total_skus} Item", "Pakaian 90an")
    c2.metric("Total Pesanan", f"{total_orders} Order", "Historis")
    c3.metric("Rata-rata Margin Kotor", f"{avg_gross_margin:.1f}%", "Sebelum Fee")
    c4.metric("Rata-rata Margin Neto", f"{avg_net_margin:.1f}%", "Setelah Fee Shopee 9%")
    c5.metric("Sisa Stok Gudang", f"{total_inventory} Unit", f"Retur: {avg_return_rate:.1f}%")

    st.markdown("---")

    col_filter1, col_filter2 = st.columns(2)
    categories = ["Semua"] + sorted(sku_features["Kategori"].unique().tolist())
    selected_cat = col_filter1.selectbox("Filter Kategori:", categories)

    styles = ["Semua"] + sorted(sku_features["Gaya_90an"].unique().tolist())
    selected_style = col_filter2.selectbox("Filter Gaya 90-an:", styles)

    filtered_df = sku_features.copy()
    if selected_cat != "Semua":
        filtered_df = filtered_df[filtered_df["Kategori"] == selected_cat]
    if selected_style != "Semua":
        filtered_df = filtered_df[filtered_df["Gaya_90an"] == selected_style]

    # Visualisasi Scatter HPP vs Harga Jual
    fig_scatter = px.scatter(
        filtered_df,
        x="Modal_Unit_Rp",
        y="Harga_Jual_Rp",
        size="Stok_Akhir",
        color="Kategori",
        hover_name="Nama_Produk",
        hover_data=["SKU", "Gaya_90an", "Net_Margin_%", "Return_Rate_%"],
        title="Distribusi Modal (HPP) vs Harga Jual Katalog (Ukuran gelembung = Sisa Stok)",
        labels={"Modal_Unit_Rp": "Modal Unit (HPP Rp)", "Harga_Jual_Rp": "Harga Jual Katalog (Rp)"},
    )
    fig_scatter.update_layout(height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Tabel Data Terintegrasi
    st.subheader("📋 Rincian SKU & Unit Economics")
    display_cols = [
        "SKU", "Nama_Produk", "Kategori", "Gaya_90an", "Ukuran", "Kondisi",
        "Modal_Unit_Rp", "Harga_Jual_Rp", "Net_Unit_Profit_Rp", "Net_Margin_%",
        "Stok_Akhir", "Daily_Velocity", "Days_Of_Inventory", "Status_Stok"
    ]
    st.dataframe(
        filtered_df[display_cols].style.format({
            "Modal_Unit_Rp": lambda x: f"Rp {x:,.0f}",
            "Harga_Jual_Rp": lambda x: f"Rp {x:,.0f}",
            "Net_Unit_Profit_Rp": lambda x: f"Rp {x:,.0f}",
            "Net_Margin_%": "{:.1f}%",
            "Daily_Velocity": "{:.2f} unit/hari",
            "Days_Of_Inventory": "{:.1f} hari",
        }),
        use_container_width=True,
    )

# ==============================================================================
# 2. SIMULATOR DYNAMIC PRICING
# ==============================================================================
elif menu == "🎮 Simulator Dynamic Pricing":
    st.markdown('<div class="main-header">Simulator Dynamic Pricing Interaktif</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Uji coba interaksi lingkungan MDP retail Shopee terhadap respon elastisitas harga, promo kampanye, dan biaya holding.</div>',
        unsafe_allow_html=True,
    )

    # Panel Kontrol Parameter
    with st.expander("⚙️ Konfigurasi SKU & Parameter Pasar", expanded=True):
        col_sku, col_agent, col_horizon = st.columns([2, 2, 1])

        sku_list = sku_features["SKU"].tolist()
        sku_labels = [f"{row['SKU']} - {row['Nama_Produk']} ({row['Gaya_90an']})" for _, row in sku_features.iterrows()]
        selected_sku_idx = col_sku.selectbox("Pilih SKU Vintage Target:", range(len(sku_list)), format_func=lambda i: sku_labels[i])
        selected_sku = sku_list[selected_sku_idx]

        agent_option = col_agent.selectbox(
            "Pilih Kebijakan / Model Penetapan Harga:",
            [
                "🧠 Tabular Q-Learning (RL Agent)",
                "🎯 LinUCB Contextual Bandit",
                "📉 Rule-Based Markdown (Clearance Heuristic)",
                "🏷️ Fixed Catalog Price (Statis)",
                "⚡ Campaign-Aware Pricing (Eksploitasi Promo)",
                "⚖️ Bandingkan Semua Kebijakan",
            ],
            index=0,
        )

        sim_horizon = col_horizon.slider("Horizon Waktu (Hari):", min_value=15, max_value=90, value=60, step=5)

    sku_info = sku_features[sku_features["SKU"] == selected_sku].iloc[0]

    # Informasi SKU
    i1, i2, i3, i4, i5 = st.columns(5)
    i1.markdown(f"**SKU**: `{selected_sku}`")
    i2.markdown(f"**Modal (HPP)**: {format_currency_idr(sku_info['Modal_Unit_Rp'])}")
    i3.markdown(f"**Harga Dasar**: {format_currency_idr(sku_info['Harga_Jual_Rp'])}")
    i4.markdown(f"**Stok Awal**: {sku_info['Stok_Akhir']} unit")
    i5.markdown(f"**Elastisitas Proxy**: `{sku_info.get('Estimated_Price_Elasticity', -1.5)}`")

    # Tombol Eksekusi
    if st.button("🚀 Jalankan Simulasi Siklus Penjualan", type="primary"):
        # Setup Environment
        env = ShopeeDynamicPricingEnv(
            sku=selected_sku,
            base_price=float(sku_info["Harga_Jual_Rp"]),
            unit_cost=float(sku_info["Modal_Unit_Rp"]),
            initial_stock=int(max(10, sku_info.get("Stok_Akhir", 50))),
            horizon_days=sim_horizon,
            base_daily_demand=float(sku_info.get("Daily_Velocity", 1.5)),
            price_elasticity=float(sku_info.get("Estimated_Price_Elasticity", -1.5)),
            platform_fee_pct=config["platform"]["fee_rates"]["total_effective_fee_pct"],
            daily_holding_cost=config["inventory_economics"]["daily_holding_cost_rp"],
            stockout_penalty=config["inventory_economics"]["stockout_penalty_rp"],
            salvage_value_ratio=config["inventory_economics"]["salvage_value_ratio"],
            discrete_actions=config["rl"]["action_space"]["discrete_actions"],
            seed=42,
        )

        def train_q_agent(train_eps=250):
            q_ag = QLearningPricingAgent(
                num_actions=env.num_actions,
                learning_rate=0.05,
                gamma=0.98,
                epsilon=1.0,
                epsilon_min=0.05,
                epsilon_decay=0.995,
                seed=42,
            )
            for ep in range(train_eps):
                obs, _ = env.reset(seed=ep)
                done = False
                while not done:
                    act = q_ag.select_action(obs, evaluate=False)
                    next_obs, rew, terminated, truncated, _ = env.step(act)
                    q_ag.update(obs, act, rew, next_obs, terminated)
                    obs = next_obs
                    done = terminated or truncated
                q_ag.decay_epsilon()
            return q_ag

        # Dictionary of agents
        agent_instances = {
            "Fixed Catalog Price": FixedCatalogPricingAgent(action_index=3, num_actions=env.num_actions),
            "Rule-Based Markdown": RuleBasedMarkdownAgent(num_actions=env.num_actions),
            "Campaign-Aware Pricing": CampaignAwarePricingAgent(num_actions=env.num_actions),
            "LinUCB Contextual Bandit": LinUCBPricingAgent(context_dim=6, num_actions=env.num_actions),
            "Tabular Q-Learning (RL)": train_q_agent(train_eps=250),
        }

        # Filter which agent(s) to run
        if "Bandingkan Semua" in agent_option:
            selected_agents = agent_instances
        elif "Q-Learning" in agent_option:
            selected_agents = {"Tabular Q-Learning (RL)": agent_instances["Tabular Q-Learning (RL)"]}
        elif "LinUCB" in agent_option:
            selected_agents = {"LinUCB Contextual Bandit": agent_instances["LinUCB Contextual Bandit"]}
        elif "Markdown" in agent_option:
            selected_agents = {"Rule-Based Markdown": agent_instances["Rule-Based Markdown"]}
        elif "Fixed" in agent_option:
            selected_agents = {"Fixed Catalog Price": agent_instances["Fixed Catalog Price"]}
        else:
            selected_agents = {"Campaign-Aware Pricing": agent_instances["Campaign-Aware Pricing"]}

        # Run trajectory simulation
        trajectories = {}
        metrics_summary = []

        for name, ag in selected_agents.items():
            obs, _ = env.reset(seed=1234)
            terminated, truncated = False, False
            hist = []
            while not (terminated or truncated):
                act = ag.select_action(obs, evaluate=True)
                next_obs, rew, terminated, truncated, s_info = env.step(act)
                hist.append(s_info)
                obs = next_obs
            trajectories[name] = pd.DataFrame(hist)
            m = compute_financial_metrics(hist, env.initial_stock, env.unit_cost)
            m["Kebijakan"] = name
            metrics_summary.append(m)

        # Plot 1: Trajektori Harga Harian
        st.subheader("📈 Trajektori Harga Dinamis vs Waktu (Hari)")
        fig_price = go.Figure()

        # Add campaign days shading
        for cday in env.campaign_days:
            if cday < sim_horizon:
                fig_price.add_vrect(
                    x0=cday - 0.5, x1=cday + 0.5,
                    fillcolor="rgba(255, 155, 38, 0.2)",
                    layer="below", line_width=0,
                    annotation_text="Promo" if cday == env.campaign_days[0] else "",
                )

        for name, df_traj in trajectories.items():
            fig_price.add_trace(
                go.Scatter(
                    x=df_traj["day"],
                    y=df_traj["offered_price"],
                    mode="lines+markers",
                    name=name,
                    line=dict(width=2.5),
                )
            )

        fig_price.add_hline(
            y=env.base_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Harga Acuan Katalog ({format_currency_idr(env.base_price)})",
        )
        fig_price.update_layout(
            xaxis_title="Hari Simulasi (t)",
            yaxis_title="Harga Jual Ditawarkan (Rp)",
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # Plot 2: Penurunan Stok & Akumulasi Laba
        c_left, c_right = st.columns(2)

        # Stok berkurang
        fig_stock = go.Figure()
        for name, df_traj in trajectories.items():
            fig_stock.add_trace(
                go.Scatter(x=df_traj["day"], y=df_traj["stock_remaining"], mode="lines", name=name)
            )
        fig_stock.update_layout(
            title="Penurunan Sisa Stok Gudang (Inventory Depletion)",
            xaxis_title="Hari",
            yaxis_title="Sisa Unit",
            height=350,
        )
        c_left.plotly_chart(fig_stock, use_container_width=True)

        # Laba kumulatif
        fig_profit = go.Figure()
        for name, df_traj in trajectories.items():
            fig_profit.add_trace(
                go.Scatter(x=df_traj["day"], y=df_traj["cumulative_profit"], mode="lines", name=name)
            )
        fig_profit.update_layout(
            title="Akumulasi Laba Bersih Realistis (Setelah Fee & Holding Cost)",
            xaxis_title="Hari",
            yaxis_title="Laba Bersih Kumulatif (Rp)",
            height=350,
        )
        c_right.plotly_chart(fig_profit, use_container_width=True)

        # Summary Table
        st.subheader("📊 Hasil Metrik Finansial Siklus Stok")
        res_df = pd.DataFrame(metrics_summary)
        disp_cols = [
            "Kebijakan", "Total_Net_Profit_Rp", "Total_GMV_Rp", "Sell_Through_Rate_%",
            "Avg_Selling_Price_Rp", "Total_Holding_Cost_Rp", "Ending_Stock_Units"
        ]
        st.dataframe(
            res_df[disp_cols].style.format({
                "Total_Net_Profit_Rp": lambda x: f"Rp {x:,.0f}",
                "Total_GMV_Rp": lambda x: f"Rp {x:,.0f}",
                "Avg_Selling_Price_Rp": lambda x: f"Rp {x:,.0f}",
                "Total_Holding_Cost_Rp": lambda x: f"Rp {x:,.0f}",
                "Sell_Through_Rate_%": "{:.1f}%",
            }),
            use_container_width=True,
        )

# ==============================================================================
# 3. EVALUASI & BENCHMARK KEBIJAKAN
# ==============================================================================
elif menu == "🏆 Evaluasi & Benchmark Kebijakan":
    st.markdown('<div class="main-header">Evaluasi Benchmark Kebijakan (Monte Carlo)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Uji ketahanan 5 kebijakan harga terhadap 50 episode pasar acak dengan proses stokastik permintaan pembeli.</div>',
        unsafe_allow_html=True,
    )

    col_bench_sku, col_num_ep = st.columns([2, 1])
    selected_bench_sku = col_bench_sku.selectbox("Pilih SKU Evaluasi:", sku_features["SKU"].tolist())
    eval_eps = col_num_ep.slider("Jumlah Episode Monte Carlo:", 20, 100, 40, step=10)

    if st.button("🏁 Mulai Uji Monte Carlo Head-to-Head", type="primary"):
        with st.spinner("Menjalankan simulasi Monte Carlo across 5 agents..."):
            sku_info = sku_features[sku_features["SKU"] == selected_bench_sku].iloc[0]

            env = ShopeeDynamicPricingEnv(
                sku=selected_bench_sku,
                base_price=float(sku_info["Harga_Jual_Rp"]),
                unit_cost=float(sku_info["Modal_Unit_Rp"]),
                initial_stock=int(max(10, sku_info.get("Stok_Akhir", 50))),
                horizon_days=60,
                base_daily_demand=float(sku_info.get("Daily_Velocity", 1.5)),
                price_elasticity=float(sku_info.get("Estimated_Price_Elasticity", -1.5)),
                platform_fee_pct=config["platform"]["fee_rates"]["total_effective_fee_pct"],
                daily_holding_cost=config["inventory_economics"]["daily_holding_cost_rp"],
                stockout_penalty=config["inventory_economics"]["stockout_penalty_rp"],
                salvage_value_ratio=config["inventory_economics"]["salvage_value_ratio"],
                discrete_actions=config["rl"]["action_space"]["discrete_actions"],
                seed=42,
            )

            # Train Q-Agent
            q_agent = QLearningPricingAgent(num_actions=env.num_actions, seed=42)
            for ep in range(250):
                obs, _ = env.reset(seed=ep)
                done = False
                while not done:
                    act = q_agent.select_action(obs, evaluate=False)
                    next_obs, rew, terminated, truncated, _ = env.step(act)
                    q_agent.update(obs, act, rew, next_obs, terminated)
                    obs = next_obs
                    done = terminated or truncated
                q_agent.decay_epsilon()

            agents = [
                FixedCatalogPricingAgent(action_index=3, num_actions=env.num_actions),
                RuleBasedMarkdownAgent(num_actions=env.num_actions),
                CampaignAwarePricingAgent(num_actions=env.num_actions),
                LinUCBPricingAgent(context_dim=6, num_actions=env.num_actions),
                q_agent,
            ]

            backtester = PricingBacktester(env)
            summary_df = backtester.compare_agents(agents, num_episodes=eval_eps, base_seed=7000)

            # Bar Chart Komparasi Laba Bersih
            fig_bar_profit = px.bar(
                summary_df,
                x="Agent",
                y="Mean_Net_Profit",
                error_y="Std_Net_Profit",
                color="Agent",
                title=f"Rata-rata Laba Bersih ({eval_eps} Episode Monte Carlo) - {selected_bench_sku}",
                labels={"Mean_Net_Profit": "Rata-rata Laba Bersih (Rp)", "Agent": "Kebijakan Harga"},
            )
            st.plotly_chart(fig_bar_profit, use_container_width=True)

            # Bar Chart Sell-Through vs Holding Cost
            col_b1, col_b2 = st.columns(2)

            fig_sell = px.bar(
                summary_df,
                x="Agent",
                y="Mean_Sell_Through",
                color="Agent",
                title="Tingkat Penjualan Stok (% Sell-Through)",
                labels={"Mean_Sell_Through": "Sell-Through (%)"},
            )
            col_b1.plotly_chart(fig_sell, use_container_width=True)

            fig_hold = px.bar(
                summary_df,
                x="Agent",
                y="Mean_Holding_Cost",
                color="Agent",
                title="Rata-rata Beban Biaya Simpan Gudang",
                labels={"Mean_Holding_Cost": "Biaya Gudang (Rp)"},
            )
            col_b2.plotly_chart(fig_hold, use_container_width=True)

            st.subheader("📋 Ringkasan Metrik Numerik")
            st.dataframe(
                summary_df.style.format({
                    "Mean_Net_Profit": lambda x: f"Rp {x:,.0f}",
                    "Std_Net_Profit": lambda x: f"Rp {x:,.0f}",
                    "Mean_GMV": lambda x: f"Rp {x:,.0f}",
                    "Mean_ASP": lambda x: f"Rp {x:,.0f}",
                    "Mean_Holding_Cost": lambda x: f"Rp {x:,.0f}",
                    "Mean_Sell_Through": "{:.1f}%",
                }),
                use_container_width=True,
            )

# ==============================================================================
# 4. KALKULATOR MARGIN & FEE SHOPEE
# ==============================================================================
elif menu == "🧮 Kalkulator Margin & Fee Shopee":
    st.markdown('<div class="main-header">Kalkulator Simulasi Margin & Biaya Marketplace Shopee</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Hitung laba riil, potongan biaya Star Seller, dan titik impas (Break-Even) per potong pakaian vintage.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Parameter Produk")
        calc_cost = st.number_input("Modal Unit / HPP (Rp):", min_value=10000, max_value=1000000, value=45000, step=5000)
        calc_price = st.number_input("Harga Jual Ditawarkan (Rp):", min_value=10000, max_value=2000000, value=99000, step=5000)
        calc_discount = st.slider("Diskon Voucher Toko (Rp):", min_value=0, max_value=30000, value=5000, step=1000)

    with c2:
        st.subheader("Rincian Fee Shopee")
        admin_fee_pct = st.slider("Fee Administrasi Star Seller Fashion (%):", min_value=3.0, max_value=7.0, value=4.5, step=0.1)
        ongkir_fee_pct = st.slider("Fee Program Gratis Ongkir Xtra (%):", min_value=2.0, max_value=5.0, value=3.5, step=0.1)
        proc_fee_pct = st.slider("Biaya Pemrosesan Transaksi (%):", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

    effective_price = max(0, calc_price - calc_discount)
    total_fee_pct = (admin_fee_pct + ongkir_fee_pct + proc_fee_pct) / 100.0

    shopee_fee_rp = effective_price * total_fee_pct
    net_omzet_rp = effective_price - shopee_fee_rp
    net_profit_rp = net_omzet_rp - calc_cost
    net_margin_pct = (net_profit_rp / effective_price * 100) if effective_price > 0 else 0
    break_even_price = calc_cost / (1.0 - total_fee_pct)

    st.markdown("---")
    st.subheader("💡 Hasil Kalkulasi Laba Bersih")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Harga Efektif Pembeli", format_currency_idr(effective_price))
    m2.metric("Total Potongan Shopee", format_currency_idr(shopee_fee_rp), f"{total_fee_pct*100:.1f}%")
    m3.metric("Laba Bersih Per Potong", format_currency_idr(net_profit_rp), f"Margin: {net_margin_pct:.1f}%")
    m4.metric("Harga Titik Impas (BEP)", format_currency_idr(break_even_price), "Batas Bawah")

    # Waterfall chart
    fig_waterfall = go.Figure(
        go.Waterfall(
            name="Unit Economics",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Harga Efektif", "Biaya Shopee (Fee)", "Modal (HPP)", "Laba Bersih"],
            textposition="outside",
            text=[
                format_currency_idr(effective_price),
                f"-{format_currency_idr(shopee_fee_rp)}",
                f"-{format_currency_idr(calc_cost)}",
                format_currency_idr(net_profit_rp),
            ],
            y=[effective_price, -shopee_fee_rp, -calc_cost, net_profit_rp],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )
    fig_waterfall.update_layout(title="Struktur Penguraian Laba Bersih (Waterfall Chart)", height=400)
    st.plotly_chart(fig_waterfall, use_container_width=True)
