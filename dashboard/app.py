import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_full_pipeline
from src.performance import compute_metrics
from src.config import ASSETS

st.set_page_config(page_title="Chokepoint Beta — MENA Risk Portfolio", layout="wide")
st.title("Chokepoint Beta — MENA Risk-Aware Multi-Asset Portfolio")
st.markdown(
    "Quantifying how Strait of Hormuz & Bab el-Mandeb disruptions transmit "
    "across equities, bonds, commodities, FX, and volatility — and using those "
    "insights for regime-aware portfolio construction."
)

st.sidebar.header("Controls")
gpr_method = st.sidebar.selectbox("GPR Source", ["published", "deepseek", "auto"], index=0)
run_pipeline = st.sidebar.button("Run / Refresh Pipeline")

if "results" not in st.session_state or run_pipeline:
    with st.spinner("Running pipeline (fetching data, fitting models)..."):
        st.session_state.results = run_full_pipeline(
            fetch_market=True, gpr_method=gpr_method
        )

res = st.session_state.results

tab1, tab2, tab3, tab4 = st.tabs([
    "Chokepoint Risk Monitor",
    "Event Study",
    "Portfolio & Asymmetry",
    "Risk Metrics",
])

with tab1:
    st.header("Chokepoint Geopolitical Risk Index")

    col1, col2 = st.columns([2, 1])
    with col1:
        gpr = res.get("gpr", pd.Series(dtype=float))
        if not gpr.empty:
            fig = px.line(
                x=gpr.index, y=gpr.values,
                labels={"x": "Date", "y": "GPR Index"},
                title=f"GPR Index ({res.get('gpr_name', 'N/A')})",
            )
            fig.update_layout(
                hovermode="x",
                shapes=[
                    dict(
                        type="line", x0=evt["date"], x1=evt["date"],
                        y0=gpr.min(), y1=gpr.max(),
                        line=dict(color="red", width=1, dash="dot"),
                    )
                    for _, evt in res["events"].iterrows()
                    if gpr.index[0] <= pd.Timestamp(evt["date"]) <= gpr.index[-1]
                ],
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Current GPR", f"{gpr.iloc[-1]:.1f}" if not gpr.empty else "N/A")
        st.metric("GPR 90th %ile", f"{gpr.quantile(0.9):.1f}" if not gpr.empty else "N/A")
        st.metric("GPR Mean", f"{gpr.mean():.1f}" if not gpr.empty else "N/A")

    st.subheader("Historical Chokepoint Events")
    events_df = res["events"].copy()
    events_df["date"] = pd.to_datetime(events_df["date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(
        events_df[["date", "name", "chokepoint", "category", "severity"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Regime Timeline")
    labels = res.get("regime_labels")
    if labels is not None:
        color_map = {"Normal": "green", "Elevated Risk": "orange", "Crisis": "red"}
        fig_reg = go.Figure()
        for regime, color in color_map.items():
            mask = labels == regime
            if mask.any():
                fig_reg.add_trace(go.Scatter(
                    x=labels.index[mask],
                    y=[1] * mask.sum(),
                    mode="markers",
                    marker=dict(color=color, size=3, symbol="square"),
                    name=regime,
                ))
        fig_reg.update_layout(
            title="Markov-Switching Regime Classification",
            yaxis=dict(showticklabels=False),
            height=200,
            hovermode="x",
        )
        st.plotly_chart(fig_reg, use_container_width=True)

        regime_counts = labels.value_counts()
        fig_pie = px.pie(
            values=regime_counts.values,
            names=regime_counts.index,
            title="Regime Distribution",
            color=regime_counts.index,
            color_discrete_map={"Normal": "green", "Elevated Risk": "orange", "Crisis": "red"},
        )
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.header("Event Study — Multi-Asset Response Around Chokepoint Events")

    es_summary = res.get("event_study")
    if es_summary is not None and not es_summary.empty:
        st.dataframe(es_summary, use_container_width=True)

        es_detail = res.get("event_study_detailed", {})
        asset_choice = st.selectbox(
            "Select asset to view AAR/CAR",
            options=list(es_detail.keys()),
        )
        if asset_choice in es_detail:
            ed = es_detail[asset_choice]
            fig_es = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    subplot_titles=("Average Abnormal Return (AAR)",
                                                    "Cumulative Abnormal Return (CAR)"))

            fig_es.add_trace(
                go.Scatter(x=ed.aar.index, y=ed.aar.values,
                           mode="lines+markers", name="AAR",
                           line=dict(color="blue")),
                row=1, col=1,
            )
            fig_es.add_trace(
                go.Scatter(x=ed.aar.index, y=ed.car.values,
                           mode="lines+markers", name="CAR",
                           line=dict(color="red")),
                row=2, col=1,
            )
            fig_es.update_layout(
                height=500,
                title=f"{ASSETS.get(asset_choice, {}).get('name', asset_choice)} — Event Study",
            )
            st.plotly_chart(fig_es, use_container_width=True)

            perf_stats = pd.DataFrame({
                "Metric": ["CAR", "p-value", "Hit Rate", "AAR Peak", "AAR Trough"],
                "Value": [
                    f"{ed.car.iloc[-1]:.4%}",
                    f"{ed.car_p_value:.4f}",
                    f"{ed.hit_rate:.0%}",
                    f"{ed.aar.max():.4%}",
                    f"{ed.aar.min():.4%}",
                ],
            })
            st.dataframe(perf_stats, hide_index=True, use_container_width=True)

with tab3:
    st.header("Portfolio Performance & Asymmetry")

    bt = res.get("backtest")
    bench = res.get("benchmark")

    if bt is not None and bench is not None:
        col1, col2 = st.columns(2)
        with col1:
            perf_df = res.get("performance")
            if perf_df is not None:
                styled = perf_df.style.format("{:.4f}")
                st.dataframe(styled, use_container_width=True)

        with col2:
            metrics_bt = compute_metrics(bt.portfolio_returns)
            metrics_bench = compute_metrics(bench)

            comp = pd.DataFrame({
                "Metric": ["Sharpe", "Sortino", "Calmar", "Hit Ratio", "Asymmetry",
                           "Max DD", "Ann. Return", "Ann. Vol"],
                "MENA Portfolio": [
                    metrics_bt.sharpe_ratio, metrics_bt.sortino_ratio,
                    metrics_bt.calmar_ratio, metrics_bt.hit_ratio,
                    metrics_bt.asymmetry, metrics_bt.max_drawdown,
                    metrics_bt.annualised_return, metrics_bt.annualised_vol,
                ],
                "Equal-Weight Benchmark": [
                    metrics_bench.sharpe_ratio, metrics_bench.sortino_ratio,
                    metrics_bench.calmar_ratio, metrics_bench.hit_ratio,
                    metrics_bench.asymmetry, metrics_bench.max_drawdown,
                    metrics_bench.annualised_return, metrics_bench.annualised_vol,
                ],
            })
            st.dataframe(comp.set_index("Metric").style.format("{:.4f}"), use_container_width=True)

        st.subheader("Cumulative Returns")
        cum_df = pd.DataFrame({
            "MENA Risk-Aware Portfolio": bt.cumulative_returns,
            "Equal-Weight Benchmark": (1 + bench).cumprod(),
        }).dropna()
        fig_cum = px.line(cum_df, title="Portfolio vs Benchmark")
        st.plotly_chart(fig_cum, use_container_width=True)

        st.subheader("Rolling Sharpe (6-month)")
        roll_sharpe_bt = bt.portfolio_returns.rolling(126).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
        )
        roll_sharpe_bench = bench.rolling(126).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
        )
        roll_df = pd.DataFrame({
            "MENA Portfolio": roll_sharpe_bt,
            "Benchmark": roll_sharpe_bench,
        }).dropna()
        fig_roll = px.line(roll_df, title="Rolling Sharpe Ratio (6-month window)")
        st.plotly_chart(fig_roll, use_container_width=True)

        st.subheader("Portfolio Weights Over Time")
        w_df = bt.weights
        if not w_df.empty:
            fig_w = px.area(w_df, title="Dynamic Weights")
            st.plotly_chart(fig_w, use_container_width=True)

        st.subheader("Turnover & Transaction Costs")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_to = px.line(bt.turnover, title="Portfolio Turnover")
            st.plotly_chart(fig_to, use_container_width=True)
        with col_t2:
            if hasattr(bt, "transaction_costs") and bt.transaction_costs is not None:
                tc_df = bt.transaction_costs.to_frame(name="cost")
                fig_tc = px.line(tc_df, title="Daily Transaction Cost (bps)")
                st.plotly_chart(fig_tc, use_container_width=True)
                st.metric("Avg Daily TC (bps)",
                          f"{bt.transaction_costs.mean() * 10000:.2f}")

with tab4:
    st.header("Risk Metrics")

    bt = res.get("backtest")
    if bt is not None:
        metrics = compute_metrics(bt.portfolio_returns)

        m_cols = st.columns(4)
        with m_cols[0]:
            st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
            st.metric("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
        with m_cols[1]:
            st.metric("Max Drawdown", f"{metrics.max_drawdown:.2%}")
            st.metric("Calmar Ratio", f"{metrics.calmar_ratio:.2f}")
        with m_cols[2]:
            st.metric("Hit Ratio", f"{metrics.hit_ratio:.0%}")
            st.metric("Asymmetry", f"{metrics.asymmetry:.2f}x")
        with m_cols[3]:
            st.metric("95% VaR (daily)", f"{metrics.var_95:.4f}")
            st.metric("95% CVaR (daily)", f"{metrics.cvar_95:.4f}")

        st.subheader("Drawdown Chart")
        cum = bt.cumulative_returns
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        fig_dd = px.area(
            x=drawdown.index, y=drawdown.values,
            labels={"x": "Date", "y": "Drawdown"},
            title="Portfolio Drawdown",
            color_discrete_sequence=["red"],
        )
        fig_dd.update_traces(fill="tozeroy")
        st.plotly_chart(fig_dd, use_container_width=True)

        st.subheader("Return Distribution")
        rets = bt.portfolio_returns.dropna()
        fig_hist = px.histogram(
            rets, nbins=50,
            labels={"value": "Daily Return"},
            title="Return Distribution with Normal Fit",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Cross-Asset Correlation (Full Sample)")
        returns = res.get("returns")
        if returns is not None:
            corr = returns.corr()
            fig_corr = px.imshow(
                corr.values,
                x=corr.columns,
                y=corr.columns,
                text_auto=".2f",
                aspect="auto",
                title="Correlation Matrix",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
            )
            st.plotly_chart(fig_corr, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Sources")
st.sidebar.markdown("- **Market**: Yahoo Finance (yfinance)")
st.sidebar.markdown("- **GPR**: Caldara & Iacoviello / DeepSeek")
st.sidebar.markdown("- **Events**: CRS Reports, Brookings, academic literature")
