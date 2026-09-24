"""
Intelligent RPA Value & Automation Opportunity Analytics
=========================================================
Streamlit dashboard analysing 50,000 enterprise automation projects.

ROI definition used in this dataset:
    ROI (Savings/Budget %) = (annual_savings_usd / budget_usd) × 100
This is NOT the conventional net-return ROI formula.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RPA Analytics Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_PATH = "dataset/automation_projects.csv"

# Consistent colour palette
COLOUR_AI = {"Yes": "#3b82d4", "No": "#e5a025"}
COLOUR_CLOUD = {"Yes": "#7c5cd8", "No": "#57a68a"}
COLOUR_STATUS = {"Completed": "#3b82d4", "Active": "#57a68a", "Planned": "#e5a025"}
ACCENT = "#3b82d4"


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    # Parse dates
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce")

    # Ensure numeric columns
    for col in ["budget_usd", "annual_savings_usd", "roi_percent",
                "robots_deployed", "employee_hours_saved"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived columns
    df["project_duration_days"] = (
        df["completion_date"] - df["start_date"]
    ).dt.days  # NaN for Active / Planned without completion_date

    df["start_year"] = df["start_date"].dt.year.astype("Int64")

    # Net savings (can be negative — investment exceeded savings)
    df["net_savings_usd"] = df["annual_savings_usd"] - df["budget_usd"]

    # Boolean flags for easy groupby arithmetic
    df["ai_flag"] = df["ai_enabled"] == "Yes"
    df["cloud_flag"] = df["cloud_deployment"] == "Yes"

    return df


# ──────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def fmt_currency(value: float, decimals: int = 1) -> str:
    """Format large USD values as $XB / $XM / $XK."""
    if abs(value) >= 1e9:
        return f"${value/1e9:.{decimals}f}B"
    if abs(value) >= 1e6:
        return f"${value/1e6:.{decimals}f}M"
    if abs(value) >= 1e3:
        return f"${value/1e3:.{decimals}f}K"
    return f"${value:,.0f}"


def fmt_hours(value: float) -> str:
    if abs(value) >= 1e6:
        return f"{value/1e6:.1f}M hrs"
    if abs(value) >= 1e3:
        return f"{value/1e3:.1f}K hrs"
    return f"{value:,.0f} hrs"


def percentile_rank(series: pd.Series, value: float) -> float:
    """Return 0-100 percentile rank of `value` within `series`."""
    if len(series) == 0:
        return 0.0
    return float(np.sum(series <= value) / len(series) * 100)


def compute_opportunity_scores(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Compute the composite Opportunity Score for each group in `group_col`.

    Score = mean of three component percentile ranks:
      1. ROI Score      — percentile rank of group median ROI
      2. Savings Score  — percentile rank of group total annual savings
      3. Productivity   — percentile rank of group avg employee hours saved

    Also flags "High Investment / Low Return" groups.
    """
    agg = (
        df.groupby(group_col)
        .agg(
            project_count=(group_col, "count"),
            median_roi=("roi_percent", "median"),
            total_savings=("annual_savings_usd", "sum"),
            total_budget=("budget_usd", "sum"),
            avg_hours_saved=("employee_hours_saved", "mean"),
            ai_pct=("ai_flag", "mean"),
            cloud_pct=("cloud_flag", "mean"),
        )
        .reset_index()
    )

    # Percentile ranks (0-100)
    agg["roi_score"] = agg["median_roi"].rank(pct=True) * 100
    agg["savings_score"] = agg["total_savings"].rank(pct=True) * 100
    agg["productivity_score"] = agg["avg_hours_saved"].rank(pct=True) * 100

    agg["opportunity_score"] = (
        agg["roi_score"] + agg["savings_score"] + agg["productivity_score"]
    ) / 3

    # High Investment / Low Return flag
    median_budget = agg["total_budget"].median()
    roi_p33 = agg["median_roi"].quantile(0.33)
    agg["high_invest_low_return"] = (
        (agg["total_budget"] > median_budget) & (agg["median_roi"] < roi_p33)
    )

    agg = agg.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ──────────────────────────────────────────────────────────────────────────────
def build_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("🔍 Filters")
    st.sidebar.markdown("---")

    def multiselect_all(label, col, df=df):
        options = sorted(df[col].dropna().unique().tolist())
        return st.sidebar.multiselect(label, options, default=[])

    industry = multiselect_all("Industry", "industry")
    department = multiselect_all("Department", "department")
    automation_type = multiselect_all("Automation Type", "automation_type")
    country = multiselect_all("Country", "country")
    status = multiselect_all("Project Status", "project_status")

    ai_choice = st.sidebar.radio("AI Enabled", ["All", "Yes", "No"], horizontal=True)
    cloud_choice = st.sidebar.radio("Cloud Deployment", ["All", "Yes", "No"], horizontal=True)

    # Apply filters (empty selection = no filter applied)
    filtered = df.copy()
    if industry:
        filtered = filtered[filtered["industry"].isin(industry)]
    if department:
        filtered = filtered[filtered["department"].isin(department)]
    if automation_type:
        filtered = filtered[filtered["automation_type"].isin(automation_type)]
    if country:
        filtered = filtered[filtered["country"].isin(country)]
    if status:
        filtered = filtered[filtered["project_status"].isin(status)]
    if ai_choice != "All":
        filtered = filtered[filtered["ai_enabled"] == ai_choice]
    if cloud_choice != "All":
        filtered = filtered[filtered["cloud_deployment"] == cloud_choice]

    st.sidebar.markdown("---")
    st.sidebar.metric("Projects Visible", f"{len(filtered):,}")
    st.sidebar.caption(
        "ℹ️ **ROI definition:** ROI (Savings/Budget %) = "
        "annual\\_savings / budget × 100. "
        "This is NOT conventional net-return ROI."
    )
    return filtered


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — EXECUTIVE OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
def section_overview(df: pd.DataFrame):
    st.header("Executive Overview")

    total = len(df)
    avg_roi = df["roi_percent"].mean()
    total_budget = df["budget_usd"].sum()
    total_savings = df["annual_savings_usd"].sum()
    total_hours = df["employee_hours_saved"].sum()
    completed = (df["project_status"] == "Completed").sum()
    completion_rate = completed / total * 100 if total > 0 else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Projects", f"{total:,}")
    c2.metric("Avg ROI (Savings/Budget %)", f"{avg_roi:.1f}%")
    c3.metric("Total Budget", fmt_currency(total_budget))
    c4.metric("Total Annual Savings", fmt_currency(total_savings))
    c5.metric("Total Hours Saved", fmt_hours(total_hours))
    c6.metric("Completion Rate", f"{completion_rate:.1f}%")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Donut — project status
    with col_left:
        status_counts = df["project_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.pie(
            status_counts, names="Status", values="Count",
            title="Project Status Distribution",
            hole=0.45,
            color="Status",
            color_discrete_map=COLOUR_STATUS,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    # Bar — projects started per year
    with col_right:
        year_df = (
            df.dropna(subset=["start_year"])
            .groupby("start_year")
            .size()
            .reset_index(name="project_count")
        )
        year_df["start_year"] = year_df["start_year"].astype(str)
        fig2 = px.bar(
            year_df, x="start_year", y="project_count",
            title="Projects Started per Year",
            labels={"start_year": "Year", "project_count": "Number of Projects"},
            color_discrete_sequence=[ACCENT],
        )
        fig2.update_layout(margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Summary stats row
    st.markdown("---")
    st.subheader("Dataset Snapshot")
    snap_col1, snap_col2, snap_col3, snap_col4 = st.columns(4)
    snap_col1.metric("Automation Types", df["automation_type"].nunique())
    snap_col2.metric("Industries Covered", df["industry"].nunique())
    snap_col3.metric("Countries", df["country"].nunique())
    snap_col4.metric("Departments", df["department"].nunique())

    # AI & Cloud adoption
    st.markdown("---")
    st.subheader("Technology Adoption")
    ta1, ta2, ta3, ta4 = st.columns(4)
    ai_pct = df["ai_flag"].mean() * 100
    cloud_pct = df["cloud_flag"].mean() * 100
    ai_cloud_pct = (df["ai_flag"] & df["cloud_flag"]).mean() * 100
    ta1.metric("AI-Enabled Projects", f"{ai_pct:.1f}%")
    ta2.metric("Cloud-Deployed Projects", f"{cloud_pct:.1f}%")
    ta3.metric("Both AI + Cloud", f"{ai_cloud_pct:.1f}%")
    ta4.metric(
        "Avg Robots per Project",
        f"{df['robots_deployed'].mean():.1f}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — AUTOMATION PERFORMANCE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def section_performance(df: pd.DataFrame):
    st.header("Automation Performance Analysis")
    st.caption(
        "ROI (Savings/Budget %) = annual_savings_usd / budget_usd × 100. "
        "Median is used for group comparisons to reduce the effect of the 500% ROI cap."
    )
    st.markdown("---")

    # Chart A — Median ROI by Automation Type
    roi_by_type = (
        df.groupby("automation_type")["roi_percent"]
        .median()
        .reset_index()
        .sort_values("roi_percent", ascending=True)
    )
    roi_by_type.columns = ["Automation Type", "Median ROI (%)"]
    fig_a = px.bar(
        roi_by_type, x="Median ROI (%)", y="Automation Type",
        orientation="h",
        title="Median ROI (Savings/Budget %) by Automation Type",
        color="Median ROI (%)",
        color_continuous_scale="Blues",
    )
    fig_a.update_layout(coloraxis_showscale=False, margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig_a, use_container_width=True)

    col1, col2 = st.columns(2)

    # Chart B — Total Annual Savings by Industry
    with col1:
        savings_by_industry = (
            df.groupby("industry")["annual_savings_usd"]
            .sum()
            .reset_index()
            .sort_values("annual_savings_usd", ascending=True)
        )
        savings_by_industry.columns = ["Industry", "Total Annual Savings (USD)"]
        fig_b = px.bar(
            savings_by_industry, x="Total Annual Savings (USD)", y="Industry",
            orientation="h",
            title="Total Annual Savings by Industry",
            color="Total Annual Savings (USD)",
            color_continuous_scale="Purples",
        )
        fig_b.update_layout(coloraxis_showscale=False, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_b, use_container_width=True)

    # Chart C — Average ROI by Department
    with col2:
        roi_by_dept = (
            df.groupby("department")["roi_percent"]
            .median()
            .reset_index()
            .sort_values("roi_percent", ascending=True)
        )
        roi_by_dept.columns = ["Department", "Median ROI (%)"]
        fig_c = px.bar(
            roi_by_dept, x="Median ROI (%)", y="Department",
            orientation="h",
            title="Median ROI (Savings/Budget %) by Department",
            color="Median ROI (%)",
            color_continuous_scale="Greens",
        )
        fig_c.update_layout(coloraxis_showscale=False, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_c, use_container_width=True)

    # Chart D — Scatter: Budget vs Annual Savings
    st.markdown("---")
    sample_df = df.sample(min(5000, len(df)), random_state=42) if len(df) > 5000 else df
    fig_d = px.scatter(
        sample_df,
        x="budget_usd", y="annual_savings_usd",
        color="project_status",
        color_discrete_map=COLOUR_STATUS,
        opacity=0.5,
        labels={
            "budget_usd": "Budget (USD)",
            "annual_savings_usd": "Annual Savings (USD)",
            "project_status": "Status",
        },
        title="Budget vs Annual Savings (sample of up to 5,000 projects)",
        hover_data=["automation_type", "industry", "roi_percent"],
    )
    # Break-even reference line
    max_val = max(sample_df["budget_usd"].max(), sample_df["annual_savings_usd"].max())
    fig_d.add_trace(
        go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines",
            line=dict(dash="dash", color="red", width=1),
            name="Break-even (savings = budget)",
        )
    )
    fig_d.update_layout(margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig_d, use_container_width=True)
    st.caption("Red dashed line = break-even. Points above the line indicate savings > budget.")

    col3, col4 = st.columns(2)

    # Chart E — Avg Employee Hours Saved by Automation Type
    with col3:
        hours_by_type = (
            df.groupby("automation_type")["employee_hours_saved"]
            .mean()
            .reset_index()
            .sort_values("employee_hours_saved", ascending=True)
        )
        hours_by_type.columns = ["Automation Type", "Avg Hours Saved"]
        fig_e = px.bar(
            hours_by_type, x="Avg Hours Saved", y="Automation Type",
            orientation="h",
            title="Avg Employee Hours Saved by Automation Type",
            color="Avg Hours Saved",
            color_continuous_scale="Oranges",
        )
        fig_e.update_layout(coloraxis_showscale=False, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_e, use_container_width=True)

    # Chart F — Avg Robots Deployed by Automation Type
    with col4:
        robots_by_type = (
            df.groupby("automation_type")["robots_deployed"]
            .mean()
            .reset_index()
            .sort_values("robots_deployed", ascending=True)
        )
        robots_by_type.columns = ["Automation Type", "Avg Robots Deployed"]
        fig_f = px.bar(
            robots_by_type, x="Avg Robots Deployed", y="Automation Type",
            orientation="h",
            title="Avg Robots Deployed by Automation Type",
            color="Avg Robots Deployed",
            color_continuous_scale="Reds",
        )
        fig_f.update_layout(coloraxis_showscale=False, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_f, use_container_width=True)

    # Chart G — ROI distribution histogram
    st.markdown("---")
    fig_g = px.histogram(
        df, x="roi_percent", nbins=50,
        title="Distribution of ROI (Savings/Budget %) Across All Projects",
        labels={"roi_percent": "ROI (Savings/Budget %)"},
        color_discrete_sequence=[ACCENT],
    )
    fig_g.update_layout(margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig_g, use_container_width=True)
    st.caption(
        "Note: ROI values are capped at 500% in this dataset. "
        "The spike at 500% represents projects at or above that threshold."
    )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — VALUE DRIVER ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def section_value(df: pd.DataFrame):
    st.header("Value Driver Analysis")
    st.caption(
        "Identifies which project characteristics are associated with higher ROI, "
        "annual savings, and employee productivity gains."
    )
    st.markdown("---")

    metrics = ["roi_percent", "annual_savings_usd", "employee_hours_saved"]
    metric_labels = {
        "roi_percent": "Avg ROI (Savings/Budget %)",
        "annual_savings_usd": "Avg Annual Savings (USD)",
        "employee_hours_saved": "Avg Employee Hours Saved",
    }

    # ── AI vs Non-AI ──────────────────────────────────────────────────────────
    st.subheader("AI-Enabled vs Non-AI Projects")
    ai_compare = (
        df.groupby("ai_enabled")[metrics].mean().reset_index()
    )
    ai_long = ai_compare.melt(id_vars="ai_enabled", var_name="Metric", value_name="Value")
    ai_long["Metric"] = ai_long["Metric"].map(metric_labels)

    fig_ai = px.bar(
        ai_long, x="Metric", y="Value", color="ai_enabled",
        barmode="group",
        color_discrete_map=COLOUR_AI,
        title="AI-Enabled vs Non-AI: Average Performance Metrics",
        labels={"ai_enabled": "AI Enabled", "Value": "Average Value"},
    )
    fig_ai.update_layout(margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig_ai, use_container_width=True)

    # ── Cloud vs Non-Cloud ────────────────────────────────────────────────────
    st.subheader("Cloud vs Non-Cloud Deployments")
    cloud_compare = (
        df.groupby("cloud_deployment")[metrics].mean().reset_index()
    )
    cloud_long = cloud_compare.melt(id_vars="cloud_deployment", var_name="Metric", value_name="Value")
    cloud_long["Metric"] = cloud_long["Metric"].map(metric_labels)

    fig_cloud = px.bar(
        cloud_long, x="Metric", y="Value", color="cloud_deployment",
        barmode="group",
        color_discrete_map=COLOUR_CLOUD,
        title="Cloud vs Non-Cloud: Average Performance Metrics",
        labels={"cloud_deployment": "Cloud Deployment", "Value": "Average Value"},
    )
    fig_cloud.update_layout(margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig_cloud, use_container_width=True)

    # ── 2×2 Heatmap: AI × Cloud → Avg ROI ───────────────────────────────────
    st.subheader("AI × Cloud Combination: Average ROI (Savings/Budget %)")
    pivot = (
        df.groupby(["ai_enabled", "cloud_deployment"])["roi_percent"]
        .mean()
        .unstack(fill_value=0)
    )
    fig_hm = px.imshow(
        pivot,
        text_auto=".1f",
        color_continuous_scale="Blues",
        labels=dict(x="Cloud Deployment", y="AI Enabled", color="Avg ROI (%)"),
        title="Average ROI (%) by AI Enabled × Cloud Deployment",
        aspect="auto",
    )
    fig_hm.update_layout(margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── Top Automation Types by ROI with AI Proportion ───────────────────────
    st.subheader("Automation Types: Median ROI and AI Adoption")
    type_agg = (
        df.groupby("automation_type")
        .agg(median_roi=("roi_percent", "median"), ai_pct=("ai_flag", "mean"))
        .reset_index()
        .sort_values("median_roi", ascending=False)
        .head(20)
    )
    type_agg["AI Adoption %"] = (type_agg["ai_pct"] * 100).round(1)
    fig_type = px.bar(
        type_agg, x="automation_type", y="median_roi",
        color="AI Adoption %",
        color_continuous_scale="Blues",
        title="Top Automation Types: Median ROI with AI Adoption Rate",
        labels={"automation_type": "Automation Type", "median_roi": "Median ROI (%)"},
    )
    fig_type.update_layout(xaxis_tickangle=-30, margin=dict(t=50, b=80, l=10, r=10))
    st.plotly_chart(fig_type, use_container_width=True)

    # ── Project Duration vs ROI (Completed only) ──────────────────────────────
    st.subheader("Project Duration vs ROI (Completed Projects Only)")
    completed_df = df[
        (df["project_status"] == "Completed") &
        df["project_duration_days"].notna() &
        (df["project_duration_days"] > 0)
    ]
    if len(completed_df) > 0:
        sample_c = completed_df.sample(min(3000, len(completed_df)), random_state=42)
        fig_dur = px.scatter(
            sample_c,
            x="project_duration_days", y="roi_percent",
            color="ai_enabled",
            color_discrete_map=COLOUR_AI,
            opacity=0.5,
            labels={
                "project_duration_days": "Project Duration (Days)",
                "roi_percent": "ROI (Savings/Budget %)",
                "ai_enabled": "AI Enabled",
            },
            title="Project Duration vs ROI — Completed Projects (sample ≤ 3,000)",
            hover_data=["automation_type", "industry"],
        )
        fig_dur.update_layout(margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_dur, use_container_width=True)
        st.caption(
            f"Based on {len(completed_df):,} completed projects with valid duration data. "
            "Showing a random sample of up to 3,000 points for performance."
        )
    else:
        st.info("No completed projects with valid duration data in the current filter selection.")

    # ── Industry ROI box plot ─────────────────────────────────────────────────
    st.subheader("ROI Distribution by Industry")
    fig_box = px.box(
        df,
        x="roi_percent", y="industry",
        title="ROI (Savings/Budget %) Distribution by Industry",
        labels={"roi_percent": "ROI (Savings/Budget %)", "industry": "Industry"},
        color_discrete_sequence=[ACCENT],
    )
    fig_box.update_layout(margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig_box, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — OPPORTUNITY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def section_opportunity(df: pd.DataFrame):
    st.header("Opportunity Analysis & Investment Prioritisation")

    with st.expander("ℹ️ How is the Opportunity Score calculated?", expanded=False):
        st.markdown(
            """
**Opportunity Score** is a composite metric (0–100) computed for each group
(Automation Type or Industry) using three equally-weighted components:

| Component | Metric | Weight |
|---|---|---|
| ROI Score | Percentile rank of the group's **median ROI** among all groups | 33% |
| Savings Score | Percentile rank of the group's **total annual savings** among all groups | 33% |
| Productivity Score | Percentile rank of the group's **avg employee hours saved** among all groups | 34% |

`Opportunity Score = (ROI_Score + Savings_Score + Productivity_Score) / 3`

A higher score means the group consistently ranks well across all three dimensions simultaneously.

**High Investment / Low Return** flag is raised when:
- The group's total budget is **above median** across all groups, AND
- The group's median ROI is **below the 33rd percentile** of all groups

This is a transparent, rule-based framework — no machine learning is involved.
            """
        )

    st.markdown("---")

    # ── AUTOMATION TYPE ANALYSIS ──────────────────────────────────────────────
    st.subheader("Opportunity Score by Automation Type")
    type_scores = compute_opportunity_scores(df, "automation_type")

    # Bubble chart
    fig_bubble = px.scatter(
        type_scores,
        x="total_budget",
        y="median_roi",
        size="total_savings",
        color="opportunity_score",
        text="automation_type",
        color_continuous_scale="RdYlGn",
        size_max=60,
        title="Automation Type Opportunity Map (Bubble = Total Savings)",
        labels={
            "total_budget": "Total Budget (USD)",
            "median_roi": "Median ROI (Savings/Budget %)",
            "opportunity_score": "Opportunity Score",
            "automation_type": "Automation Type",
        },
        hover_data={
            "automation_type": True,
            "total_budget": ":,.0f",
            "median_roi": ":.1f",
            "total_savings": ":,.0f",
            "opportunity_score": ":.1f",
            "project_count": True,
        },
    )
    fig_bubble.update_traces(textposition="top center", textfont_size=9)
    fig_bubble.update_layout(margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig_bubble, use_container_width=True)
    st.caption(
        "Bubble size = total annual savings. Colour = Opportunity Score (green = high, red = low). "
        "Points in the upper-left are high-ROI relative to investment."
    )

    # Top-10 / Bottom-10 side by side
    col1, col2 = st.columns(2)
    with col1:
        top10 = type_scores.head(10)
        fig_top = px.bar(
            top10, x="opportunity_score", y="automation_type",
            orientation="h",
            title="Top 10 Automation Types by Opportunity Score",
            color="opportunity_score",
            color_continuous_scale="Greens",
            labels={"automation_type": "Automation Type", "opportunity_score": "Opportunity Score"},
        )
        fig_top.update_layout(coloraxis_showscale=False, margin=dict(t=50, b=10, l=10, r=10),
                              yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        bottom10 = type_scores.tail(10).sort_values("opportunity_score")
        fig_bot = px.bar(
            bottom10, x="opportunity_score", y="automation_type",
            orientation="h",
            title="Bottom 10 Automation Types by Opportunity Score",
            color="opportunity_score",
            color_continuous_scale="Reds_r",
            labels={"automation_type": "Automation Type", "opportunity_score": "Opportunity Score"},
        )
        fig_bot.update_layout(coloraxis_showscale=False, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_bot, use_container_width=True)

    # Scored table — Automation Type
    st.subheader("Automation Type — Full Scored Table")
    display_type = type_scores.rename(columns={
        "automation_type": "Automation Type",
        "project_count": "# Projects",
        "median_roi": "Median ROI (%)",
        "total_savings": "Total Savings (USD)",
        "total_budget": "Total Budget (USD)",
        "avg_hours_saved": "Avg Hours Saved",
        "ai_pct": "AI Adoption",
        "cloud_pct": "Cloud Adoption",
        "opportunity_score": "Opportunity Score",
        "high_invest_low_return": "⚠️ High Invest / Low Return",
    })
    display_type["Median ROI (%)"] = display_type["Median ROI (%)"].round(1)
    display_type["Total Savings (USD)"] = display_type["Total Savings (USD)"].apply(
        lambda x: f"${x:,.0f}"
    )
    display_type["Total Budget (USD)"] = display_type["Total Budget (USD)"].apply(
        lambda x: f"${x:,.0f}"
    )
    display_type["Avg Hours Saved"] = display_type["Avg Hours Saved"].round(0).astype(int)
    display_type["AI Adoption"] = (display_type["AI Adoption"] * 100).round(1).astype(str) + "%"
    display_type["Cloud Adoption"] = (display_type["Cloud Adoption"] * 100).round(1).astype(str) + "%"
    display_type["Opportunity Score"] = display_type["Opportunity Score"].round(1)
    st.dataframe(display_type, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── INDUSTRY ANALYSIS ─────────────────────────────────────────────────────
    st.subheader("Opportunity Score by Industry")
    industry_scores = compute_opportunity_scores(df, "industry")

    fig_ind_bubble = px.scatter(
        industry_scores,
        x="total_budget",
        y="median_roi",
        size="total_savings",
        color="opportunity_score",
        text="industry",
        color_continuous_scale="RdYlGn",
        size_max=60,
        title="Industry Opportunity Map (Bubble = Total Savings)",
        labels={
            "total_budget": "Total Budget (USD)",
            "median_roi": "Median ROI (Savings/Budget %)",
            "opportunity_score": "Opportunity Score",
            "industry": "Industry",
        },
        hover_data={
            "industry": True,
            "total_budget": ":,.0f",
            "median_roi": ":.1f",
            "total_savings": ":,.0f",
            "opportunity_score": ":.1f",
            "project_count": True,
        },
    )
    fig_ind_bubble.update_traces(textposition="top center", textfont_size=9)
    fig_ind_bubble.update_layout(margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig_ind_bubble, use_container_width=True)

    ind_col1, ind_col2 = st.columns(2)
    with ind_col1:
        top10_ind = industry_scores.head(10)
        fig_top_ind = px.bar(
            top10_ind, x="opportunity_score", y="industry",
            orientation="h",
            title="Top 10 Industries by Opportunity Score",
            color="opportunity_score",
            color_continuous_scale="Greens",
            labels={"industry": "Industry", "opportunity_score": "Opportunity Score"},
        )
        fig_top_ind.update_layout(coloraxis_showscale=False,
                                  margin=dict(t=50, b=10, l=10, r=10),
                                  yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_top_ind, use_container_width=True)

    with ind_col2:
        bot10_ind = industry_scores.tail(10).sort_values("opportunity_score")
        fig_bot_ind = px.bar(
            bot10_ind, x="opportunity_score", y="industry",
            orientation="h",
            title="Bottom 10 Industries by Opportunity Score",
            color="opportunity_score",
            color_continuous_scale="Reds_r",
            labels={"industry": "Industry", "opportunity_score": "Opportunity Score"},
        )
        fig_bot_ind.update_layout(coloraxis_showscale=False,
                                  margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_bot_ind, use_container_width=True)

    # Scored table — Industry
    st.subheader("Industry — Full Scored Table")
    display_ind = industry_scores.rename(columns={
        "industry": "Industry",
        "project_count": "# Projects",
        "median_roi": "Median ROI (%)",
        "total_savings": "Total Savings (USD)",
        "total_budget": "Total Budget (USD)",
        "avg_hours_saved": "Avg Hours Saved",
        "ai_pct": "AI Adoption",
        "cloud_pct": "Cloud Adoption",
        "opportunity_score": "Opportunity Score",
        "high_invest_low_return": "⚠️ High Invest / Low Return",
    })
    display_ind["Median ROI (%)"] = display_ind["Median ROI (%)"].round(1)
    display_ind["Total Savings (USD)"] = display_ind["Total Savings (USD)"].apply(
        lambda x: f"${x:,.0f}"
    )
    display_ind["Total Budget (USD)"] = display_ind["Total Budget (USD)"].apply(
        lambda x: f"${x:,.0f}"
    )
    display_ind["Avg Hours Saved"] = display_ind["Avg Hours Saved"].round(0).astype(int)
    display_ind["AI Adoption"] = (display_ind["AI Adoption"] * 100).round(1).astype(str) + "%"
    display_ind["Cloud Adoption"] = (display_ind["Cloud Adoption"] * 100).round(1).astype(str) + "%"
    display_ind["Opportunity Score"] = display_ind["Opportunity Score"].round(1)
    st.dataframe(display_ind, use_container_width=True, hide_index=True)

    # ── HIGH INVESTMENT / LOW RETURN FLAGS ────────────────────────────────────
    hilr_type = type_scores[type_scores["high_invest_low_return"]]
    hilr_ind = industry_scores[industry_scores["high_invest_low_return"]]

    if len(hilr_type) > 0 or len(hilr_ind) > 0:
        st.markdown("---")
        st.subheader("⚠️ High Investment / Low Return Areas")
        st.markdown(
            "These groups have above-median total budget **and** below-33rd-percentile median ROI. "
            "They may warrant re-evaluation of automation strategy or implementation approach."
        )
        if len(hilr_type) > 0:
            st.markdown("**Automation Types flagged:**")
            st.dataframe(
                hilr_type[["automation_type", "project_count", "median_roi",
                            "total_budget", "opportunity_score"]]
                .rename(columns={
                    "automation_type": "Automation Type",
                    "project_count": "# Projects",
                    "median_roi": "Median ROI (%)",
                    "total_budget": "Total Budget (USD)",
                    "opportunity_score": "Opportunity Score",
                }),
                use_container_width=True,
                hide_index=True,
            )
        if len(hilr_ind) > 0:
            st.markdown("**Industries flagged:**")
            st.dataframe(
                hilr_ind[["industry", "project_count", "median_roi",
                           "total_budget", "opportunity_score"]]
                .rename(columns={
                    "industry": "Industry",
                    "project_count": "# Projects",
                    "median_roi": "Median ROI (%)",
                    "total_budget": "Total Budget (USD)",
                    "opportunity_score": "Opportunity Score",
                }),
                use_container_width=True,
                hide_index=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.title("🤖 Intelligent RPA Value & Automation Opportunity Analytics")
    st.markdown(
        "Analysing **50,000 enterprise automation projects** across industries, departments, "
        "and geographies to surface performance KPIs, value drivers, and investment opportunities."
    )
    st.markdown("---")

    df = load_data()
    df_filtered = build_sidebar(df)

    if len(df_filtered) == 0:
        st.warning("No projects match the current filter selection. Please adjust the sidebar filters.")
        return

    tabs = st.tabs([
        "📊 Executive Overview",
        "⚙️ Performance Analysis",
        "💡 Value Drivers",
        "🎯 Opportunity Analysis",
    ])

    with tabs[0]:
        section_overview(df_filtered)
    with tabs[1]:
        section_performance(df_filtered)
    with tabs[2]:
        section_value(df_filtered)
    with tabs[3]:
        section_opportunity(df_filtered)


if __name__ == "__main__":
    main()
