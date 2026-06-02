"""
验证可视化图表组件

包含 IC 分析、分组测试等图表函数。
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


def plot_ic_timeseries(ic_series: pd.Series, title: str = "IC Time Series"):
    """
    IC 时间序列图

    Args:
        ic_series: IC 时间序列
        title: 图表标题

    Returns:
        plotly Figure 对象
    """
    fig = go.Figure()

    # IC 曲线
    fig.add_trace(go.Scatter(
        x=ic_series.index,
        y=ic_series.values,
        mode='lines',
        name='IC',
        line=dict(color='#1f77b4', width=1),
        hovertemplate='Date: %{x}<br>IC: %{y:.4f}<extra></extra>',
    ))

    # 零线
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # 均值线
    ic_mean = ic_series.mean()
    fig.add_hline(
        y=ic_mean,
        line_dash="dot",
        line_color="#2ca02c",
        annotation_text=f"Mean: {ic_mean:.4f}",
        annotation_position="right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="IC",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )

    return fig


def plot_ic_distribution(ic_series: pd.Series, title: str = "IC Distribution"):
    """
    IC 分布直方图

    Args:
        ic_series: IC 时间序列
        title: 图表标题

    Returns:
        plotly Figure 对象
    """
    ic_clean = ic_series.dropna()

    fig = go.Figure()

    # 直方图
    fig.add_trace(go.Histogram(
        x=ic_clean.values,
        nbinsx=30,
        marker_color='#1f77b4',
        opacity=0.7,
        name='IC Distribution',
    ))

    # 均值线
    ic_mean = ic_clean.mean()
    ic_std = ic_clean.std()

    fig.add_vline(x=ic_mean, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {ic_mean:.4f}")

    fig.update_layout(
        title=title,
        xaxis_title="IC",
        yaxis_title="Count",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        annotations=[
            dict(
                x=0.95, y=0.95,
                xref="paper", yref="paper",
                text=f"Std: {ic_std:.4f}<br>N: {len(ic_clean)}",
                showarrow=False,
                align="right",
            )
        ]
    )

    return fig


def plot_ic_decay(ic_decay: pd.Series, title: str = "IC Decay by Holding Period"):
    """
    IC 衰减曲线

    Args:
        ic_decay: 各持有期的 IC 值
        title: 图表标题

    Returns:
        plotly Figure 对象
    """
    fig = go.Figure()

    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in ic_decay.values]

    fig.add_trace(go.Bar(
        x=[f"{p}d" for p in ic_decay.index],
        y=ic_decay.values,
        marker_color=colors,
        text=[f"{v:.4f}" for v in ic_decay.values],
        textposition='outside',
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=0.02, line_dash="dot", line_color="orange", opacity=0.5,
                  annotation_text="Threshold (0.02)")

    fig.update_layout(
        title=title,
        xaxis_title="Holding Period",
        yaxis_title="IC",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )

    return fig


def plot_ic_cumulative(ic_series: pd.Series, title: str = "Cumulative IC"):
    """
    IC 累积曲线

    Args:
        ic_series: IC 时间序列
        title: 图表标题

    Returns:
        plotly Figure 对象
    """
    cumsum = ic_series.cumsum()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cumsum.index,
        y=cumsum.values,
        mode='lines',
        fill='tozeroy',
        marker_color='#ff7f0e',
        name='Cumulative IC',
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative IC",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )

    return fig


def plot_group_returns(group_returns: pd.Series, title: str = "Group Returns"):
    """
    分组收益柱状图

    Args:
        group_returns: 各组收益
        title: 图表标题

    Returns:
        plotly Figure 对象
    """
    fig = go.Figure()

    colors = ['#d62728' if r < 0 else '#2ca02c' for r in group_returns.values]

    fig.add_trace(go.Bar(
        x=[f"Q{i+1}" for i in group_returns.index],
        y=group_returns.values,
        marker_color=colors,
        text=[f"{v:.2%}" for v in group_returns.values],
        textposition='outside',
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Quantile Group",
        yaxis_title="Return",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )

    return fig


def plot_long_short_nav(
    long_short_returns: pd.Series,
    title: str = "Long-Short NAV Curve"
):
    """
    多空净值曲线

    Args:
        long_short_returns: 多空收益序列
        title: 图表标题

    Returns:
        plotly Figure 对象
    """
    nav = (1 + long_short_returns).cumprod()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=nav.index,
        y=nav.values,
        mode='lines',
        name='Long-Short NAV',
        line=dict(color='#1f77b4', width=2),
    ))

    # 基准线
    fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="NAV",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )

    return fig


def plot_validation_summary(
    ic_result: Optional[Dict] = None,
    group_result: Optional[Dict] = None,
):
    """
    验证结果汇总图

    Args:
        ic_result: IC 分析结果
        group_result: 分组测试结果

    Returns:
        plotly Figure 对象
    """
    if ic_result is None and group_result is None:
        return go.Figure()

    # 创建子图
    n_rows = 2 if ic_result and group_result else 1
    n_cols = 2 if ic_result else 1

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    row = 1
    col = 1

    if ic_result:
        # IC 指标
        metrics = {
            "IC Mean": ic_result.get("ic_mean", 0),
            "IC IR": ic_result.get("ic_ir", 0),
            "IC t-stat": ic_result.get("ic_tstat", 0),
            "Positive Ratio": ic_result.get("positive_ratio", 0),
        }

        fig.add_trace(go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color=['#2ca02c' if v > 0 else '#d62728' for v in metrics.values()],
        ), row=row, col=col)

        row = 1
        col = 2

    if group_result:
        # 分组收益
        group_rets = group_result.get("group_returns", {})
        if isinstance(group_rets, dict):
            groups = list(group_rets.keys())
            values = list(group_rets.values())
        else:
            groups = list(range(len(group_rets)))
            values = list(group_rets)

        fig.add_trace(go.Bar(
            x=[f"Q{g+1}" for g in groups],
            y=values,
            marker_color=['#2ca02c' if v > 0 else '#d62728' for v in values],
        ), row=row, col=col)

    fig.update_layout(
        title="Validation Summary",
        height=400 * n_rows,
        showlegend=False,
    )

    return fig


def create_validation_dashboard(
    ic_series: Optional[pd.Series] = None,
    ic_decay: Optional[pd.Series] = None,
    group_returns: Optional[pd.Series] = None,
    long_short_returns: Optional[pd.Series] = None,
):
    """
    创建完整的验证仪表板

    Args:
        ic_series: IC 时间序列
        ic_decay: IC 衰减数据
        group_returns: 分组收益
        long_short_returns: 多空收益

    Returns:
        plotly Figure 对象
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("IC Time Series", "IC Distribution", "Group Returns", "Long-Short NAV"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    if ic_series is not None:
        # IC 时间序列
        fig.add_trace(go.Scatter(
            x=ic_series.index,
            y=ic_series.values,
            mode='lines',
            name='IC',
            line=dict(color='#1f77b4', width=1),
        ), row=1, col=1)

        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

        # IC 分布
        fig.add_trace(go.Histogram(
            x=ic_series.dropna().values,
            nbinsx=30,
            marker_color='#1f77b4',
            opacity=0.7,
            name='IC Dist',
        ), row=1, col=2)

    if group_returns is not None:
        # 分组收益
        colors = ['#d62728' if r < 0 else '#2ca02c' for r in group_returns.values]
        fig.add_trace(go.Bar(
            x=[f"Q{i+1}" for i in group_returns.index],
            y=group_returns.values,
            marker_color=colors,
            name='Group Ret',
        ), row=2, col=1)

    if long_short_returns is not None:
        # 多空净值
        nav = (1 + long_short_returns).cumprod()
        fig.add_trace(go.Scatter(
            x=nav.index,
            y=nav.values,
            mode='lines',
            name='LS NAV',
            line=dict(color='#ff7f0e', width=1),
        ), row=2, col=2)

    fig.update_layout(
        title="Factor Validation Dashboard",
        height=700,
        showlegend=False,
    )

    return fig
