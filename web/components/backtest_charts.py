"""
回测可视化图表组件

包含净值曲线、回撤曲线、绩效表格等。
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List


def plot_nav_curve(
    nav_curve: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "NAV Curve"
):
    """
    净值曲线图

    Args:
        nav_curve: 净值曲线
        benchmark: 基准净值曲线（可选）
        title: 图表标题

    Returns:
        plotly Figure 对象
    """
    # 确保数据为数值类型
    nav_curve = pd.to_numeric(nav_curve, errors='coerce').dropna()

    if len(nav_curve) == 0:
        fig = go.Figure()
        fig.add_annotation(text="无有效数据", showarrow=False)
        return fig

    fig = go.Figure()

    # 策略净值
    fig.add_trace(go.Scatter(
        x=nav_curve.index,
        y=nav_curve.values,
        mode='lines',
        name='Factor Strategy',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Date: %{x}<br>NAV: %{y:.4f}<extra></extra>',
    ))

    # 基准净值
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index,
            y=benchmark.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='gray', width=1, dash='dash'),
        ))

    fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.3)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="NAV",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def plot_drawdown(nav_curve: pd.Series, title: str = "Drawdown"):
    """
    回撤曲线

    Args:
        nav_curve: 净值曲线
        title: 图表标题

    Returns:
        plotly Figure 对象
    """
    # 确保数据为数值类型
    nav_curve = pd.to_numeric(nav_curve, errors='coerce').dropna()

    if len(nav_curve) == 0:
        fig = go.Figure()
        fig.add_annotation(text="无有效数据", showarrow=False)
        return fig

    rolling_max = nav_curve.cummax()
    drawdown = (nav_curve - rolling_max) / rolling_max

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill='tozeroy',
        marker_color='#d62728',
        name='Drawdown',
        hovertemplate='Date: %{x}<br>Drawdown: %{y:.2%}<extra></extra>',
    ))

    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()

    fig.add_trace(go.Scatter(
        x=[max_dd_date],
        y=[max_dd],
        mode='markers',
        marker=dict(size=10, color='red'),
        name=f'Max DD: {max_dd:.2%}',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis_tickformat='.2%',
    )

    return fig


def plot_monthly_returns(returns: pd.Series, title: str = "Monthly Returns Heatmap"):
    """
    月度收益热力图

    Args:
        returns: 日收益率序列
        title: 图表标题

    Returns:
        plotly Figure 对象
    """
    # 确保索引是日期类型
    if not isinstance(returns.index, pd.DatetimeIndex):
        try:
            returns = returns.copy()
            returns.index = pd.to_datetime(returns.index)
        except Exception:
            # 如果转换失败，返回空图
            fig = go.Figure()
            fig.add_annotation(text="无法创建月度收益图：日期索引无效", showarrow=False)
            return fig

    # 确保数据为数值类型
    returns = pd.to_numeric(returns, errors='coerce').dropna()

    if len(returns) == 0:
        fig = go.Figure()
        fig.add_annotation(text="无有效数据", showarrow=False)
        return fig

    # 计算月度收益
    monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    # 转换为年月矩阵
    monthly_df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values,
    })

    pivot = monthly_df.pivot(index='year', columns='month', values='return')

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=month_names,
        y=pivot.index,
        colorscale=[
            [0, '#d62728'],
            [0.5, '#ffffff'],
            [1, '#2ca02c'],
        ],
        zmid=0,
        text=[[f"{v:.2%}" if not pd.isna(v) else '' for v in row] for row in pivot.values],
        texttemplate='%{text}',
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2%}<extra></extra>',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Year",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def plot_group_nav_curves(group_navs: Dict[int, pd.Series], title: str = "Group NAV Curves"):
    """
    各组净值曲线

    Args:
        group_navs: 各组净值曲线字典
        title: 图表标题

    Returns:
        plotly Figure 对象
    """
    fig = go.Figure()

    colors = ['#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#1f77b4',
              '#2ca02c', '#17becf', '#bcbd22', '#7f7f7f', '#e377c2']

    for group_id, nav in sorted(group_navs.items()):
        # 确保数据为数值类型
        nav = pd.to_numeric(nav, errors='coerce').dropna()
        if len(nav) == 0:
            continue

        fig.add_trace(go.Scatter(
            x=nav.index,
            y=nav.values,
            mode='lines',
            name=f'Q{group_id + 1}',
            line=dict(color=colors[group_id % len(colors)], width=1),
        ))

    fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.3)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="NAV",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def plot_performance_table(stats: Dict[str, float]):
    """
    绩效指标表格

    Args:
        stats: 绩效统计字典

    Returns:
        plotly Figure 对象
    """
    metrics = [
        ("Annual Return", stats.get("annual_return", 0), ".2%"),
        ("Annual Volatility", stats.get("annual_volatility", 0), ".2%"),
        ("Sharpe Ratio", stats.get("sharpe_ratio", 0), ".2f"),
        ("Max Drawdown", stats.get("max_drawdown", 0), ".2%"),
        ("Calmar Ratio", stats.get("calmar_ratio", 0), ".2f"),
        ("Win Rate", stats.get("win_rate", 0), ".2%"),
        ("Sortino Ratio", stats.get("sortino_ratio", 0), ".2f"),
        ("Avg Daily Return", stats.get("avg_daily_return", 0), ".4%"),
    ]

    fig = go.Figure()

    fig.add_trace(go.Table(
        header=dict(
            values=["Metric", "Value"],
            fill_color='#1f77b4',
            font=dict(color='white', size=12),
            align='left',
        ),
        cells=dict(
            values=[[m[0] for m in metrics], [f"{m[1]:{m[2]}}" for m in metrics]],
            fill_color='#f8f9fa',
            font=dict(color='#333333', size=11),  # 深色字体确保可读性
            align='left',
        ),
    ))

    fig.update_layout(
        title="Performance Statistics",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',  # 透明背景
    )

    return fig


def plot_full_backtest_report(
    nav_curve: pd.Series,
    returns: pd.Series,
    stats: Dict[str, float],
    benchmark: Optional[pd.Series] = None,
):
    """
    完整的回测报告图

    Args:
        nav_curve: 净值曲线
        returns: 日收益率
        stats: 绩效统计
        benchmark: 基准净值（可选）

    Returns:
        plotly Figure 对象
    """
    # 确保数据为数值类型
    nav_curve = pd.to_numeric(nav_curve, errors='coerce').dropna()
    returns = pd.to_numeric(returns, errors='coerce').dropna()

    # 确保索引是日期类型
    if not isinstance(returns.index, pd.DatetimeIndex):
        try:
            returns = returns.copy()
            returns.index = pd.to_datetime(returns.index)
        except Exception:
            pass

    if not isinstance(nav_curve.index, pd.DatetimeIndex):
        try:
            nav_curve = nav_curve.copy()
            nav_curve.index = pd.to_datetime(nav_curve.index)
        except Exception:
            pass

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=("NAV Curve", "Drawdown", "Monthly Returns", "Daily Returns Distribution", "Cumulative Returns", "Performance Stats"),
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "heatmap"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "table"}],
        ],
    )

    # 净值曲线
    fig.add_trace(go.Scatter(
        x=nav_curve.index,
        y=nav_curve.values,
        mode='lines',
        name='NAV',
        line=dict(color='#1f77b4', width=2),
    ), row=1, col=1)

    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index,
            y=benchmark.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='gray', width=1, dash='dash'),
        ), row=1, col=1)

    # 回撤
    rolling_max = nav_curve.cummax()
    drawdown = (nav_curve - rolling_max) / rolling_max
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill='tozeroy',
        marker_color='#d62728',
        name='Drawdown',
    ), row=1, col=2)

    # 月度收益热力图
    try:
        if isinstance(returns.index, pd.DatetimeIndex):
            monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            monthly_df = pd.DataFrame({
                'year': monthly_returns.index.year,
                'month': monthly_returns.index.month,
                'return': monthly_returns.values,
            })
            pivot = monthly_df.pivot(index='year', columns='month', values='return')

            fig.add_trace(go.Heatmap(
                z=pivot.values,
                colorscale=[[0, '#d62728'], [0.5, '#ffffff'], [1, '#2ca02c']],
                zmid=0,
            ), row=2, col=1)
        else:
            # 如果不是日期索引，添加空的热力图
            fig.add_trace(go.Heatmap(
                z=[[0]],
                colorscale=[[0, '#d62728'], [0.5, '#ffffff'], [1, '#2ca02c']],
                zmid=0,
            ), row=2, col=1)
    except Exception:
        fig.add_trace(go.Heatmap(
            z=[[0]],
            colorscale=[[0, '#d62728'], [0.5, '#ffffff'], [1, '#2ca02c']],
            zmid=0,
        ), row=2, col=1)

    # 日收益分布
    fig.add_trace(go.Histogram(
        x=returns.dropna().values,
        nbinsx=50,
        marker_color='#1f77b4',
        opacity=0.7,
    ), row=2, col=2)

    # 累积收益
    cum_returns = (1 + returns).cumprod() - 1
    fig.add_trace(go.Scatter(
        x=cum_returns.index,
        y=cum_returns.values,
        mode='lines',
        fill='tozeroy',
        marker_color='#2ca02c',
    ), row=3, col=1)

    # 绩效表格
    metrics = [
        ("Annual Return", stats.get("annual_return", 0)),
        ("Sharpe Ratio", stats.get("sharpe_ratio", 0)),
        ("Max Drawdown", stats.get("max_drawdown", 0)),
        ("Win Rate", stats.get("win_rate", 0)),
    ]

    fig.add_trace(go.Table(
        header=dict(values=["Metric", "Value"], fill_color='#1f77b4', font=dict(color='white')),
        cells=dict(
            values=[
                [m[0] for m in metrics],
                [f"{m[1]:.2%}" if abs(m[1]) < 10 else f"{m[1]:.2f}" for m in metrics]
            ],
            fill_color='#f8f9fa',
            font=dict(color='#333333'),
        ),
    ), row=3, col=2)

    fig.update_layout(
        title="Factor Backtest Report",
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def compute_performance_stats(returns: pd.Series) -> Dict[str, float]:
    """
    计算绩效统计指标

    Args:
        returns: 日收益率序列

    Returns:
        绩效统计字典
    """
    # 确保输入是 Series 且为数值类型
    if isinstance(returns, pd.DataFrame):
        # 如果是 DataFrame，取第一列
        returns = returns.iloc[:, 0]

    # 转换为数值类型
    returns_clean = pd.to_numeric(returns, errors='coerce').dropna()

    if len(returns_clean) == 0:
        return {}

    # 确保索引是日期类型（用于 resample）
    if not isinstance(returns_clean.index, pd.DatetimeIndex):
        try:
            returns_clean.index = pd.to_datetime(returns_clean.index)
        except Exception:
            pass  # 如果转换失败，继续使用原索引

    # 年化收益
    annual_return = float(returns_clean.mean()) * 252

    # 年化波动率
    annual_vol = float(returns_clean.std()) * np.sqrt(252)

    # 夏普比率
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    # 最大回撤
    nav = (1 + returns_clean).cumprod()
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    # Calmar 比率
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # 胜率
    win_rate = float((returns_clean > 0).mean())

    # Sortino 比率
    downside_returns = returns_clean[returns_clean < 0]
    downside_vol = float(downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = annual_return / downside_vol if downside_vol > 0 else 0

    # 日均收益
    avg_daily_return = float(returns_clean.mean())

    # 总收益
    total_return = float((1 + returns_clean).prod()) - 1

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "sortino_ratio": sortino,
        "avg_daily_return": avg_daily_return,
        "total_return": total_return,
        "n_days": len(returns_clean),
    }