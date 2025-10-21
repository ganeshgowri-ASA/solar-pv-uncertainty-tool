"""
Visualization module for PV uncertainty analysis results.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class UncertaintyVisualizer:
    """
    Create visualizations for uncertainty analysis results.
    """

    @staticmethod
    def create_uncertainty_budget_chart(budget: Dict) -> go.Figure:
        """
        Create a horizontal bar chart showing uncertainty budget contributions.

        Args:
            budget: Uncertainty budget dictionary from calculator

        Returns:
            Plotly figure object
        """
        if "components" not in budget:
            return go.Figure()

        components = budget["components"]

        # Prepare data
        names = [comp["name"] for comp in components]
        contributions = [comp["percentage_contribution"] for comp in components]
        std_uncertainties = [comp["standard_uncertainty"] for comp in components]

        # Create color scale based on contribution
        colors = px.colors.sequential.Blues_r[:len(names)]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=names,
            x=contributions,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgb(8,48,107)', width=1.5)
            ),
            text=[f"{c:.1f}%" for c in contributions],
            textposition='auto',
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Contribution: %{x:.2f}%<br>" +
                "<extra></extra>"
            )
        ))

        fig.update_layout(
            title="Uncertainty Budget - Contribution by Component",
            xaxis_title="Percentage Contribution (%)",
            yaxis_title="Component",
            height=max(400, len(names) * 50),
            template="plotly_white",
            showlegend=False,
            margin=dict(l=200, r=50, t=80, b=80)
        )

        return fig

    @staticmethod
    def create_monte_carlo_histogram(
        samples: List[float],
        mean: float,
        std: float,
        confidence_intervals: Dict,
        title: str = "Monte Carlo Distribution"
    ) -> go.Figure:
        """
        Create histogram of Monte Carlo samples with statistics overlay.

        Args:
            samples: Monte Carlo sample values
            mean: Mean value
            std: Standard deviation
            confidence_intervals: Dictionary of confidence intervals
            title: Chart title

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=samples,
            name="Distribution",
            nbinsx=50,
            marker=dict(
                color='rgba(58, 71, 80, 0.6)',
                line=dict(color='rgba(58, 71, 80, 1.0)', width=1)
            ),
            hovertemplate="Value: %{x:.2f}<br>Count: %{y}<extra></extra>"
        ))

        # Add mean line
        fig.add_vline(
            x=mean,
            line=dict(color="red", width=2, dash="dash"),
            annotation_text=f"Mean: {mean:.2f}",
            annotation_position="top"
        )

        # Add 95% confidence interval
        if "95_percent" in confidence_intervals:
            ci_lower, ci_upper = confidence_intervals["95_percent"]

            fig.add_vrect(
                x0=ci_lower,
                x1=ci_upper,
                fillcolor="green",
                opacity=0.1,
                line_width=0,
                annotation_text="95% CI",
                annotation_position="top left"
            )

            fig.add_vline(
                x=ci_lower,
                line=dict(color="green", width=1, dash="dot")
            )
            fig.add_vline(
                x=ci_upper,
                line=dict(color="green", width=1, dash="dot")
            )

        fig.update_layout(
            title=title,
            xaxis_title="Value",
            yaxis_title="Frequency",
            template="plotly_white",
            showlegend=False,
            height=500
        )

        return fig

    @staticmethod
    def create_sensitivity_chart(sensitivities: Dict) -> go.Figure:
        """
        Create sensitivity analysis chart.

        Args:
            sensitivities: Dictionary of sensitivity values

        Returns:
            Plotly figure object
        """
        if not sensitivities:
            return go.Figure()

        names = list(sensitivities.keys())
        correlations = [sensitivities[name]["correlation"] for name in names]
        importances = [sensitivities[name]["importance"] for name in names]

        # Color based on positive/negative correlation
        colors = ['red' if c < 0 else 'blue' for c in correlations]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=names,
            x=correlations,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='black', width=1)
            ),
            text=[f"{c:.3f}" for c in correlations],
            textposition='auto',
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Correlation: %{x:.3f}<br>" +
                "<extra></extra>"
            )
        ))

        fig.update_layout(
            title="Sensitivity Analysis - Input Correlation with Output",
            xaxis_title="Correlation Coefficient",
            yaxis_title="Input Variable",
            height=max(400, len(names) * 50),
            template="plotly_white",
            showlegend=False,
            margin=dict(l=200, r=50, t=80, b=80),
            xaxis=dict(range=[-1.1, 1.1])
        )

        # Add vertical line at zero
        fig.add_vline(x=0, line=dict(color="black", width=1))

        return fig

    @staticmethod
    def create_comparison_chart(
        gum_result: float,
        gum_uncertainty: float,
        mc_result: float,
        mc_uncertainty: float,
        label: str = "Value"
    ) -> go.Figure:
        """
        Create comparison chart between GUM and Monte Carlo methods.

        Args:
            gum_result: GUM calculated result
            gum_uncertainty: GUM uncertainty
            mc_result: Monte Carlo result
            mc_uncertainty: Monte Carlo uncertainty
            label: Label for the value

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        methods = ["GUM Method", "Monte Carlo"]
        values = [gum_result, mc_result]
        uncertainties = [gum_uncertainty, mc_uncertainty]

        # Error bars
        fig.add_trace(go.Bar(
            x=methods,
            y=values,
            error_y=dict(
                type='data',
                array=uncertainties,
                visible=True,
                color='red',
                thickness=2,
                width=10
            ),
            marker=dict(
                color=['rgba(58, 71, 80, 0.8)', 'rgba(46, 134, 193, 0.8)'],
                line=dict(color='black', width=1.5)
            ),
            text=[f"{v:.2f} ± {u:.2f}" for v, u in zip(values, uncertainties)],
            textposition='outside',
            hovertemplate=(
                "<b>%{x}</b><br>" +
                f"{label}: " + "%{y:.4f}<br>" +
                "Uncertainty: %{error_y.array:.4f}<br>" +
                "<extra></extra>"
            )
        ))

        fig.update_layout(
            title=f"Comparison: GUM vs Monte Carlo Methods",
            yaxis_title=label,
            template="plotly_white",
            showlegend=False,
            height=500
        )

        return fig

    @staticmethod
    def create_confidence_interval_plot(
        value: float,
        confidence_intervals: Dict,
        label: str = "Measured Value"
    ) -> go.Figure:
        """
        Create a visual representation of confidence intervals.

        Args:
            value: Central value
            confidence_intervals: Dictionary with confidence intervals
            label: Value label

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        # Prepare data for different confidence levels
        levels = []
        lower_bounds = []
        upper_bounds = []

        if "68_percent" in confidence_intervals:
            levels.append("68%")
            lower_bounds.append(confidence_intervals["68_percent"][0])
            upper_bounds.append(confidence_intervals["68_percent"][1])

        if "95_percent" in confidence_intervals:
            levels.append("95%")
            lower_bounds.append(confidence_intervals["95_percent"][0])
            upper_bounds.append(confidence_intervals["95_percent"][1])

        if "99_percent" in confidence_intervals:
            levels.append("99%")
            lower_bounds.append(confidence_intervals["99_percent"][0])
            upper_bounds.append(confidence_intervals["99_percent"][1])

        # Create error bars for each level
        colors = ['rgba(46, 134, 193, 0.8)', 'rgba(26, 188, 156, 0.8)', 'rgba(241, 196, 15, 0.8)']

        for i, (level, lower, upper) in enumerate(zip(levels, lower_bounds, upper_bounds)):
            error = upper - value
            fig.add_trace(go.Scatter(
                x=[value],
                y=[i],
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=[error],
                    arrayminus=[value - lower],
                    color=colors[i],
                    thickness=8,
                    width=0
                ),
                mode='markers',
                marker=dict(size=12, color=colors[i], line=dict(width=2, color='black')),
                name=f"{level} CI",
                hovertemplate=(
                    f"<b>{level} Confidence Interval</b><br>" +
                    f"{label}: {value:.4f}<br>" +
                    f"Lower: {lower:.4f}<br>" +
                    f"Upper: {upper:.4f}<br>" +
                    "<extra></extra>"
                )
            ))

        fig.update_layout(
            title=f"Confidence Intervals for {label}",
            xaxis_title=label,
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(levels))),
                ticktext=levels
            ),
            template="plotly_white",
            height=300,
            showlegend=True,
            legend=dict(x=1.02, y=1)
        )

        return fig

    @staticmethod
    def create_uncertainty_breakdown_pie(budget: Dict) -> go.Figure:
        """
        Create pie chart showing uncertainty budget breakdown.

        Args:
            budget: Uncertainty budget dictionary

        Returns:
            Plotly figure object
        """
        if "components" not in budget:
            return go.Figure()

        components = budget["components"]

        names = [comp["name"] for comp in components]
        contributions = [comp["percentage_contribution"] for comp in components]

        fig = go.Figure(data=[go.Pie(
            labels=names,
            values=contributions,
            hole=0.3,
            marker=dict(
                line=dict(color='white', width=2)
            ),
            textinfo='label+percent',
            hovertemplate=(
                "<b>%{label}</b><br>" +
                "Contribution: %{value:.2f}%<br>" +
                "<extra></extra>"
            )
        )])

        fig.update_layout(
            title="Uncertainty Budget Distribution",
            template="plotly_white",
            height=500
        )

        return fig

    @staticmethod
    def create_combined_dashboard(
        budget: Dict,
        mc_samples: Optional[List[float]] = None,
        mc_stats: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create a combined dashboard with multiple visualizations.

        Args:
            budget: Uncertainty budget
            mc_samples: Monte Carlo samples (optional)
            mc_stats: Monte Carlo statistics (optional)

        Returns:
            Plotly figure object with subplots
        """
        # Determine subplot layout
        if mc_samples and mc_stats:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Uncertainty Budget",
                    "Monte Carlo Distribution",
                    "Contribution Breakdown",
                    "Sensitivity Analysis"
                ),
                specs=[
                    [{"type": "bar"}, {"type": "histogram"}],
                    [{"type": "pie"}, {"type": "bar"}]
                ],
                vertical_spacing=0.15,
                horizontal_spacing=0.12
            )
        else:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Uncertainty Budget", "Contribution Breakdown"),
                specs=[[{"type": "bar"}, {"type": "pie"}]],
                horizontal_spacing=0.15
            )

        # Add uncertainty budget
        if "components" in budget:
            components = budget["components"]
            names = [comp["name"] for comp in components]
            contributions = [comp["percentage_contribution"] for comp in components]

            fig.add_trace(
                go.Bar(
                    y=names,
                    x=contributions,
                    orientation='h',
                    marker=dict(color='lightblue'),
                    showlegend=False
                ),
                row=1, col=1
            )

            # Add pie chart
            if mc_samples and mc_stats:
                fig.add_trace(
                    go.Pie(labels=names, values=contributions, hole=0.3, showlegend=False),
                    row=2, col=1
                )
            else:
                fig.add_trace(
                    go.Pie(labels=names, values=contributions, hole=0.3, showlegend=False),
                    row=1, col=2
                )

        # Add Monte Carlo histogram
        if mc_samples and mc_stats:
            fig.add_trace(
                go.Histogram(x=mc_samples, nbinsx=50, marker=dict(color='lightcoral'), showlegend=False),
                row=1, col=2
            )

            # Add sensitivity if available
            if "sensitivities" in mc_stats:
                sens = mc_stats["sensitivities"]
                sens_names = list(sens.keys())
                sens_corr = [sens[name]["correlation"] for name in sens_names]

                fig.add_trace(
                    go.Bar(
                        y=sens_names,
                        x=sens_corr,
                        orientation='h',
                        marker=dict(color='lightgreen'),
                        showlegend=False
                    ),
                    row=2, col=2
                )

        fig.update_layout(
            title_text="PV Uncertainty Analysis Dashboard",
            template="plotly_white",
            height=800,
            showlegend=False
        )

        return fig
