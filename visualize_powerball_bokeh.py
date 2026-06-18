#!/usr/bin/env python3
"""
Bokeh visualization script for Powerball data and predictions.
Creates interactive charts including:
- Time series of sum_white_balls
- Frequency bar charts for white balls (1-69) and powerball (1-26)
- Comparison panel of latest predictions
"""

import pandas as pd
from bokeh.io import output_file, save
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Legend


def load_data():
    """Load all Powerball data and prediction CSV files."""
    data = pd.read_csv("powerball_data.csv")
    predictions = pd.read_csv("powerball_predictions.csv")
    enhanced_predictions = pd.read_csv("enhanced_random_forest_predictions.csv")
    historical_predictions = pd.read_csv("historical_predictions.csv")
    return data, predictions, enhanced_predictions, historical_predictions


def create_time_series_plot(data):
    """Create time series plot of sum_white_balls."""
    # Sort by date for proper time series
    data_sorted = data.sort_values("date").reset_index(drop=True)
    
    source = ColumnDataSource(data_sorted)
    
    p = figure(
        width=800,
        height=400,
        title="Powerball White Balls Sum Over Time",
        x_axis_type="datetime",
        x_axis_label="Date",
        y_axis_label="Sum of White Balls",
    )
    
    p.line(
        "date",
        "sum_white_balls",
        source=source,
        line_width=2,
        color="blue",
        legend_label="Sum",
    )
    p.circle(
        "date",
        "sum_white_balls",
        source=source,
        size=4,
        color="blue",
        alpha=0.6,
    )
    
    p.add_tools(
        HoverTool(
            tooltips=[
                ("Date", "@date"),
                ("Sum", "@sum_white_balls"),
                ("Ball 1", "@ball_1"),
                ("Ball 2", "@ball_2"),
                ("Ball 3", "@ball_3"),
                ("Ball 4", "@ball_4"),
                ("Ball 5", "@ball_5"),
            ]
        )
    )
    
    p.legend.location = "top_left"
    p.grid.grid_line_alpha = 0.3
    
    return p


def create_white_ball_frequency_plot(data):
    """Create frequency bar chart for white balls (1-69)."""
    # Count frequency of each white ball (ball_1 through ball_5)
    all_white_balls = []
    for col in ["ball_1", "ball_2", "ball_3", "ball_4", "ball_5"]:
        all_white_balls.extend(data[col].tolist())
    
    # Count frequencies for balls 1-69
    freq = {i: 0 for i in range(1, 70)}
    for ball in all_white_balls:
        if ball in freq:
            freq[ball] += 1
    
    balls = list(freq.keys())
    counts = list(freq.values())
    
    source = ColumnDataSource(data=dict(balls=balls, counts=counts))
    
    p = figure(
        width=800,
        height=400,
        title="White Ball Frequency (1-69)",
        x_axis_label="Ball Number",
        y_axis_label="Frequency",
        x_range=[1, 69],
    )
    
    p.vbar(
        x="balls",
        top="counts",
        width=0.8,
        source=source,
        color="navy",
        alpha=0.7,
    )
    
    p.add_tools(
        HoverTool(
            tooltips=[("Ball", "@balls"), ("Frequency", "@counts")]
        )
    )
    
    p.grid.grid_line_alpha = 0.3
    p.xaxis.major_label_orientation = 0.5
    
    return p


def create_powerball_frequency_plot(data):
    """Create frequency bar chart for powerball (1-26)."""
    # Count frequency of powerball
    freq = {i: 0 for i in range(1, 27)}
    for pb in data["powerball"]:
        if pd.notna(pb) and pb != "X" and str(pb).strip():
            try:
                pb_int = int(float(pb)) if isinstance(pb, str) else int(pb)
                if pb_int in freq:
                    freq[pb_int] += 1
            except (ValueError, TypeError):
                pass
    
    balls = list(freq.keys())
    counts = list(freq.values())
    
    source = ColumnDataSource(data=dict(balls=balls, counts=counts))
    
    p = figure(
        width=800,
        height=400,
        title="Powerball Frequency (1-26)",
        x_axis_label="Powerball Number",
        y_axis_label="Frequency",
        x_range=[1, 26],
    )
    
    p.vbar(
        x="balls",
        top="counts",
        width=0.8,
        source=source,
        color="red",
        alpha=0.7,
    )
    
    p.add_tools(
        HoverTool(
            tooltips=[("Powerball", "@balls"), ("Frequency", "@counts")]
        )
    )
    
    p.grid.grid_line_alpha = 0.3
    p.xaxis.major_label_orientation = 0.5
    
    return p


def create_predictions_comparison_panel(predictions, enhanced_predictions, historical_predictions):
    """Create a comparison panel showing latest predictions from different models."""
    
    # Prepare data for display
    pred_data = []
    
    # Add Fourier predictions
    if len(predictions) > 0:
        row_data = {
            "Model": predictions.iloc[0]["model"],
            "Ball 1": predictions.iloc[0]["ball_1"],
            "Ball 2": predictions.iloc[0]["ball_2"],
            "Ball 3": predictions.iloc[0]["ball_3"],
            "Ball 4": predictions.iloc[0]["ball_4"],
            "Ball 5": predictions.iloc[0]["ball_5"],
            "Powerball": predictions.iloc[0]["powerball"],
            "Device": predictions.iloc[0]["device"],
        }
        pred_data.append(row_data)
    
    # Add Enhanced Random Forest predictions
    if len(enhanced_predictions) > 0:
        row_data = {
            "Model": enhanced_predictions.iloc[0]["model"],
            "Ball 1": enhanced_predictions.iloc[0]["ball_1"],
            "Ball 2": enhanced_predictions.iloc[0]["ball_2"],
            "Ball 3": enhanced_predictions.iloc[0]["ball_3"],
            "Ball 4": enhanced_predictions.iloc[0]["ball_4"],
            "Ball 5": enhanced_predictions.iloc[0]["ball_5"],
            "Powerball": enhanced_predictions.iloc[0]["powerball"],
            "Device": "CPU" if not enhanced_predictions.iloc[0].get("use_gpu", False) else "GPU",
        }
        pred_data.append(row_data)
    
    # Add latest historical prediction
    if len(historical_predictions) > 0:
        latest_hist = historical_predictions.iloc[-1]
        row_data = {
            "Model": latest_hist["model_type"].upper(),
            "Ball 1": latest_hist["ball_1"],
            "Ball 2": latest_hist["ball_2"],
            "Ball 3": latest_hist["ball_3"],
            "Ball 4": latest_hist["ball_4"],
            "Ball 5": latest_hist["ball_5"],
            "Powerball": latest_hist.get("powerball", "N/A"),
            "Device": "N/A",
        }
        pred_data.append(row_data)
    
    # Create a simple table-like visualization
    if not pred_data:
        return None
    
    # Create a figure for the predictions table
    p = figure(
        width=800,
        height=350,
        title="Latest Predictions Comparison",
        toolbar_location=None,
        x_range=[0, 7],
        y_range=[0, len(pred_data)],
    )
    
    # Draw grid background
    p.grid.grid_line_color = "#e0e0e0"
    p.grid.grid_line_alpha = 0.5
    
    # Labels for columns
    headers = ["Model", "Ball 1", "Ball 2", "Ball 3", "Ball 4", "Ball 5", "Powerball", "Device"]
    for i, header in enumerate(headers):
        p.text(
            x=i + 0.5,
            y=len(pred_data) + 0.3,
            text=[header],
            text_font_size="11px",
            text_font_style="bold",
            text_color="black",
        )
    
    # Draw prediction rows
    colors = ["#f0f0f0", "#e8e8e8", "#e0e0e0"]
    for i, row_data in enumerate(pred_data):
        y_pos = len(pred_data) - i - 0.5
        bg_color = colors[i % len(colors)]
        
        # Draw background rectangle for row
        p.rect(
            x=3.5,
            y=y_pos,
            width=7,
            height=0.85,
            fill_color=bg_color,
            line_color="#d0d0d0",
            line_width=1,
        )
        
        # Add text for each column
        p.text(
            x=0.5,
            y=y_pos,
            text=[row_data["Model"]],
            text_font_size="10px",
            text_color="black",
        )
        p.text(
            x=1.5,
            y=y_pos,
            text=[str(row_data["Ball 1"])],
            text_font_size="10px",
            text_color="black",
        )
        p.text(
            x=2.5,
            y=y_pos,
            text=[str(row_data["Ball 2"])],
            text_font_size="10px",
            text_color="black",
        )
        p.text(
            x=3.5,
            y=y_pos,
            text=[str(row_data["Ball 3"])],
            text_font_size="10px",
            text_color="black",
        )
        p.text(
            x=4.5,
            y=y_pos,
            text=[str(row_data["Ball 4"])],
            text_font_size="10px",
            text_color="black",
        )
        p.text(
            x=5.5,
            y=y_pos,
            text=[str(row_data["Ball 5"])],
            text_font_size="10px",
            text_color="black",
        )
        p.text(
            x=6.5,
            y=y_pos,
            text=[str(row_data["Powerball"])],
            text_font_size="10px",
            text_color="black",
        )
        p.text(
            x=7.5,
            y=y_pos,
            text=[str(row_data["Device"])],
            text_font_size="10px",
            text_color="black",
        )
    
    p.axis.visible = False
    p.grid.visible = False
    
    return p


def main():
    """Main function to create and save all visualizations."""
    # Load data
    data, predictions, enhanced_predictions, historical_predictions = load_data()
    
    # Set output file
    output_file("powerball_results.html")
    
    # Create all plots
    time_series_plot = create_time_series_plot(data)
    white_ball_freq = create_white_ball_frequency_plot(data)
    powerball_freq = create_powerball_frequency_plot(data)
    predictions_panel = create_predictions_comparison_panel(
        predictions, enhanced_predictions, historical_predictions
    )
    
    # Arrange plots in layout
    # Top: time series
    # Middle: frequency charts side by side
    # Bottom: predictions comparison
    layout = column(
        time_series_plot,
        row(white_ball_freq, powerball_freq),
    )
    
    if predictions_panel:
        layout.children.append(predictions_panel)
    
    # Save the layout
    save(layout)
    
    print("Visualization saved to powerball_results.html")


if __name__ == "__main__":
    main()
