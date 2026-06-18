#!/usr/bin/env python3
"""
Interactive Bokeh BACKTEST report for the Powerball models.

This visualizes how well the Fourier and Random-Forest models *predict actual draws*,
using the project's existing walk-forward backtest engine
(``plot_time_series.PowerballTimeSeriesPlotter``): the last ``PREDICTION_DAYS`` of drawings
are held out, each is predicted from the data available before it, and the prediction is
compared to the number that was actually drawn.

It produces a single self-contained ``powerball_results.html`` with:
  1. Per-ball "actual vs predicted" series (Fourier & RF) with hover + absolute error.
  2. A grouped MAE bar chart (mean absolute error per ball, per model) — lower is better.
  3. An absolute-error heatmap (ball x held-out draw) per model.
  4. A summary panel: overall MAE, exact-match count, and within-5 hit-rate per model.

Frequency-of-numbers / sum-over-time charts are intentionally omitted: the draws are
random, so only backtest accuracy says anything about the models.

Run with the project venv (needs torch/sklearn for the models, plus bokeh):
    ./venv-powerball/bin/python visualize_powerball_bokeh.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from bokeh.io import output_file, save
from bokeh.layouts import column
from bokeh.models import (
    ColumnDataSource,
    Div,
    HoverTool,
    LinearColorMapper,
    ColorBar,
    FactorRange,
)
from bokeh.palettes import RdYlBu11
from bokeh.plotting import figure
from bokeh.transform import dodge

from plot_time_series import PowerballTimeSeriesPlotter

PREDICTION_DAYS = 60
OUT = "powerball_results.html"
BALLS = [f"ball_{i}" for i in range(1, 6)]
MODELS = [("Fourier", "fourier", "#e6550d"), ("RandomForest", "rf", "#31a354")]


def build_backtest_frame() -> pd.DataFrame:
    """Run the walk-forward backtest and return one row per held-out draw with the
    actual numbers plus each model's predicted numbers."""
    plotter = PowerballTimeSeriesPlotter("powerball_games_only.csv")
    fourier = plotter.create_fourier_predictions_series(prediction_days=PREDICTION_DAYS)
    rf = plotter.create_bin_predictions_series(prediction_days=PREDICTION_DAYS)

    actual = plotter.df[["date", *BALLS]].copy()
    df = actual[actual["date"] > actual["date"].max() - pd.Timedelta(days=PREDICTION_DAYS)]
    df = df.sort_values("date").reset_index(drop=True)

    if fourier is not None:
        df = df.merge(fourier, on="date", how="left")
    if rf is not None:
        df = df.merge(rf, on="date", how="left")

    # Per-model absolute error per ball (only where predictions exist).
    for _, key, _ in MODELS:
        for b in BALLS:
            pcol = f"{b}_{key}_pred"
            if pcol in df.columns:
                df[f"{b}_{key}_err"] = (df[b] - df[pcol]).abs()
    return df


def per_ball_figures(df: pd.DataFrame) -> list:
    figs = []
    for b in BALLS:
        src = ColumnDataSource(dict(
            date=df["date"],
            actual=df[b],
            fourier=df.get(f"{b}_fourier_pred"),
            rf=df.get(f"{b}_rf_pred"),
        ))
        p = figure(title=f"{b}: actual vs predicted (last {PREDICTION_DAYS} days)",
                   x_axis_type="datetime", height=240, width=1100,
                   tools="pan,box_zoom,wheel_zoom,reset,save", toolbar_location="right")
        p.line("date", "actual", source=src, color="#222222", line_width=2, legend_label="actual")
        p.scatter("date", "actual", source=src, color="#222222", size=6)
        if f"{b}_fourier_pred" in df.columns:
            p.scatter("date", "fourier", source=src, color="#e6550d", size=8,
                      marker="triangle", legend_label="Fourier")
        if f"{b}_rf_pred" in df.columns:
            p.scatter("date", "rf", source=src, color="#31a354", size=8,
                      marker="square", legend_label="RandomForest")
        p.add_tools(HoverTool(tooltips=[("date", "@date{%F}"), ("actual", "@actual"),
                                        ("Fourier", "@fourier"), ("RF", "@rf")],
                              formatters={"@date": "datetime"}, mode="vline"))
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.yaxis.axis_label = "number"
        figs.append(p)
    return figs


def mae_bar(df: pd.DataFrame):
    x = BALLS
    p = figure(x_range=FactorRange(*x), title="Mean Absolute Error per ball (lower = better)",
               height=320, width=560, tools="save", toolbar_location=None)
    offsets = {"fourier": -0.17, "rf": 0.17}
    for name, key, color in MODELS:
        errs = [float(df[f"{b}_{key}_err"].mean()) if f"{b}_{key}_err" in df.columns else 0.0
                for b in x]
        src = ColumnDataSource(dict(balls=x, mae=errs))
        p.vbar(x=dodge("balls", offsets[key], range=p.x_range), top="mae", width=0.3,
               source=src, color=color, legend_label=name)
    p.add_tools(HoverTool(tooltips=[("ball", "@balls"), ("MAE", "@mae{0.00}")]))
    p.y_range.start = 0
    p.yaxis.axis_label = "MAE"
    p.legend.location = "top_right"
    return p


def error_heatmaps(df: pd.DataFrame) -> list:
    figs = []
    dates = [d.strftime("%m-%d") for d in df["date"]]
    mapper = LinearColorMapper(palette=list(reversed(RdYlBu11)), low=0, high=35)
    for name, key, _ in MODELS:
        cols = [f"{b}_{key}_err" for b in BALLS]
        if not all(c in df.columns for c in cols):
            continue
        xs, ys, vals = [], [], []
        for b in BALLS:
            for di, d in enumerate(dates):
                xs.append(d); ys.append(b); vals.append(float(df[f"{b}_{key}_err"].iloc[di]))
        src = ColumnDataSource(dict(x=xs, y=ys, val=vals))
        p = figure(title=f"{name}: |actual - predicted| heatmap", x_range=dates,
                   y_range=list(reversed(BALLS)), height=300, width=1100,
                   tools="save", toolbar_location=None)
        p.rect("x", "y", width=1, height=1, source=src,
               fill_color={"field": "val", "transform": mapper}, line_color=None)
        p.add_tools(HoverTool(tooltips=[("draw", "@x"), ("ball", "@y"), ("|err|", "@val{0.0}")]))
        p.add_layout(ColorBar(color_mapper=mapper, title="abs error"), "right")
        p.xaxis.major_label_orientation = 1.0
        figs.append(p)
    return figs


def summary_div(df: pd.DataFrame) -> Div:
    rows = []
    n = len(df)
    for name, key, _ in MODELS:
        cols = [f"{b}_{key}_err" for b in BALLS]
        if not all(c in df.columns for c in cols):
            rows.append(f"<tr><td>{name}</td><td colspan=3>no predictions generated</td></tr>")
            continue
        allerr = pd.concat([df[c] for c in cols])
        mae = allerr.mean()
        exact = int((allerr == 0).sum())
        within5 = float((allerr <= 5).mean()) * 100
        rows.append(f"<tr><td>{name}</td><td>{mae:.2f}</td>"
                    f"<td>{exact} / {n * 5}</td><td>{within5:.1f}%</td></tr>")
    html = (
        f"<h1 style='font-family:sans-serif'>Powerball model backtest</h1>"
        f"<p style='font-family:sans-serif;color:#555;max-width:1080px'>Walk-forward backtest over "
        f"the last {PREDICTION_DAYS} days ({n} held-out draws). Each draw is predicted from the data "
        f"available before it, then compared to the number actually drawn. Powerball is random, so "
        f"errors are expected to be large &mdash; the point is to <i>compare the two models</i>, not "
        f"to beat the lottery.</p>"
        f"<table style='font-family:sans-serif;border-collapse:collapse' border='1' cellpadding='6'>"
        f"<tr style='background:#eee'><th>model</th><th>overall MAE</th>"
        f"<th>exact matches</th><th>within-5 hit rate</th></tr>{''.join(rows)}</table>"
    )
    return Div(text=html, width=1100)


def main():
    df = build_backtest_frame()
    layout = column(
        summary_div(df),
        mae_bar(df),
        *per_ball_figures(df),
        *error_heatmaps(df),
    )
    output_file(OUT, title="Powerball Model Backtest", mode="inline")
    save(layout)
    print(f"Wrote {OUT} ({len(df)} held-out draws backtested)")


if __name__ == "__main__":
    main()
