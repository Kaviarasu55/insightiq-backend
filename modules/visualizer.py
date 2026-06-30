import pandas as pd
import numpy as np


def generate_chart_data(df, column_analysis):
    # Main function — looks at each column and generates
    # appropriate chart data based on column type
    # Returns list of chart objects ready for React + Recharts

    charts = []

    # Find numeric and categorical columns
    numeric_cols = [c for c in column_analysis if c["type"] == "numeric"]
    categorical_cols = [c for c in column_analysis if c["type"] == "categorical"]
    datetime_col = detect_time_column(df)

    # Numeric columns → histogram + boxplot data
    for col in numeric_cols:
        name = col["name"]
        series = df[name].dropna()

        if len(series) == 0:
            continue

        # Histogram data — divide into 20 bins
        hist_data = get_histogram_data(series, name)
        charts.append(
            {
                "chart_type": "histogram",
                "column": name,
                "data": hist_data,
            }
        )

        # Box plot data — 5-number summary
        box_data = get_boxplot_data(series, name)
        charts.append(
            {
                "chart_type": "boxplot",
                "column": name,
                "data": box_data,
            }
        )

    # Categorical columns → bar chart
    for col in categorical_cols:
        name = col["name"]
        series = df[name].dropna()

        if len(series) == 0:
            continue

        bar_data = get_bar_data(series, name)
        charts.append(
            {
                "chart_type": "bar",
                "column": name,
                "data": bar_data,
            }
        )

    # First two numeric columns → scatter plot
    if len(numeric_cols) >= 2:
        x_col = numeric_cols[0]["name"]
        y_col = numeric_cols[1]["name"]
        scatter_data = get_scatter_data(df, x_col, y_col)
        charts.append(
            {
                "chart_type": "scatter",
                "column": f"{x_col} vs {y_col}",
                "x_col": x_col,
                "y_col": y_col,
                "data": scatter_data,
            }
        )

    # Time column detected → line chart with first numeric column
    if datetime_col and len(numeric_cols) > 0:
        y_col = numeric_cols[0]["name"]
        line_data = get_line_data(df, datetime_col, y_col)
        charts.append(
            {
                "chart_type": "line",
                "column": f"{datetime_col} vs {y_col}",
                "x_col": datetime_col,
                "y_col": y_col,
                "data": line_data,
            }
        )

    return charts


def get_histogram_data(series, col_name):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return []
    counts, bin_edges = np.histogram(series, bins=20)
    result = []
    for i in range(len(counts)):
        # Label each bin with its range
        label = f"{round(bin_edges[i], 2)}–{round(bin_edges[i+1], 2)}"
        result.append({"bin": label, "count": int(counts[i])})
    return result


def get_boxplot_data(series, col_name):
    # 5-number summary for box plot
    return {
        "min": round(float(series.min()), 4),
        "q1": round(float(series.quantile(0.25)), 4),
        "median": round(float(series.median()), 4),
        "q3": round(float(series.quantile(0.75)), 4),
        "max": round(float(series.max()), 4),
    }


def get_bar_data(series, col_name, df=None, y_col=None):
    # If y_col is provided — aggregate Y values grouped by X categories
    # e.g., X=Player, Y=Runs → shows total runs per player (sum)
    # If no y_col — just count frequency of each category (original behavior)

    if df is not None and y_col and y_col in df.columns:
        try:
            # Group by X column and sum the Y column
            grouped = df.groupby(col_name)[y_col].sum().sort_values(ascending=False).head(15)
            return [
                {"category": str(k), "value": round(float(v), 2)}
                for k, v in grouped.items()
            ]
        except Exception:
            # If aggregation fails (e.g., Y is not numeric), fall back to count
            pass

    # Default: count frequency of each category
    counts = series.value_counts().head(15)
    return [{"category": str(k), "count": int(v)} for k, v in counts.items()]


def get_scatter_data(df, x_col, y_col):
    # Sample max 500 rows to keep payload light
    temp = df[[x_col, y_col]].dropna()
    if len(temp) > 500:
        temp = temp.sample(500, random_state=42)
    return [
        {"x": round(float(row[x_col]), 4), "y": round(float(row[y_col]), 4)}
        for _, row in temp.iterrows()
    ]


def get_line_data(df, time_col, y_col):
    # Sort by time column, return x/y pairs for line chart
    temp = df[[time_col, y_col]].dropna().copy()
    temp[time_col] = pd.to_datetime(temp[time_col])
    temp = temp.sort_values(time_col)
    # Sample if too many rows
    if len(temp) > 500:
        temp = temp.sample(500, random_state=42).sort_values(time_col)
    return [
        {"x": str(row[time_col]), "y": round(float(row[y_col]), 4)}
        for _, row in temp.iterrows()
    ]


def detect_time_column(df):
    # Look for columns with common datetime keywords in name
    keywords = ["date", "time", "month", "year", "day", "timestamp"]
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    return None


def get_manual_chart_data(df, chart_type, x_col, y_col=None):
    # For user-selected manual charts on Page 4
    # chart_type: "bar", "histogram", "scatter", "line"

    if chart_type == "histogram":
        series = df[x_col].dropna()
        return get_histogram_data(series, x_col)

    elif chart_type == "bar":
        series = df[x_col].dropna()
        # Pass df and y_col so bar chart can aggregate Y values per X category
        return get_bar_data(series, x_col, df=df, y_col=y_col)

    elif chart_type == "scatter" and y_col:
        return get_scatter_data(df, x_col, y_col)

    elif chart_type == "line" and y_col:
        return get_line_data(df, x_col, y_col)

    return []
