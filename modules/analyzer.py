import pandas as pd
import numpy as np

def analyze_csv(file):
    # Read CSV into DataFrame
    # file is the uploaded file object from Flask request
    df = pd.read_csv(file)

    row_count = len(df)           # total number of rows
    col_count = len(df.columns)   # total number of columns

    column_analysis = []

    for col in df.columns:
        # Start building info dict for this column
        info = {
            "name": col,
            "null_count": int(df[col].isnull().sum()),
            "null_percentage": round(df[col].isnull().sum() / row_count * 100, 2),
            "unique_count": int(df[col].nunique()),
        }

        # Detect column type and calculate type-specific stats
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numeric column — calculate descriptive stats
            info["type"] = "numeric"
            info["mean"] = round(float(df[col].mean()), 4) if not df[col].isnull().all() else None
            info["median"] = round(float(df[col].median()), 4) if not df[col].isnull().all() else None
            info["min"] = round(float(df[col].min()), 4) if not df[col].isnull().all() else None
            info["max"] = round(float(df[col].max()), 4) if not df[col].isnull().all() else None
            info["std"] = round(float(df[col].std()), 4) if not df[col].isnull().all() else None
            # Summary string for Groq context injection
            info["summary"] = f"mean={info['mean']}, min={info['min']}, max={info['max']}"

        elif is_datetime_column(df[col]):
            # Datetime column
            info["type"] = "datetime"
            info["summary"] = "datetime column"

        else:
            # Categorical column — get top 3 most frequent values
            info["type"] = "categorical"
            top = df[col].value_counts().head(3)
            # Convert to list of [value, count] pairs
            info["top_values"] = [{"value":str(k), "count":int(v)} for k, v in top.items()]
            info["summary"] = f"top values: {[str(k) for k in top.index.tolist()]}"

        column_analysis.append(info)

    # First 10 rows as list of dicts — for preview table in React
    # replace NaN with None so it converts to JSON null cleanly
    preview = df.head(10).replace({np.nan: None}).to_dict("records")

    # First 5 rows as list of dicts — for Groq context injection
    sample_rows = df.head(5).replace({np.nan: None}).to_dict("records")

    return {
        "row_count": row_count,
        "col_count": col_count,
        "columns": column_analysis,
        "preview": preview,
        "sample_rows": sample_rows,
    }

def is_datetime_column(series):
    # Try to parse a sample of values as datetime
    # If most succeed → it's a datetime column
    if series.dtype == "object":
        try:
            sample = series.dropna().head(10)
            pd.to_datetime(sample)
            return True
        except Exception:
            return False
    return False

def get_column_info_for_groq(column_analysis):
    # Returns a clean list of dicts for Groq context injection
    # Only includes name, type, summary — no heavy stats
    return [
        {
            "name": col["name"],
            "type": col["type"],
            "summary": col.get("summary", "")
        }
        for col in column_analysis
    ]