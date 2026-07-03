import os
from groq import Groq

# Create Groq client once — reused for all calls
# api_key reads from .env via load_dotenv() in app.py
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"


def build_dataset_context(column_info, sample_rows):
    # Builds a plain text description of the dataset
    # Injected into system prompt so Groq actually knows the data
    # column_info: list of dicts with name, type, stats
    # sample_rows: first 5 rows as a list of dicts

    lines = ["Dataset columns:"]
    for col in column_info:
        lines.append(f"- {col['name']} ({col['type']}): {col.get('summary', '')}")

    lines.append("\nSample rows:")
    for row in sample_rows[:5]:
        lines.append(str(row))

    return "\n".join(lines)


def compute_dataset_stats(df):
    # Computes rich statistics from the FULL dataset
    # This replaces the 5-row sample in chat context
    # So Groq can answer questions like "who scored most runs in 2016"

    lines = []

    # 1. Basic shape
    lines.append(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    lines.append(f"Columns: {', '.join(df.columns.tolist())}")
    lines.append("")

    # 2. Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Also treat low-cardinality numeric columns as grouping columns
    # Example: Year has only 15 unique values — perfect for grouping
    # Threshold: 30 unique values or less = treat as a category too
    low_cardinality_nums = [col for col in numeric_cols if df[col].nunique() <= 30]
    # Combine for grouping — but keep numeric_cols pure for stats
    grouping_cols = categorical_cols + low_cardinality_nums
    # key_numeric excludes low-cardinality cols (Year, Matches etc.)
    # So Wickets moves up to position 3 instead of 9!
    key_numeric = [
        col for col in numeric_cols 
        if col not in low_cardinality_nums
    ][:8]

    # 3. Overall numeric stats for each numeric column
    # Gives Groq: min, max, mean, sum for every number column
    if numeric_cols:
        lines.append("=== Numeric Column Statistics ===")
        for col in numeric_cols:
            s = df[col].dropna()
            lines.append(
                f"{col}: min={s.min()}, max={s.max()}, "
                f"mean={round(s.mean(), 2)}, sum={round(s.sum(), 2)}, "
                f"median={round(s.median(), 2)}"
            )
        lines.append("")

    # 4. Categorical column value counts
    # Gives Groq: how many times each value appears
    if categorical_cols:
        lines.append("=== Categorical Column Value Counts ===")
        for col in categorical_cols:
            counts = df[col].value_counts().head(10)
            counts_str = ", ".join([f"{v}({c})" for v, c in counts.items()])
            lines.append(f"{col}: {counts_str}")
        lines.append("")

    # 5. Group-by aggregations — THE KEY FIX
    # Uses grouping_cols which includes Year-like numeric columns
    if grouping_cols and numeric_cols:
        lines.append("=== Aggregations (sum per group) ===")
        for cat_col in grouping_cols[:2]:
            for num_col in key_numeric[:4]:
                try:
                    grouped = (
                        df.groupby(cat_col)[num_col]
                        .sum()
                        .sort_values(ascending=False)
                        .head(15)  # top 15 values
                    )
                    group_str = ", ".join(
                        [f"{k}={round(v, 2)}" for k, v in grouped.items()]
                    )
                    lines.append(f"Total {num_col} by {cat_col}: {group_str}")
                except Exception:
                    continue
        lines.append("")
    
    # 6. Multi-column group-by — KEY for year-filtered questions
    # Example: "who scored most runs in 2016?"
    # Needs: Player × Year → Runs breakdown
    for year_col in low_cardinality_nums[:1]:
            unique_vals = sorted(df[year_col].dropna().unique())
            for cat_col in categorical_cols[:1]:
                for num_col in key_numeric[:8]:
                    lines.append(
                        f"\nTop {cat_col} by {num_col} for each {year_col}:"
                    )
                    for val in unique_vals:
                        try:
                            filtered = df[df[year_col] == val]
                            top = (
                                filtered.groupby(cat_col)[num_col]
                                .sum()
                                .sort_values(ascending=False)
                                .head(3)  # top 3 per year
                            )
                            top_str = ", ".join(
                                [f"{k}={round(v,1)}" for k, v in top.items()]
                            )
                            lines.append(f"  {year_col}={val}: {top_str}")
                        except Exception:
                            continue
    lines.append("")

    return "\n".join(lines)


def get_dataset_summary(column_info, sample_rows, row_count):
    # Page 3 — Dataset Overview
    # Generates a full paragraph summary of the entire dataset

    context = build_dataset_context(column_info, sample_rows)

    prompt = f"""You are a data analyst explaining a dataset to a non-technical user.
{context}
Total rows: {row_count}

Write a clear 3-4 sentence paragraph summary of this dataset.
Mention: what the data appears to be about, how many rows and columns,
which columns have missing values, and anything notable at first glance.
Use plain English. No bullet points. No technical jargon."""

    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], timeout=15
        )
        return response.choices[0].message.content
    except Exception as e:
        # Warn not crash — return fallback message
        return f"Summary unavailable: {str(e)}"


def get_chart_explanation(chart_type, column_name, stats, column_info, sample_rows):
    # Page 4 — Visualizations
    # Generates explanation for a single chart

    context = build_dataset_context(column_info, sample_rows)

    prompt = f"""You are a data analyst explaining a chart to a non-technical user.
{context}

Chart type: {chart_type}
Column being shown: {column_name}
Statistics: {stats}

Write a clear 2-3 sentence explanation of what this chart shows.
Mention the distribution pattern, any outliers, and what it means
for the business or use case. Plain English only."""

    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], timeout=15
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Explanation unavailable: {str(e)}"


def get_chat_response(user_message, chat_history, column_info, dataset_stats):
    # Page 5 — AI Chatbot
    # Multi-turn conversation about the dataset
    # Now uses full dataset stats instead of 5 sample rows
    # So Groq can answer aggregation questions like
    # "who scored most runs in 2016"

    # Build column descriptions from column_info
    col_lines = ["Dataset columns:"]
    for col in column_info:
        col_lines.append(f"- {col['name']} ({col['type']}): {col.get('summary', '')}")
    col_context = "\n".join(col_lines)

    # System prompt now includes FULL dataset stats
    # not just 5 sample rows
    system_prompt = f"""You are a helpful data analyst assistant.
The user has uploaded a dataset with the following structure:

{col_context}

Here are the full statistics computed from the entire dataset:
{dataset_stats}

Use the statistics above to answer questions accurately.
If a user asks "who scored most runs" — look at the aggregations section.
If a user asks about averages or totals — use the numeric stats section.
Answer in plain English using simple paragraphs and short bullet points only.
Do NOT use markdown tables, bold text with asterisks, or headers.
Be specific and reference actual values.
If something is not in the statistics, say so honestly."""

    # Build messages array: system prompt + full history + new message
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history[-10:])  # limit to 10 messages to avoid token overflow 
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, timeout=15
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Response unavailable: {str(e)}"


def get_ml_explanation(task_type, metrics, feature_importance, target_column):
    # Page 6 — ML Prediction
    # Explains model results in plain English

    prompt = f"""You are explaining machine learning results to a non-technical user.

Task: {task_type}
Target column: {target_column}
Model performance: {metrics}
Most important features: {feature_importance}

Write a clear 3-4 sentence explanation of these results.
Explain what the accuracy/score means in plain English,
which factors mattered most, and what this tells us about the data.
No technical jargon."""

    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], timeout=15
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Explanation unavailable: {str(e)}"


def get_prediction_explanation(task_type, input_values, prediction, confidence):
    # Page 6 — ML Prediction
    # Explains a single prediction result

    prompt = f"""You are explaining a prediction to a non-technical user.

Task: {task_type}
Input values provided: {input_values}
Prediction result: {prediction}
Confidence: {confidence}

Write 2-3 sentences explaining what this prediction means in plain English.
Connect the input values to the outcome. Be clear and specific."""

    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], timeout=15
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Explanation unavailable: {str(e)}"


def get_automl_explanation(results, best_model, task_type):
    # Page 7 — AutoML
    # Explains why the best model won

    prompt = f"""You are explaining AutoML results to a non-technical user.

Task type: {task_type}
Models compared: {results}
Best model: {best_model}

Write a clear 3-4 sentence explanation of why {best_model} performed best.
Compare it to the others simply. Explain what this means about the data.
Plain English only. No technical jargon."""

    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], timeout=15
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Explanation unavailable: {str(e)}"


def get_ai_chart_configs(column_info, sample_rows):
    # Asks Groq to decide which charts make sense for this dataset
    # Returns a list of chart config dicts
    # Each config has: chart_type, x_col, y_col (optional), reason

    context = build_dataset_context(column_info, sample_rows)

    prompt = f"""You are a data visualization expert.
Here is a dataset:
{context}

Decide the most meaningful charts to generate for this dataset.
Consider what insights would be most useful for a non-technical user.

Rules:
- histogram: use for numeric columns to show distribution
- bar: use for categorical columns to show frequency
- scatter: use for two numeric columns to show relationship
- line: only use if there is a datetime/date column

Return ONLY a JSON array. No explanation, no markdown, no backticks.
Maximum 6 charts. Each item must have:
- chart_type: one of histogram, bar, scatter, line
- x_col: column name for X axis (must exist in the dataset)
- y_col: column name for Y axis (null if not needed)
- reason: one sentence why this chart is useful

Example format:
[
  {{"chart_type": "scatter", "x_col": "Age", "y_col": "Income", "reason": "Shows if older customers earn more"}},
  {{"chart_type": "bar", "x_col": "City", "y_col": null, "reason": "Compares customer count per city"}}
]"""

    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], timeout=15
        )
        raw = response.choices[0].message.content

        # Strip any accidental markdown backticks
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        import json

        configs = json.loads(raw.strip())

        # Validate each config has required fields
        valid = []
        for c in configs:
            if "chart_type" in c and "x_col" in c:
                valid.append(c)

        return valid

    except Exception as e:
        # Fallback — return basic configs if Groq fails
        numeric_cols = [c["name"] for c in column_info if c["type"] == "numeric"]
        categorical_cols = [
            c["name"] for c in column_info if c["type"] == "categorical"
        ]

        fallback = []
        for col in numeric_cols[:2]:
            fallback.append(
                {
                    "chart_type": "histogram",
                    "x_col": col,
                    "y_col": None,
                    "reason": "Distribution of " + col,
                }
            )
        for col in categorical_cols[:2]:
            fallback.append(
                {
                    "chart_type": "bar",
                    "x_col": col,
                    "y_col": None,
                    "reason": "Frequency of " + col,
                }
            )
        if len(numeric_cols) >= 2:
            fallback.append(
                {
                    "chart_type": "scatter",
                    "x_col": numeric_cols[0],
                    "y_col": numeric_cols[1],
                    "reason": f"Relationship between {numeric_cols[0]} and {numeric_cols[1]}",
                }
            )

        return fallback


def get_best_chart_type(x_col, y_col, column_info, sample_rows):
    # Called when user selects "Any (AI decides)"
    # Groq picks the best chart type for these two columns

    context = build_dataset_context(column_info, sample_rows)

    y_info = f"Y column: {y_col}" if y_col else "No Y column selected"

    prompt = f"""You are a data visualization expert.
{context}

User wants to visualize:
X column: {x_col}
{y_info}

Pick the single best chart type from: histogram, bar, scatter, line

Rules:
- histogram: X is numeric, no Y needed
- bar: X is categorical, no Y needed
- scatter: X and Y are both numeric
- line: X is datetime/date

Return ONLY one word — the chart type. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], timeout=15
        )
        result = response.choices[0].message.content.strip().lower()

        # Validate it returned a valid chart type
        valid_types = ["histogram", "bar", "scatter", "line"]
        if result in valid_types:
            return result
        return "bar"  # safe fallback

    except Exception:
        return "bar"  # safe fallback
