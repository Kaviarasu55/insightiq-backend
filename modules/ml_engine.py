import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)


def detect_task_type(df, target_col):
    # If target has 10 or fewer unique values → classification
    # Otherwise → regression
    unique_count = df[target_col].nunique()
    if unique_count <= 10:
        return "classification"
    return "regression"


def prepare_data(df, target_col):
    df = df.dropna(subset=[target_col]).copy()
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = X.dropna(axis=1, how="all")

    # NEW: drop identifier-like columns before they're used as features
    id_like = []
    for col in X.columns:
        name_hints_id = col.lower() in ("id", "index") or col.lower().endswith("_id")
        looks_unique_sequential = pd.api.types.is_numeric_dtype(X[col]) and X[
            col
        ].nunique() == len(X)
        if name_hints_id or looks_unique_sequential:
            id_like.append(col)
    X = X.drop(columns=id_like)

    # Impute missing numeric values (LogisticRegression etc. can't handle NaN natively)
    for col in X.columns:
      if pd.api.types.is_numeric_dtype(X[col]) and X[col].isna().any():
       X[col] = X[col].fillna(X[col].median())

    encoders = {}
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    target_encoder = None
    if not pd.api.types.is_numeric_dtype(y):
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
    return X, y, encoders, target_encoder, id_like  # surface what was dropped


CLASSIFICATION_MODELS = {"Logistic Regression", "Random Forest", "Gradient Boosting"}
REGRESSION_MODELS = {
    "Linear Regression",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
}


def get_model_by_name(model_name, task_type):
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    model_map = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
    }

    valid_names = (
        CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
    )

    if model_name and model_name in valid_names:
        return model_map[model_name]

    if task_type == "classification":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    return RandomForestRegressor(n_estimators=100, random_state=42)


def train_model(df, target_col, preferred_model=None):
    # Main function — trains model and returns results

    task_type = detect_task_type(df, target_col)
    X, y, encoders, target_encoder, dropped_id_cols = prepare_data(df, target_col)

    # Stratify classification splits so both classes appear in test set
    stratify_arg = (
        y
        if task_type == "classification" and pd.Series(y).value_counts().min() >= 2
        else None
    )

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    # Flag unreliable test sizes
    small_sample_warning = None
    if len(X_test) < 20:
        small_sample_warning = (
            f"Test set is only {len(X_test)} rows — metrics may not be reliable."
        )

    # Pick model — use AutoML recommendation if available
    # Otherwise fall back to Random Forest
    model = get_model_by_name(preferred_model, task_type)

    # Train the model
    model.fit(X_train, y_train)

    # Get predictions on test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    if task_type == "classification":
        cm = confusion_matrix(y_test, y_pred)
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred, average="weighted"), 4),
            "confusion_matrix": {str(i): row.tolist() for i, row in enumerate(cm)},
        }
    else:
        metrics = {
            "r2": round(r2_score(y_test, y_pred), 4),
            "mae": round(mean_absolute_error(y_test, y_pred), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        }

    # Top 5 most important features
    feature_importance = get_top_features(model, X.columns.tolist())

    return {
        "task_type": task_type,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "model": model,
        "encoders": encoders,
        "target_encoder": target_encoder,
        "feature_columns": X.columns.tolist(),
        "dropped_id_cols": dropped_id_cols,
        # Track which model was actually used
        "model_name": preferred_model
        or (
            "Random Forest Classifier"
            if task_type == "classification"
            else "Random Forest Regressor"
        ),
        "small_sample_warning": small_sample_warning,
    }


def get_top_features(model, feature_names):
    # Tree models (Random Forest, Gradient Boosting)
    # have feature_importances_
    # Linear models (Linear Regression, Logistic Regression)
    # have coef_ instead

    try:
        importances = model.feature_importances_
    except AttributeError:
        try:
            importances = abs(model.coef_).flatten()
        except Exception:
            return []

    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [
        {"feature": name, "importance": round(float(score), 4)}
        for name, score in pairs[:5]
    ]


def predict_single(model_result, input_values):
    # Called when user enters new values for prediction
    # input_values: dict of {column_name: value}

    model = model_result["model"]
    encoders = model_result["encoders"]
    target_encoder = model_result["target_encoder"]
    feature_columns = model_result["feature_columns"]

    # Build input row in the exact column order model expects
    row = {}
    for col in feature_columns:
        val = input_values.get(col, 0)
        if col in encoders:
            # Encode text value using same encoder from training
            try:
                val = encoders[col].transform([str(val)])[0]
            except ValueError:
                # Unseen label — default to 0
                val = 0
        row[col] = val

    input_df = pd.DataFrame([row])
    prediction = model.predict(input_df)[0]

    # Get confidence for classification
    confidence = None
    if model_result["task_type"] == "classification":
        proba = model.predict_proba(input_df)[0]
        confidence = round(float(max(proba)), 4)
        # Decode numeric prediction back to original label
        if target_encoder:
            prediction = target_encoder.inverse_transform([int(prediction)])[0]

    return {
        "prediction": str(prediction),
        "confidence": confidence,
    }
