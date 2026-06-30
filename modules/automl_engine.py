import time
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.model_selection import cross_val_score
from modules.ml_engine import detect_task_type, prepare_data

def run_automl(df, target_col):
    # Main function — trains all models and returns comparison

    task_type = detect_task_type(df, target_col)
    X, y, encoders, target_encoder = prepare_data(df, target_col)

    # Pick models based on task type
    if task_type == "classification":
        models = get_classification_models()
        scoring = "accuracy"
        metric_label = "Accuracy"
    else:
        models = get_regression_models()
        scoring = "r2"
        metric_label = "R²"

    results = []

    for model_name, model in models:
        # Record start time
        start = time.time()

        # cross_val_score: 3-fold cross validation
        # Returns array of 3 scores — we take the mean
        scores = cross_val_score(model, X, y, cv=3, scoring=scoring)

        # Record end time
        elapsed = round(time.time() - start, 2)

        if task_type == "classification":
            # Also get F1 score for classification
            f1_scores = cross_val_score(
                model, X, y, cv=3, scoring="f1_weighted"
            )
            results.append({
                "model": model_name,
                "accuracy": round(float(scores.mean()), 4),
                "f1_score": round(float(f1_scores.mean()), 4),
                "training_time": elapsed,
            })
        else:
            # Also get MAE for regression
            mae_scores = cross_val_score(
                model, X, y, cv=3, scoring="neg_mean_absolute_error"
            )
            results.append({
                "model": model_name,
                "r2": round(float(scores.mean()), 4),
                # neg_mean_absolute_error is negative — flip it
                "mae": round(float(-mae_scores.mean()), 4),
                "training_time": elapsed,
            })

    # Find best model — highest accuracy or R²
    score_key = "accuracy" if task_type == "classification" else "r2"
    best = max(results, key=lambda x: x[score_key])

    return {
        "task_type": task_type,
        "metric_label": metric_label,
        "results": results,
        "best_model": best["model"],
    }

def get_classification_models():
    # Returns list of (name, model) tuples
    return [
        ("Logistic Regression", LogisticRegression(
            max_iter=1000, random_state=42
        )),
        ("Random Forest", RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )),
        ("Gradient Boosting", GradientBoostingClassifier(
            n_estimators=100, random_state=42
        )),
    ]

def get_regression_models():
    return [
        ("Linear Regression", LinearRegression()),
        ("Random Forest Regressor", RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )),
        ("Gradient Boosting Regressor", GradientBoostingRegressor(
            n_estimators=100, random_state=42
        )),
    ]