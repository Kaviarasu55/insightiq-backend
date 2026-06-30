import os
import uuid
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from io import BytesIO
import firebase_admin
from firebase_admin import credentials, auth

# Load .env variables
load_dotenv()

# Create Flask app
app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:5173",
    "https://your-netlify-site.netlify.app"
])

# Initialize Firebase Admin
import json
firebase_cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
cred = credentials.Certificate(json.loads(firebase_cred_json))
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Import all modules
from modules.analyzer import analyze_csv, get_column_info_for_groq
from modules.visualizer import generate_chart_data, get_manual_chart_data
from modules.groq_engine import (
    get_dataset_summary,
    get_chart_explanation,
    get_chat_response,
    get_ml_explanation,
    get_prediction_explanation,
    get_automl_explanation,
)
from modules.ml_engine import train_model, predict_single
from modules.automl_engine import run_automl
from modules.firebase_handler import (
    save_dataset_metadata,
    get_dataset_metadata,
    get_all_datasets,
    save_ml_results,
    save_automl_results,
    save_chat_message,
    get_chat_history,
    save_dataset_summary,
)
from modules.rate_limiter import check_chat_limit, check_viz_limit
from modules.report_generator import generate_report
from modules.supabase_handler import upload_csv, download_csv, delete_csv

# In-memory store for trained ML models
# Key: "{uid}_{dataset_id}" → value: model_result dict
model_store = {}

# ── AUTH HELPER ────────────────────────────────────────────────


def verify_firebase_token(request):
    # Reads and verifies Bearer token from Authorization header
    # Returns uid string if valid, raises Exception if not

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise Exception("No token provided")

    token = auth_header.split("Bearer ")[1]
    decoded = auth.verify_id_token(token)
    return decoded["uid"]


# ── CSV HELPER ─────────────────────────────────────────────────


def get_csv_dataframe(uid, dataset_id):
    # Helper function — downloads CSV from Supabase and returns DataFrame
    # Used by visualize, ml_train, automl routes
    from io import BytesIO
    import pandas as pd

    csv_bytes = download_csv(uid, dataset_id)
    return pd.read_csv(BytesIO(csv_bytes))


# ── TEST ROUTE ─────────────────────────────────────────────────


@app.route("/ping")
def ping():
    return jsonify({"status": "ok", "message": "InsightIQ backend is running"})


# ── UPLOAD + ANALYZE ───────────────────────────────────────────


@app.route("/upload", methods=["POST"])
def upload():
    try:
        uid = verify_firebase_token(request)
    except Exception as e:
        return jsonify({"error": "Unauthorized"}), 401

    # Check file exists in request
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Check it's a CSV
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400

    try:
        # Analyze the CSV
        result = analyze_csv(file)

        # Enforce 50,000 row limit from spec
        if result["row_count"] > 50000:
            return (
                jsonify(
                    {
                        "error": f"Dataset has {result['row_count']} rows. Maximum allowed is 50,000."
                    }
                ),
                400,
            )

        # Generate unique dataset ID
        dataset_id = uuid.uuid4().hex

        # Get column info for Groq
        column_info = get_column_info_for_groq(result["columns"])

        # Generate Groq dataset summary
        groq_summary = get_dataset_summary(
            column_info, result["sample_rows"], result["row_count"]
        )

        # Upload CSV to Supabase Storage for later use
        file.seek(0)  # rewind file pointer to start before reading again
        csv_bytes = file.read()
        upload_csv(uid, dataset_id, csv_bytes)

        # Save everything to Firestore
        save_dataset_metadata(
            uid,
            dataset_id,
            file.filename,
            result["row_count"],
            result["col_count"],
            result["columns"],
            result["sample_rows"],
        )
        save_dataset_summary(uid, dataset_id, groq_summary)

        return jsonify(
            {
                "dataset_id": dataset_id,
                "row_count": result["row_count"],
                "col_count": result["col_count"],
                "columns": result["columns"],
                "preview": result["preview"],
                "sample_rows": result["sample_rows"],
                "groq_summary": groq_summary,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── DASHBOARD ──────────────────────────────────────────────────


@app.route("/datasets", methods=["GET"])
def get_datasets():
    # Returns all datasets for the logged-in user
    try:
        uid = verify_firebase_token(request)
    except:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        datasets = get_all_datasets(uid)
        return jsonify({"datasets": datasets})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/datasets/<dataset_id>", methods=["GET"])
def get_dataset(dataset_id):
    # Returns metadata for a single dataset
    try:
        uid = verify_firebase_token(request)
    except:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = get_dataset_metadata(uid, dataset_id)
        if not data:
            return jsonify({"error": "Dataset not found"}), 404
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/datasets/<dataset_id>", methods=["DELETE"])
def delete_dataset_route(dataset_id):
    # Deletes dataset from both Firestore and Supabase
    # Called when user clicks delete button on Dashboard
    try:
        uid = verify_firebase_token(request)
    except:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        # First check dataset exists and belongs to this user
        data = get_dataset_metadata(uid, dataset_id)
        if not data:
            return jsonify({"error": "Dataset not found"}), 404

        # Delete from Firestore — metadata, chat, ML results
        from modules.firebase_handler import delete_dataset

        delete_dataset(uid, dataset_id)

        # Delete CSV from Supabase Storage
        delete_csv(uid, dataset_id)

        return jsonify({"success": True, "deleted": dataset_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── VISUALIZATIONS ─────────────────────────────────────────────


@app.route("/visualize/<dataset_id>", methods=["GET"])
def visualize(dataset_id):
    try:
        uid = verify_firebase_token(request)
    except:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        # Get dataset metadata from Firestore
        metadata = get_dataset_metadata(uid, dataset_id)
        if not metadata:
            return jsonify({"error": "Dataset not found"}), 404

        column_analysis = metadata["column_analysis"]
        sample_rows = metadata.get("sample_rows", [])

        # Get column info for Groq
        column_info = get_column_info_for_groq(column_analysis)

        # Ask Groq which charts make sense for this dataset
        from modules.groq_engine import get_ai_chart_configs

        chart_configs = get_ai_chart_configs(column_info, sample_rows)

        # Download real CSV from Supabase
        df = get_csv_dataframe(uid, dataset_id)

        # Generate actual chart data for each AI-decided config
        charts = []
        for config in chart_configs:
            chart_type = config.get("chart_type")
            x_col = config.get("x_col")
            y_col = config.get("y_col")
            reason = config.get("reason", "")

            # Validate columns exist in the dataframe
            if x_col not in df.columns:
                continue
            if y_col and y_col not in df.columns:
                y_col = None

            # Generate chart data using visualizer.py
            # For bar charts, y_col enables aggregation (e.g., sum of Runs per Player)
            data = get_manual_chart_data(df, chart_type, x_col, y_col)

            if data:
                # Build descriptive column label
                if y_col:
                    col_label = f"{y_col} by {x_col}"
                else:
                    col_label = x_col

                charts.append(
                    {
                        "chart_type": chart_type,
                        "column": col_label,
                        "x_col": x_col,
                        "y_col": y_col,
                        "data": data,
                        "reason": reason,  # AI's reason for this chart
                    }
                )

        return jsonify(
            {
                "charts": charts,
                "column_analysis": column_analysis,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/visualize/<dataset_id>/explain", methods=["POST"])
def explain_chart(dataset_id):
    # Generates Groq explanation for a single chart
    # Rate limited — 10 viz explanations per upload
    try:
        uid = verify_firebase_token(request)
    except:
        return jsonify({"error": "Unauthorized"}), 401

    allowed, count = check_viz_limit(uid)
    if not allowed:
        return (
            jsonify(
                {"error": "Visualization explanation limit reached (10 per upload)."}
            ),
            429,
        )

    try:
        body = request.get_json()
        chart_type = body["chart_type"]
        column_name = body["column_name"]
        stats = body.get("stats", {})

        metadata = get_dataset_metadata(uid, dataset_id)
        column_info = get_column_info_for_groq(metadata["column_analysis"])
        sample_rows = metadata["sample_rows"]

        explanation = get_chart_explanation(
            chart_type, column_name, stats, column_info, sample_rows
        )
        return jsonify({"explanation": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/visualize/<dataset_id>/manual", methods=["POST"])
def manual_chart(dataset_id):
    try:
        uid = verify_firebase_token(request)
    except:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        body = request.get_json()
        chart_type = body.get("chart_type")
        x_col = body.get("x_col")
        y_col = body.get("y_col")

        df = get_csv_dataframe(uid, dataset_id)
        metadata = get_dataset_metadata(uid, dataset_id)
        column_info = get_column_info_for_groq(metadata["column_analysis"])
        sample_rows = metadata.get("sample_rows", [])

        # If user chose "any" — let Groq decide the best chart type
        if chart_type == "any":
            from modules.groq_engine import get_best_chart_type

            chart_type = get_best_chart_type(x_col, y_col, column_info, sample_rows)

        if x_col not in df.columns:
            return jsonify({"error": f"Column {x_col} not found"}), 400

        data = get_manual_chart_data(df, chart_type, x_col, y_col)

        return jsonify(
            {
                "data": data,
                "chart_type": chart_type,
                "x_col": x_col,
                "y_col": y_col,
                "ai_decided": body.get("chart_type") == "any",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── CHATBOT ────────────────────────────────────────────────────


@app.route("/chat/<dataset_id>", methods=["POST"])
def chat(dataset_id):
    # Handles a single chat message
    # Rate limited — 20 messages per user per day
    try:
        uid = verify_firebase_token(request)
    except:
        return jsonify({"error": "Unauthorized"}), 401

    allowed, count = check_chat_limit(uid)
    if not allowed:
        return (
            jsonify(
                {"error": "You've reached your daily chat limit. Come back tomorrow!"}
            ),
            429,
        )

    try:
        body = request.get_json()
        user_message = body["message"]

        # Get metadata for column info
        metadata = get_dataset_metadata(uid, dataset_id)
        column_info = get_column_info_for_groq(metadata["column_analysis"])

        # Download FULL CSV from Supabase and compute rich stats
        # This replaces the 5 sample rows with complete dataset statistics
        df = get_csv_dataframe(uid, dataset_id)
        from modules.groq_engine import compute_dataset_stats

        dataset_stats = compute_dataset_stats(df)

        # Get existing chat history from Firestore
        chat_history = get_chat_history(uid, dataset_id)

        # Format history for Groq — only role and content
        groq_history = [
            {"role": m["role"], "content": m["content"]} for m in chat_history
        ]

        # Get Groq response — now with full dataset stats
        reply = get_chat_response(
            user_message, groq_history, column_info, dataset_stats
        )

        # Save both messages to Firestore
        save_chat_message(uid, dataset_id, "user", user_message)
        save_chat_message(uid, dataset_id, "assistant", reply)

        return jsonify(
            {
                "reply": reply,
                "messages_used": count,
                "messages_remaining": 20 - count,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat/<dataset_id>/history", methods=["GET"])
def chat_history(dataset_id):
    # Returns full chat history for a dataset
    try:
        uid = verify_firebase_token(request)
    except:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        history = get_chat_history(uid, dataset_id)
        return jsonify({"history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── ML PREDICTION ──────────────────────────────────────────────


@app.route("/ml/<dataset_id>/train", methods=["POST"])
def ml_train(dataset_id):
    # Trains ML model on uploaded CSV
    try:
        uid = verify_firebase_token(request)
    except:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        # Download CSV from Supabase
        df = get_csv_dataframe(uid, dataset_id)

        body = request.get_json()
        target_col = body.get("target_column")

        if not target_col or target_col not in df.columns:
            return jsonify({"error": "Invalid target column"}), 400

        # Check if AutoML already ran for this dataset
        # If yes — use its best model recommendation
        metadata = get_dataset_metadata(uid, dataset_id)
        automl_results = metadata.get("automl_results")
        preferred_model = None
        if automl_results:
            preferred_model = automl_results.get("best_model")

        # Train the model — with AutoML recommendation if available
        model_result = train_model(df, target_col, preferred_model)

        # Store model in memory for predictions
        store_key = f"{uid}_{dataset_id}"
        model_store[store_key] = model_result

        # Get Groq explanation
        groq_explanation = get_ml_explanation(
            model_result["task_type"],
            model_result["metrics"],
            model_result["feature_importance"],
            target_col,
        )

        # Save results to Firestore
        save_ml_results(
            uid,
            dataset_id,
            model_result["task_type"],
            model_result["metrics"],
            model_result["feature_importance"],
            groq_explanation,
        )

        # Build unique values for categorical columns
        # IMPORTANT: check original df BEFORE encoding
        # because prepare_data() converts text → numbers
        column_unique_values = {}
        for col in model_result["feature_columns"]:
            if col in df.columns:
                original_col = df[col]
                if (
                    original_col.dtype == "object"
                    or original_col.dtype.name == "category"
                    or original_col.nunique() <= 20
                ):
                    unique_vals = original_col.dropna().unique().tolist()
                    # Only add if values look categorical (not just numbers)
                    if any(isinstance(v, str) for v in unique_vals):
                        column_unique_values[col] = sorted(
                            [str(v) for v in unique_vals]
                        )

        return jsonify(
            {
                "task_type": model_result["task_type"],
                "metrics": model_result["metrics"],
                "feature_importance": model_result["feature_importance"],
                "groq_explanation": groq_explanation,
                "feature_columns": model_result["feature_columns"],
                "model_name": model_result["model_name"],
                "column_unique_values": column_unique_values,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ml/<dataset_id>/predict", methods=["POST"])
def ml_predict(dataset_id):
    # Makes a single prediction using trained model
    try:
        uid = verify_firebase_token(request)
    except:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        store_key = f"{uid}_{dataset_id}"
        if store_key not in model_store:
            return jsonify({"error": "Model not trained yet"}), 400

        body = request.get_json()
        input_values = body["input_values"]

        result = predict_single(model_store[store_key], input_values)

        # Get Groq explanation for this prediction
        groq_explanation = get_prediction_explanation(
            model_store[store_key]["task_type"],
            input_values,
            result["prediction"],
            result["confidence"],
        )

        return jsonify(
            {
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "groq_explanation": groq_explanation,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── AUTOML ─────────────────────────────────────────────────────


@app.route("/automl/<dataset_id>", methods=["POST"])
def automl(dataset_id):
    # Runs all models and returns comparison
    try:
        uid = verify_firebase_token(request)
    except:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        df = get_csv_dataframe(uid, dataset_id)
        body = request.get_json()
        target_col = body.get("target_column")

        if not target_col or target_col not in df.columns:
            return jsonify({"error": "Invalid target column"}), 400

        result = run_automl(df, target_col)

        groq_explanation = get_automl_explanation(
            result["results"], result["best_model"], result["task_type"]
        )

        save_automl_results(
            uid, dataset_id, result["results"], result["best_model"], groq_explanation
        )

        return jsonify(
            {
                "task_type": result["task_type"],
                "metric_label": result["metric_label"],
                "results": result["results"],
                "best_model": result["best_model"],
                "groq_explanation": groq_explanation,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── EXPORT REPORT ──────────────────────────────────────────────


@app.route("/report/<dataset_id>", methods=["GET"])
def export_report(dataset_id):
    # Generates PDF report and sends as download
    try:
        uid = verify_firebase_token(request)
    except:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        metadata = get_dataset_metadata(uid, dataset_id)
        if not metadata:
            return jsonify({"error": "Dataset not found"}), 404

        # Get ML and AutoML results if they exist
        ml_results = metadata.get("ml_results")
        automl_results = metadata.get("automl_results")

        # Generate PDF bytes
        pdf_bytes = generate_report(metadata, ml_results, automl_results)

        # Send as downloadable file
        return send_file(
            BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"InsightIQ_Report_{dataset_id[:8]}.pdf",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5000)
