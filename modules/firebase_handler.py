from firebase_admin import firestore
from datetime import datetime

def get_db():
    # Gets Firestore client — called inside each function
    # so we always get a fresh reference
    return firestore.client()

# ─── DATASET METADATA ───────────────────────────────────────────

def save_dataset_metadata(uid, dataset_id, filename, row_count,
                           col_count, column_analysis, sample_rows):
    # Saves dataset info to Firestore after CSV is uploaded and analyzed
    # Path: users/{uid}/datasets/{dataset_id}

    db = get_db()
    db.collection("users").document(uid)\
      .collection("datasets").document(dataset_id)\
      .set({
          "filename": filename,
          "row_count": row_count,
          "col_count": col_count,
          "column_analysis": column_analysis,
          "sample_rows": sample_rows,
          "uploaded_at": firestore.SERVER_TIMESTAMP,
      })

def get_dataset_metadata(uid, dataset_id):
    # Fetches dataset metadata for a specific dataset
    # Used when user reopens old dataset

    db = get_db()
    doc = db.collection("users").document(uid)\
            .collection("datasets").document(dataset_id)\
            .get()

    if doc.exists:
        data = doc.to_dict()
        data["id"] = doc.id   # include the document ID
        return data
    return None

def get_all_datasets(uid):
    # Fetches all datasets for a user — shown on Dashboard
    # Returns list sorted by upload date newest first

    db = get_db()
    docs = db.collection("users").document(uid)\
             .collection("datasets")\
             .order_by("uploaded_at",
                       direction=firestore.Query.DESCENDING)\
             .stream()

    datasets = []
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        # Convert Firestore timestamp to string for JSON
        if data.get("uploaded_at"):
            try:
                data["uploaded_at"] = data["uploaded_at"].strftime(
                    "%Y-%m-%d %H:%M"
                )
            except Exception:
                data["uploaded_at"] = str(data.get("uploaded_at", ""))
        datasets.append(data)

    return datasets

# ─── ML AND AUTOML RESULTS ──────────────────────────────────────

def save_ml_results(uid, dataset_id, task_type, metrics,
                    feature_importance, groq_explanation):
    # Saves ML Prediction results to Firestore
    # Used by Export Report to include ML results in PDF

    db = get_db()
    db.collection("users").document(uid)\
      .collection("datasets").document(dataset_id)\
      .set({
          "ml_results": {
              "task_type": task_type,
              "metrics": metrics,
              "feature_importance": feature_importance,
              "groq_explanation": groq_explanation,
          }
      }, merge=True)   # merge=True so we don't overwrite other fields

def save_automl_results(uid, dataset_id, results,
                        best_model, groq_explanation):
    # Saves AutoML results to Firestore

    db = get_db()
    db.collection("users").document(uid)\
      .collection("datasets").document(dataset_id)\
      .set({
          "automl_results": {
              "results": results,
              "best_model": best_model,
              "groq_explanation": groq_explanation,
          }
      }, merge=True)

# ─── CHAT HISTORY ───────────────────────────────────────────────

def save_chat_message(uid, dataset_id, role, content):
    # Appends a single message to chat history in Firestore
    # role: "user" or "assistant"
    # Called after every message — both user message and AI reply

    db = get_db()
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    }

    db.collection("users").document(uid)\
      .collection("datasets").document(dataset_id)\
      .set({
          "chat_history": firestore.ArrayUnion([message])
      }, merge=True)

def get_chat_history(uid, dataset_id):
    # Fetches all chat messages for a dataset
    # Used when user reopens old dataset to restore conversation

    db = get_db()
    doc = db.collection("users").document(uid)\
            .collection("datasets").document(dataset_id)\
            .get()

    if doc.exists:
        return doc.to_dict().get("chat_history", [])
    return []

# ─── GROQ DATASET SUMMARY ───────────────────────────────────────

def save_dataset_summary(uid, dataset_id, summary):
    # Saves Groq-generated dataset summary paragraph
    # So we don't regenerate it every time user opens the dataset

    db = get_db()
    db.collection("users").document(uid)\
      .collection("datasets").document(dataset_id)\
      .set({"groq_summary": summary}, merge=True)

# ─── DELETE DATASET ───────────────────────────────────────

def delete_dataset(uid, dataset_id):
    # Deletes entire dataset document from Firestore
    # Including all subcollections like chat_history, ml_results etc
    # Called when user clicks delete on Dashboard

    db = get_db()
    db.collection("users").document(uid)\
      .collection("datasets").document(dataset_id)\
      .delete()