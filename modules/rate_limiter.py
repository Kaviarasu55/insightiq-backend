from firebase_admin import firestore
from datetime import datetime

# Limits — matches your project spec exactly
CHAT_LIMIT = 20
VIZ_LIMIT = 10

def get_usage_ref(uid):
    # Gets reference to today's usage document for this user
    # Path: users/{uid}/usage/{YYYY-MM-DD}
    # The date as document ID means it auto-resets every day
    db = firestore.client()
    today = datetime.now().strftime("%Y-%m-%d")
    return db.collection("users").document(uid).collection("usage").document(today)

def check_and_increment(uid, counter_field, limit):
    # counter_field is either "chat_count" or "viz_count"
    # Returns (allowed: bool, current_count: int)

    ref = get_usage_ref(uid)
    doc = ref.get()

    if doc.exists:
        # Document exists — read current count
        current = doc.to_dict().get(counter_field, 0)
    else:
        # First call of the day — count is 0
        current = 0

    if current >= limit:
        # Over limit — block the call
        return False, current

    # Under limit — increment and allow
    ref.set({counter_field: firestore.Increment(1)}, merge=True)
    return True, current + 1

def check_chat_limit(uid):
    # Called before every chatbot Groq call
    allowed, count = check_and_increment(uid, "chat_count", CHAT_LIMIT)
    return allowed, count

def check_viz_limit(uid):
    # Called before every visualization explanation Groq call
    allowed, count = check_and_increment(uid, "viz_count", VIZ_LIMIT)
    return allowed, count