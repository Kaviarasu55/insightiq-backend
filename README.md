# InsightIQ — Backend

> AI-Powered Data Analytics Platform — Flask REST API

## 🔗 Live Demo
Frontend: [https://insightiqk.netlify.app](https://insightiqk.netlify.app)

## 🧠 What it does
REST API backend for InsightIQ — handles CSV analysis, AI-powered summaries, visualizations, ML training, AutoML, chatbot, and PDF report generation.

## ⚙️ Tech Stack
- **Flask** — Python web framework
- **Groq API** — AI summaries, explanations, chatbot (model: openai/gpt-oss-120b)
- **Firebase Admin** — Auth token verification + Firestore database
- **Supabase Storage** — CSV file storage
- **scikit-learn** — ML model training (Random Forest, Gradient Boosting, Linear/Logistic Regression)
- **ReportLab** — PDF report generation
- **Gunicorn** — Production WSGI server
- **Render** — Cloud deployment

## 🚀 Features
- CSV upload + analysis (up to 50,000 rows)
- AI dataset summary via Groq
- Auto-generated chart configurations
- Multi-turn AI chatbot with full dataset statistics context
- ML model training + single prediction
- AutoML — trains 3 models and compares performance
- PDF report export
- Firebase Auth token verification on every route
- Rate limiting (20 chat messages/day, 10 viz explanations/upload)

## 📁 Project Structure
```
backend/
├── app.py                  # Main Flask app + all routes
├── modules/
│   ├── analyzer.py         # CSV analysis
│   ├── visualizer.py       # Chart data generation
│   ├── groq_engine.py      # All Groq API calls
│   ├── ml_engine.py        # ML model training + prediction
│   ├── automl_engine.py    # AutoML comparison
│   ├── firebase_handler.py # Firestore operations
│   ├── supabase_handler.py # Supabase storage operations
│   ├── rate_limiter.py     # Rate limiting logic
│   └── report_generator.py # PDF generation
├── Procfile                # Gunicorn start command
└── requirements.txt
```

## 🛠️ Run Locally
```bash
git clone https://github.com/Kaviarasu55/insightiq-backend
cd insightiq-backend
pip install -r requirements.txt
python app.py
```

Create a `.env` file with:
```
GROQ_API_KEY=your_key
FIREBASE_CREDENTIALS_JSON=your_service_account_json
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
```

## 🔒 Environment Variables
| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key |
| `FIREBASE_CREDENTIALS_JSON` | Firebase service account JSON (full contents) |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase anon key |
