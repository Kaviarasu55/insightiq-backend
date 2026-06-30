from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)

def generate_report(dataset_metadata, ml_results=None, automl_results=None):
    # Main function — builds PDF and returns bytes
    # dataset_metadata: dict from Firestore
    # ml_results: dict or None (if user didn't run ML)
    # automl_results: dict or None (if user didn't run AutoML)

    # BytesIO = in-memory buffer, no disk file needed
    buffer = BytesIO()

    # Create PDF document — A4 page, 2cm margins
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )

    # Get base styles and define custom ones
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#1e3a5f"),
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#2e75b6"),
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,    # line height
        spaceAfter=6,
    )

    # Build list of elements — ReportLab flows them into pages
    elements = []

    # ── SECTION 1: HEADER ──────────────────────────────
    elements.append(Paragraph("InsightIQ — Analysis Report", title_style))
    elements.append(Spacer(1, 0.3*cm))

    # Dataset basic info
    filename = dataset_metadata.get("filename", "Unknown")
    row_count = dataset_metadata.get("row_count", 0)
    col_count = dataset_metadata.get("col_count", 0)
    uploaded_at = dataset_metadata.get("uploaded_at", "Unknown")

    elements.append(Paragraph(f"Dataset: {filename}", body_style))
    elements.append(Paragraph(f"Rows: {row_count} | Columns: {col_count}", body_style))
    elements.append(Paragraph(f"Uploaded: {uploaded_at}", body_style))
    elements.append(Spacer(1, 0.5*cm))

    # ── SECTION 2: DATASET SUMMARY ────────────────────
    elements.append(Paragraph("Dataset Summary", heading_style))
    summary = dataset_metadata.get("groq_summary", "Summary not available.")
    elements.append(Paragraph(summary, body_style))
    elements.append(Spacer(1, 0.5*cm))

    # ── SECTION 3: COLUMN ANALYSIS TABLE ──────────────
    elements.append(Paragraph("Column Analysis", heading_style))

    column_analysis = dataset_metadata.get("column_analysis", [])
    if column_analysis:
        # Build table data — header row first
        table_data = [["Column", "Type", "Nulls", "Unique", "Summary"]]
        for col in column_analysis:
            table_data.append([
                col.get("name", ""),
                col.get("type", ""),
                str(col.get("null_count", 0)),
                str(col.get("unique_count", 0)),
                col.get("summary", "")[:50],  # truncate long summaries
            ])

        col_table = Table(table_data, colWidths=[3.5*cm, 2.5*cm, 1.8*cm, 1.8*cm, 6*cm])
        col_table.setStyle(TableStyle([
            # Header row — dark blue background
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            # Alternating row colors for readability
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(col_table)

    elements.append(Spacer(1, 0.5*cm))

    # ── SECTION 4: ML RESULTS ─────────────────────────
    if ml_results:
        elements.append(Paragraph("ML Prediction Results", heading_style))

        task_type = ml_results.get("task_type", "")
        metrics = ml_results.get("metrics", {})
        feature_importance = ml_results.get("feature_importance", [])
        groq_explanation = ml_results.get("groq_explanation", "")

        elements.append(Paragraph(f"Task Type: {task_type.capitalize()}", body_style))

        # Metrics table
        if task_type == "classification":
            metrics_data = [
                ["Metric", "Value"],
                ["Accuracy", str(metrics.get("accuracy", ""))],
                ["F1 Score", str(metrics.get("f1_score", ""))],
            ]
        else:
            metrics_data = [
                ["Metric", "Value"],
                ["R²", str(metrics.get("r2", ""))],
                ["MAE", str(metrics.get("mae", ""))],
                ["RMSE", str(metrics.get("rmse", ""))],
            ]

        metrics_table = Table(metrics_data, colWidths=[5*cm, 5*cm])
        metrics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e75b6")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.3*cm))

        # Top features
        if feature_importance:
            elements.append(Paragraph("Top Important Features:", body_style))
            for feat in feature_importance:
                elements.append(Paragraph(
                    f"• {feat['feature']}: {feat['importance']}",
                    body_style
                ))

        elements.append(Spacer(1, 0.3*cm))
        elements.append(Paragraph("AI Explanation:", body_style))
        elements.append(Paragraph(groq_explanation, body_style))
        elements.append(Spacer(1, 0.5*cm))

    # ── SECTION 5: AUTOML RESULTS ─────────────────────
    if automl_results:
        elements.append(Paragraph("AutoML Comparison", heading_style))

        results = automl_results.get("results", [])
        best_model = automl_results.get("best_model", "")
        groq_explanation = automl_results.get("groq_explanation", "")
        task_type = automl_results.get("task_type", "classification")

        if results:
            if task_type == "classification":
                table_data = [["Model", "Accuracy", "F1 Score", "Time (s)"]]
                for r in results:
                    # Highlight best model row
                    table_data.append([
                        r["model"],
                        str(r.get("accuracy", "")),
                        str(r.get("f1_score", "")),
                        str(r.get("training_time", "")),
                    ])
            else:
                table_data = [["Model", "R²", "MAE", "Time (s)"]]
                for r in results:
                    table_data.append([
                        r["model"],
                        str(r.get("r2", "")),
                        str(r.get("mae", "")),
                        str(r.get("training_time", "")),
                    ])

            automl_table = Table(table_data,
                                 colWidths=[5.5*cm, 3*cm, 3*cm, 3*cm])

            # Find index of best model row to highlight it green
            best_row_idx = next(
                (i+1 for i, r in enumerate(results)
                 if r["model"] == best_model), None
            )

            table_style = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
            # Green highlight for best model row
            if best_row_idx:
                table_style.append((
                    "BACKGROUND",
                    (0, best_row_idx), (-1, best_row_idx),
                    colors.HexColor("#d4edda")
                ))

            automl_table.setStyle(TableStyle(table_style))
            elements.append(automl_table)
            elements.append(Spacer(1, 0.3*cm))
            elements.append(Paragraph(f"Best Model: {best_model}", body_style))
            elements.append(Spacer(1, 0.3*cm))
            elements.append(Paragraph("AI Explanation:", body_style))
            elements.append(Paragraph(groq_explanation, body_style))

    # Build PDF — writes into the BytesIO buffer
    doc.build(elements)

    # Return buffer contents as bytes
    buffer.seek(0)    # rewind buffer to start before reading
    return buffer.read()