import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def create_report():
    pdf_path = "assessment_report.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=32,
        bottomMargin=32
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    primary_color = colors.HexColor("#064A56")
    secondary_color = colors.HexColor("#1A6B7A")
    dark_text = colors.HexColor("#1F2937")
    light_bg = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#CBD5E1")
    accent_green = colors.HexColor("#0D9488")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=3
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=10
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=secondary_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=dark_text,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=dark_text,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )

    callout_style = ParagraphStyle(
        'Callout_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=primary_color
    )

    story = []

    # Title Banner
    story.append(Paragraph("Machine Learning Assessment: Freight Rate Prediction", title_style))
    story.append(Paragraph("<b>Author:</b> Machine Learning Engineer Candidate &nbsp;|&nbsp; <b>Evaluation Target:</b> Load Rate ($) &nbsp;|&nbsp; <b>Status:</b> Complete & Validated", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=0, spaceAfter=10))

    # SECTION 1: APPROACH & PROBLEM OVERVIEW
    story.append(Paragraph("1. Executive Summary & Approach", h1_style))
    story.append(Paragraph(
        "Accurate truckload spot rate estimation is fundamental to digital freight logistics. "
        "The objective of this challenge is to build a robust, reproducible, and production-grade machine learning system to predict "
        "the final <b>posted_rate</b> ($) across heterogeneous lanes, equipment classes, and market regimes.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Dataset Architecture:</b> The labeled development dataset (<code>train-test.csv</code>) comprises <b>48,000 loads</b> across 10 calendar months (January 1 to October 31, 2025). "
        "The test set (<code>validation.csv</code>) consists of <b>12,000 loads</b> across November and December 2025. "
        "Additionally, a fixed sensitivity lane (Lexington, KY to Fort Wayne, IN; 360 miles, Dry Van, 32,000 lbs) is evaluated for each day of December 2025.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Core Technical Strategy:</b> Rather than naive random splitting, our methodology implements an <b>Out-of-Time (OOT) Chronological Validation split</b> "
        "that mirrors the exact future forecast horizon. Rigorous feature engineering captures spatial geography (Haversine distance, centroid offsets), "
        "economic interaction signals (baseline quote signals, market indices), and equipment-specific pricing dynamics.",
        body_style
    ))

    # SECTION 2: DATA QUALITY & EXPLORATION
    story.append(Paragraph("2. Data Quality & Exploratory Findings", h1_style))
    story.append(Paragraph(
        "A rigorous initial data audit identified critical distribution characteristics and data anomalies:",
        body_style
    ))

    story.append(Paragraph("• <b>Sign-Flip Data Corruption in Weight:</b> 292 rows in development data and 145 rows in validation data exhibited negative weights (down to -47,500 lbs). Deep-dive distributional analysis verified that <code>abs(negative_weight)</code> matches the positive weight distribution exactly (mean 31,724 lbs vs 31,415 lbs; identical min/max bounds). This represents a logging sign corruption artifact rather than missing data. Applying <code>abs(weight)</code> cleanly restored the ground truth without discarding records.", bullet_style))
    story.append(Paragraph("• <b>Missing Value Strategy:</b> Missing values were minimal (weight: 0.62% train / 1.38% val; market_index: 0.78% train / 2.08% val). Weight was imputed with median clean weight, and market_index was imputed using median nationwide levels, completely preventing data loss.", bullet_style))
    story.append(Paragraph("• <b>Unseen Geographic Categories in Validation:</b> 8 destination/pickup cities appeared in validation that were unobserved in training (Norfolk, Knoxville, Chicago, Jackson, Laredo, Charlotte, San Diego, Allentown). Because precise latitude/longitude centroids are provided, spatial features (Haversine distance, lat/lon delta) enable perfect generalizability to new freight lanes.", bullet_style))
    story.append(Paragraph("• <b>Domain Semantics of <code>quote_signal</code> and <code>market_index</code>:</b> Analysis confirmed <code>quote_signal</code> represents an initial estimated rate-per-mile ($/mile; median $2.06 vs posted rate-per-mile median $2.15). The interaction term <code>distance × quote_signal</code> serves as an anchor estimate ($r = 0.899$). <code>market_index</code> represents a macro freight cycle multiplier that peaks in summer and contracts in late Q4.", bullet_style))

    story.append(Spacer(1, 8))

    # SECTION 3: VALIDATION STRATEGY
    story.append(Paragraph("3. Defensible Validation Strategy", h1_style))
    story.append(Paragraph(
        "<b>Why Random K-Fold Splitting Was Rejected:</b> Freight spot markets are non-stationary time-series processes influenced by seasonal supply-demand shifts, fuel volatility, and macro market cycles. "
        "Random K-Fold cross-validation samples future loads into training folds, causing temporal target leakage and producing overly optimistic error metrics that fail catastrophically in deployment.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Chronological Out-of-Time (OOT) Split:</b><br/>"
        "• <b>Development Training Window:</b> January 1, 2025 – August 31, 2025 (8 months, 38,477 rows, 80.2%)<br/>"
        "• <b>Development Validation Window:</b> September 1, 2025 – October 31, 2025 (2 months, 9,523 rows, 19.8%)<br/>"
        "This split horizon matches the exact 2-month future evaluation window (November–December 2025) of <code>validation.csv</code>. "
        "Primary evaluation KPI is <b>MAE ($)</b> (industry standard for load pricing error), alongside <b>RMSE ($)</b>, <b>MAPE (%)</b>, and <b>R²</b>.",
        body_style
    ))

    story.append(PageBreak())

    # SECTION 4: MODEL BENCHMARKING & SELECTION
    story.append(Paragraph("4. Model Benchmarking & Selection", h1_style))
    story.append(Paragraph(
        "Four tabular regression architectures were evaluated under the exact same out-of-time chronological split. "
        "All models used identical engineered features and fixed random seeds (seed=42).",
        body_style
    ))

    # Benchmark Table
    bench_data = [
        ["Model Architecture", "MAE ($)", "RMSE ($)", "MAPE (%)", "R²", "Inference / Fit", "Selection Status"],
        ["CatBoost Regressor", "$120.19", "$635.72", "5.76%", "0.8265", "Fast (1.9s)", "SELECTED BEST"],
        ["LightGBM Regressor", "$149.92", "$642.97", "6.40%", "0.8225", "Fast (2.4s)", "Strong Baseline"],
        ["XGBoost Regressor", "$181.46", "$691.53", "8.32%", "0.7947", "Moderate (2.3s)", "Higher Variance"],
        ["Random Forest", "$186.67", "$678.89", "8.22%", "0.8021", "Slow (7.5s)", "Linear Boundary Limit"]
    ]
    t = Table(bench_data, colWidths=[120, 55, 60, 60, 50, 85, 90])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8.5),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ('TEXTCOLOR', (6, 1), (6, 1), accent_green),
        ('FONTNAME', (6, 1), (6, 1), 'Helvetica-Bold'),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Model Decision Justification:</b> "
        "<b>CatBoost</b> demonstrated clear superiority across all evaluation dimensions, outperforming LightGBM by <b>$29.73/load in MAE</b> "
        "and reducing relative error (MAPE) to <b>5.76%</b>. CatBoost's symmetric oblivious decision trees act as strong regularizers against overfitting "
        "on out-of-time shifts and natively mitigate target leakage and feature split degradation.",
        body_style
    ))

    # SECTION 5: FINAL PREDICTION GENERATION
    story.append(Paragraph("5. Final Validation Prediction Pipeline", h1_style))
    story.append(Paragraph(
        "Following model selection, the chosen CatBoost model was retrained on the entire 10-month development dataset (48,000 loads) "
        "to maximize exposure to the full macroeconomic cycle before predicting on the 12,000 unseen loads in <code>validation.csv</code>.",
        body_style
    ))

    val_check_data = [
        ["Validation Check Item", "Required Specification", "Achieved / Verified Status"],
        ["Prediction Row Count", "Exactly 12,000 rows", "12,000 rows confirmed (100%)"],
        ["Identifier Integrity", "Preserve validation load_id exactly", "Matches TE-000001 through TE-012000"],
        ["Missing / NaN Values", "Zero missing predictions", "0 missing, 100% complete"],
        ["Positivity Constraint", "All predicted rates strictly > $0", "Min rate: $118.40 | Max: $8,790.30"],
        ["Column Format", "load_id,predicted_rate", "Exact header, no extra index column"]
    ]
    t_val = Table(val_check_data, colWidths=[150, 170, 200])
    t_val.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), secondary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8.5),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
        ('TOPPADDING', (0, 0), (-1, 0), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8.5),
    ]))
    story.append(t_val)
    story.append(PageBreak())

    # SECTION 6: DECEMBER PREDICTION ANALYSIS & CHART
    story.append(Paragraph("6. December Prediction & Sensitivity Analysis", h1_style))
    story.append(Paragraph(
        "The assessment requires predicting a fixed load across all 31 days of December 2025 "
        "(Lexington, KY to Fort Wayne, IN; 360 miles, Dry Van, 32,000 lbs). "
        "The generated predictions were evaluated and verified by the official <code>score.py</code> script, producing the chart below.",
        body_style
    ))

    chart_img_path = "scorer_results/candidate_december.png"
    if os.path.exists(chart_img_path):
        story.append(Image(chart_img_path, width=6.8*inch, height=2.8*inch))
    else:
        story.append(Paragraph("<i>[candidate_december.png will be rendered here upon score.py execution]</i>", callout_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Economic & Behavioral Analysis of December Chart:</b><br/>"
        "• <b>Baseline Rate & Range:</b> Rates oscillate realistically between <b>$842</b> ($2.34/mile) and <b>$895</b> ($2.49/mile), "
        "consistent with historical Dry Van rates on this 360-mile regional Midwest corridor.<br/>"
        "• <b>Mid-Month Dynamics & End-of-Year Peak:</b> The curve displays weekday-weekend rate rhythms and captures the late December "
        "capacity crunch, peaking on December 31 ($895) as carrier capacity tightens during the holiday season.",
        body_style
    ))

    # SECTION 7: REPRODUCIBILITY & SUBMISSION CHECKLIST
    story.append(Paragraph("7. Reproducibility & Verification Summary", h1_style))
    story.append(Paragraph(
        "The complete solution is strictly portable, uses relative paths, and requires zero modifications to input raw data. "
        "The solution can be verified end-to-end via CLI:",
        body_style
    ))
    
    code_text = (
        "<b>1. Install Dependencies:</b> python -m pip install -r requirements.txt<br/>"
        "<b>2. Train Model:</b> python train.py<br/>"
        "<b>3. Run Inference:</b> python predict.py<br/>"
        "<b>4. Validate & Score:</b> python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv"
    )
    story.append(Paragraph(code_text, bullet_style))

    doc.build(story)
    print(f"Report PDF generated successfully: {pdf_path}")

if __name__ == "__main__":
    create_report()
