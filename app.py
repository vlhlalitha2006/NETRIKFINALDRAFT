from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from src.explainability import explainer_service as es
from src.inference.multimodal_predict import multimodal_predict
from streamlit_option_menu import option_menu
from src.db.session import session_scope
from src.db.models import User
from src.db.auth_service import register_user, authenticate_user


CPU_DEVICE = torch.device("cpu")
EVAL_REPORT_PATH = Path("artifacts/evaluation_report.json")
FAIRNESS_REPORT_PATH = Path("artifacts/fairness_report.json")
PREDICTIONS_CSV_PATH = Path("artifacts/test_predictions_with_explanations.csv")
TRAIN_CSV_PATH = Path("data/raw/TRAIN.csv")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@400;700&display=swap');
        
        .stApp {
            background-color: #f8faff;
            color: #1e293b;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        .main .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 1200px;
        }
        
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
        }
        
        .brand-title {
            font-family: 'Outfit', sans-serif;
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        }
        
        .brand-subtitle {
            font-size: 0.95rem;
            color: #64748b;
            margin-bottom: 2rem;
            font-weight: 500;
        }
        
        /* Metric Cards */
        .metric-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover { transform: translateY(-4px); border-color: #3b82f6; }
        .metric-label { color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem; }
        .metric-value { color: #1e293b; font-size: 1.8rem; font-weight: 800; }
        
        /* Status Cards */
        .status-card { border-radius: 20px; padding: 2rem; margin-bottom: 2rem; text-align: center; border: 1px solid transparent; }
        .approved { background: #f0fdf4; border-color: #bbf7d0; color: #16a34a; }
        .rejected { background: #fef2f2; border-color: #fecaca; color: #dc2626; }
        .status-text { font-family: 'Outfit', sans-serif; font-size: 2.6rem; font-weight: 800; margin: 0; }
        
        /* Progress Bars */
        .progress-outer { width: 100%; height: 10px; border-radius: 5px; background: #e2e8f0; margin: 1rem 0; overflow: hidden; }
        .progress-inner { height: 100%; border-radius: 5px; background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%); }
        
        /* Form and Input Elements - CRITICAL VISIBILITY FIXES */
        div[data-testid="stForm"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 20px !important;
            padding: 2.5rem !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05) !important;
        }
        
        label { color: #1e293b !important; font-weight: 600 !important; }
        
        .stTextInput>div>div>input {
            background-color: #ffffff !important;
            color: #1e293b !important;
            border: 1px solid #cbd5e0 !important;
            border-radius: 8px !important;
        }

        .stTextInput>div>div>input:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 1px #3b82f6 !important;
        }
        
        /* Buttons */
        .stButton>button {
            border-radius: 10px !important;
            background-color: #3b82f6 !important;
            color: white !important;
            font-weight: 600 !important;
        }
        
        /* Metrics and Streamlit Elements Visibility */
        [data-testid="stMetricValue"] {
            color: #1e293b !important;
            font-size: 1.8rem !important;
            font-weight: 800 !important;
        }
        [data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-weight: 600 !important;
        }

        /* General Streamlit Overrides to Fix Dark-on-Light Issues */
        [data-testid="stHeader"] { background-color: rgba(248, 250, 255, 0.82); }
        .st-emotion-cache-1vt4885 { background-color: #ffffff !important; border: 1px solid #e2e8f0 !important; }
        
        /* Table Visibility Fix */
        
        /* Table Visibility Fix */
        .stTable {
            background-color: #ffffff !important;
            color: #1e293b !important;
        }
        .stTable th {
            background-color: #f1f5f9 !important;
            color: #475569 !important;
            font-weight: 700 !important;
        }
        .stTable td {
            color: #1e293b !important;
            border-bottom: 1px solid #e2e8f0 !important;
        }
        
        .pred-card {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            margin-bottom: 1.5rem !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }
        .pred-card:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
        }
        
        .badge-approved {
            background-color: #f0fdf4 !important;
            color: #16a34a !important;
            padding: 0.25rem 0.75rem !important;
            border-radius: 9999px !important;
            font-size: 0.75rem !important;
            font-weight: 700 !important;
            border: 1px solid #bbf7d0 !important;
            text-transform: uppercase !important;
        }
        .badge-rejected {
            background-color: #fef2f2 !important;
            color: #dc2626 !important;
            padding: 0.25rem 0.75rem !important;
            border-radius: 9999px !important;
            font-size: 0.75rem !important;
            font-weight: 700 !important;
            border: 1px solid #fecaca !important;
            text-transform: uppercase !important;
        }

        .small-muted { color: #64748b !important; font-size: 0.875rem !important; }
        
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_bar(prob: float) -> None:
    pct = max(0.0, min(100.0, prob * 100.0))
    st.markdown(
        f"""
        <div class="small-muted">Approval Probability</div>
        <div class="metric-value" style="font-size:2.1rem;">{pct:.2f}%</div>
        <div class="progress-outer">
          <div class="progress-inner" style="width:{pct:.2f}%"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_label(prob: float) -> str:
    if prob > 0.85:
        return "High"
    if prob >= 0.65:
        return "Moderate"
    return "Low"


def clean_feature_name(raw_name: str) -> str:
    explicit_map = {
        "num__Credit_History": "Credit History",
        "num__LoanAmount": "Loan Amount",
        "num__ApplicantIncome": "Applicant Income",
        "num__CoapplicantIncome": "Co-applicant Income",
        "num__Loan_Amount_Term": "Loan Amount Term",
    }
    if raw_name in explicit_map:
        return explicit_map[raw_name]
    if raw_name.startswith("cat__Property_Area_"):
        suffix = raw_name.replace("cat__Property_Area_", "").replace("_", " ")
        return f"Property Area ({suffix.title()})"
    if raw_name.startswith("cat__"):
        return raw_name.replace("cat__", "").replace("_", " ").title()
    if raw_name.startswith("num__"):
        return raw_name.replace("num__", "").replace("_", " ").title()
    return raw_name.replace("_", " ").title()


def build_explanation_sentence(
    decision: str,
    approval_probability: float,
    top_features: list[str],
) -> str:
    f1, f2, f3 = (top_features + ["Financial Feature"] * 3)[:3]
    if decision == "Approved":
        return (
            f"Loan approved because {f1}, {f2}, and {f3} positively influenced approval. "
            f"Overall Approval Probability was {approval_probability:.4f}."
        )
    return (
        f"Loan rejected because {f1}, {f2}, and {f3} reduced approval likelihood. "
        f"Overall Approval Probability was {approval_probability:.4f}."
    )


def sidebar_nav() -> str:
    with st.sidebar:
        st.markdown('<div class="brand-title">NETRIK</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-subtitle">Multimodal Loan Risk AI</div>', unsafe_allow_html=True)

        sections = [
            "Loan Predictor",
            "Model Performance",
            "Fairness Analysis",
            "Predictions Explorer",
            "Data Insights",
            "Testing Guide",
        ]
        
        if st.session_state.get("user") and st.session_state["user"].get("role") == "employee":
            sections.append("All Applications")

        return option_menu(
            menu_title=None,
            options=sections,
            icons=["bank", "speedometer2", "shield-check", "table", "bar-chart-line", "clipboard-check"],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#ffffff"},
                "icon": {"color": "#3b82f6", "font-size": "16px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "4px 0",
                    "--hover-color": "#f1f5f9",
                    "border-radius": "8px",
                    "color": "#475569",
                    "font-weight": "500",
                },
                "nav-link-selected": {
                    "background-color": "#eff6ff",
                    "color": "#2563eb",
                    "font-weight": "600",
                },
            },
        )


def page_loan_predictor() -> None:
    st.subheader("Loan Predictor")
    st.caption(f"Inference device: `{CPU_DEVICE}`")
    st.markdown("### Applicant Input")

    # Loan_ID is used only for explanation lookup and does not affect prediction.
    loan_id_for_explain = st.text_input(
        "Loan_ID (optional, explanation only)",
        value="",
        help="Used only to fetch existing explanations. Prediction uses structured features only.",
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Financial Features")
        applicant_income = st.number_input("ApplicantIncome", min_value=0.0, value=5000.0, step=100.0)
        coapplicant_income = st.number_input("CoapplicantIncome", min_value=0.0, value=1500.0, step=100.0)
        loan_amount = st.number_input("LoanAmount", min_value=1.0, value=150.0, step=1.0)
        loan_term = st.number_input("Loan_Amount_Term", min_value=1.0, value=360.0, step=1.0)
        credit_history = st.selectbox("Credit_History", options=[1.0, 0.0], index=0)

    with right:
        st.markdown("#### Personal Features")
        gender = st.selectbox("Gender", options=["Male", "Female"])
        married = st.selectbox("Married", options=["Yes", "No"])
        dependents = st.selectbox("Dependents", options=["0", "1", "2", "3+"])
        education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self_Employed", options=["Yes", "No"])
        property_area = st.selectbox("Property_Area", options=["Urban", "Semiurban", "Rural"])

    if st.button("Predict", type="primary", use_container_width=True):
        input_df = pd.DataFrame(
            [
                {
                    "Gender": gender,
                    "Married": married,
                    "Dependents": dependents,
                    "Education": education,
                    "Self_Employed": self_employed,
                    "ApplicantIncome": float(applicant_income),
                    "CoapplicantIncome": float(coapplicant_income),
                    "LoanAmount": float(loan_amount),
                    "Loan_Amount_Term": float(loan_term),
                    "Credit_History": float(credit_history),
                    "Property_Area": property_area,
                }
            ]
        )
        try:
            # IMPORTANT: Loan_ID is intentionally not passed to keep prediction dependent
            # only on structured features (manual mode stability).
            pred = multimodal_predict(df_row=input_df, loan_id=None, debug=False)
            st.session_state["last_prediction"] = pred
            st.session_state.setdefault("live_predictions", [])
            st.session_state["live_predictions"].append(
                {
                    "Loan_ID": loan_id_for_explain.strip() or "Manual_Input",
                    "Predicted_Loan_Status": str(pred["prediction"]),
                    "Approval_Probability": float(pred["approval_probability"]),
                    "Top_Tabular_Features": "N/A",
                    "Top_Sequence_Features": "N/A",
                    "Graph_Influence_Score": np.nan,
                    "Explanation_Text": "Live prediction generated from dashboard input.",
                    "Source": "Live Prediction",
                }
            )
            
            # Auto-update status for logged-in customers
            if st.session_state.get("user") and st.session_state["user"].get("role") == "customer":
                with session_scope() as session:
                    current_user = session.query(User).filter(User.id == st.session_state["user"]["id"]).first()
                    if current_user:
                        current_user.loan_status = str(pred["prediction"])
                        session.commit()
            
            st.success("Prediction completed.")
        except Exception as exc:
            st.error("Prediction failed. Please verify model artifacts and inputs.")
            st.exception(exc)

    st.markdown("### Decision Output")
    pred = st.session_state.get("last_prediction")
    if pred is None:
        st.info("Submit applicant input to view prediction.")
        return

    decision = str(pred["prediction"])
    approval_probability = float(pred["approval_probability"])
    status_class = "approved" if decision == "Approved" else "rejected"
    status_text_class = "status-approved" if decision == "Approved" else "status-rejected"
    st.markdown(
        f"""
        <div class="status-card {status_class}">
          <p class="status-text {status_text_class}">{decision.upper()}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    probability_bar(approval_probability)
    metric_card("Confidence", confidence_label(approval_probability))

    st.markdown("### Explainability of Loan Status")
    if loan_id_for_explain.strip():
        try:
            explanation = es.explain_applicant(loan_id_for_explain.strip())
            c1, c2, c3 = st.columns([1.3, 1.3, 1.0], gap="large")
            with c1:
                st.markdown("**Tabular Explanations**")
                st.dataframe(pd.DataFrame(explanation.get("tabular_explanations", [])), use_container_width=True)
            with c2:
                st.markdown("**Sequence Explanations**")
                st.dataframe(pd.DataFrame(explanation.get("sequence_explanations", [])), use_container_width=True)
            with c3:
                st.markdown("**Graph Influence Score**")
                score = explanation.get("graph_influence_score")
                st.metric("Score", f"{float(score):.4f}" if score is not None else "N/A")
            with st.container():
                tab_features = [
                    clean_feature_name(str(item.get("feature", "")))
                    for item in explanation.get("tabular_explanations", [])[:3]
                ]
                st.success(
                    build_explanation_sentence(
                        decision=decision,
                        approval_probability=approval_probability,
                        top_features=tab_features,
                    )
                )
        except Exception:
            try:
                pseudo = es._compute_tabular_shap(
                    pd.DataFrame(
                        [
                            {
                                "Loan_ID": "MANUAL_INPUT",
                                "Gender": gender,
                                "Married": married,
                                "Dependents": dependents,
                                "Education": education,
                                "Self_Employed": self_employed,
                                "ApplicantIncome": float(applicant_income),
                                "CoapplicantIncome": float(coapplicant_income),
                                "LoanAmount": float(loan_amount),
                                "Loan_Amount_Term": float(loan_term),
                                "Credit_History": float(credit_history),
                                "Property_Area": property_area,
                            }
                        ]
                    )
                )
                tab_features = [clean_feature_name(str(item.get("feature", ""))) for item in pseudo[:3]]
                st.info(build_explanation_sentence(decision, approval_probability, tab_features))
            except Exception:
                st.info("Explainability currently unavailable.")
    else:
        try:
            pseudo = es._compute_tabular_shap(
                pd.DataFrame(
                    [
                        {
                            "Loan_ID": "MANUAL_INPUT",
                            "Gender": gender,
                            "Married": married,
                            "Dependents": dependents,
                            "Education": education,
                            "Self_Employed": self_employed,
                            "ApplicantIncome": float(applicant_income),
                            "CoapplicantIncome": float(coapplicant_income),
                            "LoanAmount": float(loan_amount),
                            "Loan_Amount_Term": float(loan_term),
                            "Credit_History": float(credit_history),
                            "Property_Area": property_area,
                        }
                    ]
                )
            )
            tab_features = [clean_feature_name(str(item.get("feature", ""))) for item in pseudo[:3]]
            st.info(build_explanation_sentence(decision, approval_probability, tab_features))
        except Exception:
            st.info("Enter Loan_ID to view full explainability output, or run with available artifacts.")


def page_model_performance() -> None:
    st.subheader("Model Performance")
    report = load_json(EVAL_REPORT_PATH)
    if not report:
        st.warning("Missing evaluation report at artifacts/evaluation_report.json")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{report.get('accuracy', 0):.4f}")
    c2.metric("Precision", f"{report.get('precision', 0):.4f}")
    c3.metric("Recall", f"{report.get('recall', 0):.4f}")
    c4.metric("F1", f"{report.get('f1_score', 0):.4f}")
    c5.metric("ROC-AUC", f"{report.get('roc_auc', 0):.4f}")

    radar_metrics = {
        "Accuracy": report.get("accuracy", 0.0),
        "Precision": report.get("precision", 0.0),
        "Recall": report.get("recall", 0.0),
        "F1": report.get("f1_score", 0.0),
        "ROC-AUC": report.get("roc_auc", 0.0),
    }
    radar_df = pd.DataFrame(
        {"metric": list(radar_metrics.keys()), "value": list(radar_metrics.values())}
    )
    radar_fig = px.line_polar(
        radar_df,
        r="value",
        theta="metric",
        line_close=True,
        template="plotly_white",
    )
    radar_fig.update_traces(fill="toself", line_color="#4ea5ff")
    radar_fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20), 
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black")
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    cm = report.get("confusion_matrix", [[0, 0], [0, 0]])
    cm_fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred 0", "Pred 1"],
            y=["Actual 0", "Actual 1"],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
        )
    )
    cm_fig.update_layout(
        title="Confusion Matrix",
        template="plotly_white",
        margin=dict(l=20, r=20, t=45, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black")
    )
    cm_fig.update_xaxes(tickfont=dict(color="black"), title_font=dict(color="black"))
    cm_fig.update_yaxes(tickfont=dict(color="black"), title_font=dict(color="black"))
    st.plotly_chart(cm_fig, use_container_width=True)


def _plot_group_chart(group_name: str, rows: list[dict], global_approval: float) -> None:
    if not rows:
        return
    chart_df = pd.DataFrame(rows).copy()
    chart_df["deviation"] = (chart_df["approval_rate"] - global_approval).abs()
    chart_df["flag"] = np.where(chart_df["deviation"] > 0.15, "Deviation > 15%", "Within limit")
    chart_df["label"] = chart_df["value"].astype(str)
    fig = px.bar(
        chart_df,
        x="label",
        y="approval_rate",
        color="deviation",
        color_continuous_scale="Blues",
        labels={"approval_rate": "Approval Rate", "deviation": "Abs Deviation"},
        template="plotly_white",
        title=f"{group_name} Approval Rate Comparison",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=45, b=20), 
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black")
    )
    fig.update_xaxes(tickfont=dict(color="black"), title_font=dict(color="black"))
    fig.update_yaxes(tickfont=dict(color="black"), title_font=dict(color="black"))
    st.plotly_chart(fig, use_container_width=True)


def page_fairness_analysis() -> None:
    st.subheader("Fairness Analysis")
    report = load_json(FAIRNESS_REPORT_PATH)
    if not report:
        st.warning("Missing fairness report at artifacts/fairness_report.json")
        return

    global_metrics = report.get("global_metrics", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Global Approval Rate", f"{global_metrics.get('approval_rate', 0):.4f}")
    c2.metric("Global FPR", f"{global_metrics.get('false_positive_rate', 0):.4f}")
    c3.metric("Global FNR", f"{global_metrics.get('false_negative_rate', 0):.4f}")

    group_metrics = report.get("group_metrics", {})
    global_approval = float(global_metrics.get("approval_rate", 0.0))
    for name in ["Gender", "Married", "Dependents", "Property_Area"]:
        _plot_group_chart(name, group_metrics.get(name, []), global_approval)

    alerts = report.get("alerts", [])
    st.markdown("### Deviation Alerts (>15%)")
    if alerts:
        st.error(f"Detected {len(alerts)} fairness alerts.")
        st.dataframe(pd.DataFrame(alerts), use_container_width=True)
    else:
        st.success("No fairness alerts above threshold.")


def page_predictions_explorer() -> None:
    st.subheader("Predictions Explorer")
    pred_df = load_csv(PREDICTIONS_CSV_PATH)
    if pred_df.empty:
        st.warning("Missing historical predictions CSV. Showing live predictions only.")
        pred_df = pd.DataFrame()
    else:
        pred_df["Source"] = "Historical TEST"

    live_predictions = st.session_state.get("live_predictions", [])
    live_df = pd.DataFrame(live_predictions) if live_predictions else pd.DataFrame()
    combined_df = pd.concat([pred_df, live_df], axis=0, ignore_index=True)
    if combined_df.empty:
        st.info("No predictions available yet.")
        return

    rows_to_show = st.slider("Rows to display", min_value=5, max_value=50, value=12, step=1)
    view_df = combined_df.tail(rows_to_show).iloc[::-1].copy()

    for _, row in view_df.iterrows():
        decision = str(row.get("Predicted_Loan_Status", "Unknown"))
        prob = float(row.get("Approval_Probability", 0.0))
        badge_class = "badge-approved" if decision == "Approved" else "badge-rejected"
        badge = "APPROVED" if decision == "Approved" else "REJECTED"
        source = str(row.get("Source", "Historical TEST"))
        st.markdown(
            f"""
            <div class="pred-card">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div><b>{row.get('Loan_ID', 'N/A')}</b></div>
                <div class="{badge_class}">{badge}</div>
              </div>
              <div class="small-muted" style="margin-top:0.25rem;">{source}</div>
              <div style="margin-top:0.4rem;" class="small-muted">Approval Probability: {prob * 100:.2f}%</div>
              <div class="progress-outer"><div class="progress-inner" style="width:{prob * 100:.2f}%"></div></div>
              <div style="margin-top:0.55rem;" class="small-muted">{row.get('Explanation_Text', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_data_insights() -> None:
    st.subheader("Data Insights")
    train_df = load_csv(TRAIN_CSV_PATH)
    if train_df.empty:
        st.error("Missing training data at data/raw/TRAIN.csv")
        return
    if "Loan_Status" not in train_df.columns:
        st.error("TRAIN.csv missing Loan_Status column.")
        return

    total = int(len(train_df))
    approved = int((train_df["Loan_Status"].astype(str) == "Y").sum())
    rejected = int((train_df["Loan_Status"].astype(str) == "N").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Applicants", f"{total}")
    c2.metric("Approved Count", f"{approved}")
    c3.metric("Rejected Count", f"{rejected}")

    hist_fig = px.histogram(
        train_df,
        x="ApplicantIncome",
        nbins=35,
        title="Applicant Income Distribution",
        template="plotly_white",
        color_discrete_sequence=["#2563eb"],
    )
    hist_fig.update_layout(
        margin=dict(l=20, r=20, t=45, b=20), 
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black")
    )
    hist_fig.update_xaxes(tickfont=dict(color="black"), title_font=dict(color="black"))
    hist_fig.update_yaxes(tickfont=dict(color="black"), title_font=dict(color="black"))
    st.plotly_chart(hist_fig, use_container_width=True)

    area_counts = (
        train_df["Property_Area"]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("Property_Area")
        .reset_index(name="count")
    )
    donut_fig = px.pie(
        area_counts,
        names="Property_Area",
        values="count",
        hole=0.55,
        title="Property Area Distribution",
        template="plotly_white",
        color_discrete_sequence=["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"],
    )
    donut_fig.update_layout(
        margin=dict(l=20, r=20, t=45, b=20), 
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black")
    )
    st.plotly_chart(donut_fig, use_container_width=True)


def page_testing_guide() -> None:
    st.subheader("Testing Guide")
    st.markdown("Use these feature value ranges to test and verify the model's prediction logic in the **Loan Predictor**.")
    
    data = [
        {"Feature": "Credit History", "best case": "1.0", "worst case": "0.0"},
        {"Feature": "Applicant Income", "best case": "8000+", "worst case": "1500 or less"},
        {"Feature": "Loan Amount", "best case": "100 or less", "worst case": "500+"},
        {"Feature": "Dependents", "best case": "0", "worst case": "3"},
        {"Feature": "Education", "best case": "Graduate", "worst case": "Not Graduate"},
        {"Feature": "Self Employed", "best case": "No", "worst case": "Yes"},
        {"Feature": "Property Area", "best case": "Urban", "worst case": "Rural"},
        {"Feature": "Married", "best case": "Yes", "worst case": "No"},
    ]
    df = pd.DataFrame(data)
    st.table(df)

def page_all_applications() -> None:
    st.subheader("All Applications")
    st.markdown("This section is restricted to **Employee** users. Here you can monitor all customer applications and their status.")
    
    with session_scope() as session:
        # Filter for customers only as requested
        users = session.query(User).filter(User.role == "customer").all()
        if not users:
            st.info("No customer applications found.")
        else:
            user_data = [
                {
                    "ID": u.id,
                    "Email": u.email,
                    "Loan Status": u.loan_status,
                    "Registered At": u.created_at.strftime("%Y-%m-%d %H:%M:%S")
                }
                for u in users
            ]
            st.table(pd.DataFrame(user_data))

def page_registration() -> None:
    st.subheader("Create an Account")
    with st.form("registration_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        role = st.selectbox("I am a:", options=["Customer", "Employee"])
        submit = st.form_submit_button("Register")
        
        if submit:
            if not email or not password:
                st.error("Please fill in all fields.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                with session_scope() as session:
                    # Convert Display Role to storage value
                    role_val = role.lower()
                    user = register_user(session, email, password, role=role_val)
                    if user:
                        st.success(f"Registration successful as {role}! Please log in.")
                    else:
                        st.error("Email already registered.")

def page_login() -> None:
    st.subheader("Login to NETRIK")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            with session_scope() as session:
                user = authenticate_user(session, email, password)
                if user:
                    st.session_state["user"] = {
                        "id": user.id, 
                        "email": user.email,
                        "role": user.role # Store role in session
                    }
                    st.success(f"Welcome back, {user.email} ({user.role.title()})!")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

def main() -> None:
    st.set_page_config(page_title="NETRIK | Multimodal Loan Risk AI", page_icon=":bank:", layout="wide")
    inject_css()

    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None
    if "live_predictions" not in st.session_state:
        st.session_state["live_predictions"] = []

    if st.session_state["user"] is None:
        selected = option_menu(
            menu_title="Welcome to NETRIK",
            options=["Login", "Register"],
            icons=["box-arrow-in-right", "person-plus"],
            orientation="horizontal",
        )
        if selected == "Login":
            page_login()
        else:
            page_registration()
    else:
        with st.sidebar:
            role_display = st.session_state['user']['role'].title()
            st.info(f"Logged in as: {st.session_state['user']['email']}\n\nID: {st.session_state['user']['id']} | Role: {role_display}")
            if st.button("Logout"):
                st.session_state["user"] = None
                st.rerun()

        selected = sidebar_nav()
        if selected == "Loan Predictor":
            page_loan_predictor()
        elif selected == "Model Performance":
            page_model_performance()
        elif selected == "Fairness Analysis":
            page_fairness_analysis()
        elif selected == "Predictions Explorer":
            page_predictions_explorer()
        elif selected == "Data Insights":
            page_data_insights()
        elif selected == "Testing Guide":
            page_testing_guide()
        else:
            page_all_applications()


if __name__ == "__main__":
    main()
