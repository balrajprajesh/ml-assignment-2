import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Page Configuration
st.set_page_config(
    page_title="Spambase Diagnostic Classifier Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

TARGET_COL = "is_spam"

MODEL_REGISTRY = {
    "Support Vector Machine": "model/svm.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "Logistic Regression": "model/logistic_regression.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib"
}


@st.cache_resource
def load_saved_models():
    """Loads all 5 pre-trained classification pipelines from disk without retraining."""
    loaded = {}
    for name, path in MODEL_REGISTRY.items():
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
    return loaded


def evaluate_loaded_model(model, X_eval, y_eval):
    """Calculates all 6 assignment evaluation metrics, predictions, and probabilities."""
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_eval, y_pred),
        "AUC": roc_auc_score(y_eval, y_prob),
        "Precision": precision_score(y_eval, y_pred, zero_division=0),
        "Recall": recall_score(y_eval, y_pred, zero_division=0),
        "F1": f1_score(y_eval, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_eval, y_pred)
    }
    return metrics, y_pred, y_prob


def main():
    st.title("🛡️ Spambase Email Threat Classification Suite")
    st.markdown(
        "Interactive evaluation workspace comparing **Five Pre-Trained Machine Learning Classifiers** "
        "on the UCI Spambase dataset (**57 continuous features**, binary target `is_spam`)."
    )

    models = load_saved_models()
    if len(models) < 5:
        st.error(
            "⚠️ Saved model files were not found in `./model/`. "
            "Please run `python train_models.py` first and ensure the `model/` folder is committed."
        )
        st.stop()

    # ---------------- SIDEBAR CONTROLS ----------------
    st.sidebar.header("⚙️ Evaluation Controls")

    # Feature a: Dataset upload option (CSV)
    uploaded_file = st.sidebar.file_uploader(
        "1. Upload Test Dataset (CSV)",
        type=["csv"],
        help="Upload a test CSV containing the 57 Spambase input features and the 'is_spam' target column."
    )

    use_bundled = st.sidebar.checkbox(
        "Use bundled 'test_data.csv' if no file is uploaded",
        value=True
    )

    # Feature b: Model selection dropdown
    selected_model_name = st.sidebar.selectbox(
        "2. Choose Classification Model",
        options=list(MODEL_REGISTRY.keys()),
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**Inference Mode:** Models are loaded directly from serialized `.joblib` pipelines "
        "and evaluated on test instances without retraining."
    )

    # ---------------- LOAD TEST DATA ----------------
    df_test = None
    data_source_label = ""

    if uploaded_file is not None:
        df_test = pd.read_csv(uploaded_file)
        data_source_label = f"Uploaded File (`{uploaded_file.name}`)"
    elif use_bundled and os.path.exists("test_data.csv"):
        df_test = pd.read_csv("test_data.csv")
        data_source_label = "Bundled Repository Test Set (`test_data.csv`)"
    else:
        st.info("👈 Please upload `test_data.csv` in the sidebar to run model inference and view metrics.")
        st.stop()

    # Validate Target Column
    if TARGET_COL not in df_test.columns:
        st.error(
            f"❌ The uploaded CSV is missing the required target column `{TARGET_COL}`. "
            "Please upload a valid test dataset containing both features and target."
        )
        st.stop()

    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL].astype(int)

    # Overview Banner
    spam_ratio = (y_test.sum() / len(y_test)) * 100
    st.success(
        f"**Active Dataset:** {data_source_label} | **Instances:** `{len(df_test)}` | "
        f"**Input Features:** `{X_test.shape[1]}` | **Spam Prevalence:** `{spam_ratio:.1f}%`"
    )

    # ---------------- TABS FOR SINGLE MODEL & BENCHMARK ----------------
    tab_single, tab_all, tab_preview = st.tabs([
        f"🔍 Active Model: {selected_model_name}",
        "📊 All 5 Models Benchmark on Test Data",
        "🗂️ Test Dataset Preview"
    ])

    # TAB 1: Selected Model Evaluation
    with tab_single:
        active_model = models[selected_model_name]
        metrics, y_pred, _ = evaluate_loaded_model(active_model, X_test, y_test)

        st.subheader(f"Performance Metrics — {selected_model_name}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        c2.metric("AUC Score", f"{metrics['AUC']:.4f}")
        c3.metric("Precision", f"{metrics['Precision']:.4f}")
        c4.metric("Recall", f"{metrics['Recall']:.4f}")
        c5.metric("F1 Score", f"{metrics['F1']:.4f}")
        c6.metric("MCC Score", f"{metrics['MCC']:.4f}")

        st.markdown("---")

        # Feature d: Confusion Matrix & Classification Report
        col_left, col_right = st.columns([1, 1.25])

        with col_left:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="YlGnBu",
                cbar=False,
                xticklabels=["Legitimate (0)", "Spam (1)"],
                yticklabels=["Legitimate (0)", "Spam (1)"],
                annot_kws={"size": 13, "weight": "bold"},
                ax=ax
            )
            ax.set_xlabel("Predicted Class", fontsize=11)
            ax.set_ylabel("Actual Class", fontsize=11)
            ax.set_title(f"{selected_model_name} Confusion Matrix", fontsize=12, pad=10)
            st.pyplot(fig)

        with col_right:
            st.markdown("#### Detailed Classification Report")
            report_dict = classification_report(
                y_test,
                y_pred,
                target_names=["Legitimate (0)", "Spam (1)"],
                output_dict=True,
                zero_division=0
            )
            report_df = pd.DataFrame(report_dict).transpose().round(4)
            st.dataframe(report_df, use_container_width=True)

    # TAB 2: All 5 Models Comparison on the Test Data
    with tab_all:
        st.subheader("Comparative Results of All 5 Models on Test Data")
        comparison_rows = []
        for m_name, m_pipe in models.items():
            m_metrics, _, _ = evaluate_loaded_model(m_pipe, X_test, y_test)
            row = {"ML Model Name": m_name}
            row.update({k: round(v, 4) for k, v in m_metrics.items()})
            comparison_rows.append(row)

        comp_df = pd.DataFrame(comparison_rows)
        st.dataframe(
            comp_df.style.highlight_max(
                subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                color="#d4edda"
            ),
            use_container_width=True,
            hide_index=True
        )

    # TAB 3: Raw Test Data Inspection
    with tab_preview:
        st.subheader("Uploaded Test Data Sample (First 20 Rows)")
        st.dataframe(df_test.head(20), use_container_width=True)


if __name__ == "__main__":
    main()