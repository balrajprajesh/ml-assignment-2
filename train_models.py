import os
import platform
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# 1. Define the 57 official Spambase features + target column
FEATURE_NAMES = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_semicolon", "char_freq_paren", "char_freq_bracket", "char_freq_excl",
    "char_freq_dollar", "char_freq_hash",
    "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"
]
TARGET_COL = "is_spam"


def load_spambase_data():
    """Loads the UCI Spambase dataset with fallback to OpenML if direct URL is blocked."""
    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    try:
        print("Fetching Spambase dataset from UCI repository...")
        df = pd.read_csv(uci_url, header=None, names=FEATURE_NAMES + [TARGET_COL])
    except Exception:
        print("Direct UCI URL blocked/unavailable. Fetching via OpenML fallback...")
        data = fetch_openml(name="spambase", version=1, as_frame=True, parser="auto")
        df = data.frame.copy()
        df.columns = FEATURE_NAMES + [TARGET_COL]
        df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def main():
    print("=" * 75)
    print(f"Model Training Execution | Host: {platform.node()} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)

    # Load and verify dataset constraints (Min features: 12, Min instances: 500)
    df = load_spambase_data()
    print(f"Dataset Loaded Successfully -> Instances: {df.shape[0]} | Features: {df.shape[1] - 1}")

    X = df[FEATURE_NAMES]
    y = df[TARGET_COL].astype(int)

    # Stratified split with custom parameters to ensure unique experiment outputs
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=104, stratify=y
    )

    # Save test_data.csv (includes all 57 features + target column 'is_spam')
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    test_df.to_csv("test_data.csv", index=False)
    print(f"Saved 'test_data.csv' with shape: {test_df.shape}")

    # Create model directory
    os.makedirs("model", exist_ok=True)

    # Define the 5 required classification pipelines
    classifiers = {
        "Logistic Regression": (
            "logistic_regression.joblib",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.25, solver="lbfgs", max_iter=1000, random_state=104))
            ])
        ),
        "Decision Tree": (
            "decision_tree.joblib",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", DecisionTreeClassifier(max_depth=11, min_samples_split=8, criterion="gini", random_state=104))
            ])
        ),
        "KNN": (
            "knn.joblib",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance", p=2))
            ])
        ),
        "Naive Bayes": (
            "naive_bayes.joblib",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GaussianNB(var_smoothing=1e-7))
            ])
        ),
        "Support Vector Machine": (
            "svm.joblib",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(C=2.2, kernel="rbf", probability=True, random_state=104))
            ])
        )
    }

    metrics_summary = []

    print("\nTraining models and computing evaluation metrics...\n")
    for model_name, (filename, pipeline) in classifiers.items():
        # Fit model
        pipeline.fit(X_train, y_train)

        # Save model artifact
        filepath = os.path.join("model", filename)
        joblib.dump(pipeline, filepath)

        # Predict classes and probabilities on test set
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Calculate all 6 required metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        metrics_summary.append({
            "ML Model Name": model_name,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4)
        })

    # Display comparison table
    results_df = pd.DataFrame(metrics_summary)
    print(results_df.to_string(index=False))
    results_df.to_csv("model/metrics_comparison.csv", index=False)
    print("\nAll 5 models saved to './model/' and metrics exported to 'model/metrics_comparison.csv'.")


if __name__ == "__main__":
    main()