import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

NUMERIC_FEATURES = [
    "tenure", "monthly_charges", "total_charges",
    "num_support_calls", "senior_citizen",
    "has_partner", "has_dependents", "contract_months"
]


def load_data(filepath="data/telecom_churn.csv"):
    df = pd.read_csv(filepath)

    X = df[NUMERIC_FEATURES]
    y = df["churned"]

    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )


def train_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_thresholds(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.1, 0.95, 0.05)

    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        preds = (probs >= t).astype(int)

        precisions.append(precision_score(y_test, preds, zero_division=0))
        recalls.append(recall_score(y_test, preds, zero_division=0))
        f1s.append(f1_score(y_test, preds, zero_division=0))

    return thresholds, np.array(precisions), np.array(recalls), np.array(f1s)


def plot_results(thresholds, p, r, f1):
    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.plot(thresholds, p, label="Precision")
    plt.plot(thresholds, r, label="Recall")
    plt.plot(thresholds, f1, label="F1 Score")

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 vs Threshold")
    plt.legend()

    plt.savefig("results/threshold_sweep.png")
    plt.close()


def analyze(thresholds, precisions, recalls, f1s):
    best_f1_idx = np.argmax(f1s)
    best_f1_threshold = thresholds[best_f1_idx]

    valid = np.where(recalls >= 0.80)[0]

    if len(valid) > 0:
        best_recall_idx = valid[np.argmax(f1s[valid])]
        best_recall_threshold = thresholds[best_recall_idx]
    else:
        best_recall_threshold = None

    print("\n--- Threshold Analysis ---")
    print(f"Best F1 threshold: {best_f1_threshold:.2f}")

    if best_recall_threshold is not None:
        print(f"Threshold with ≥80% recall: {best_recall_threshold:.2f}")
    else:
        print("No threshold achieves 80% recall")

    print("\n--- Business Recommendation ---")
    print(
        "If Petra Telecom can contact 200 customers/month:\n"
        "- We prioritize recall (catching churners) first\n"
        "- Then control precision to avoid wasted offers\n\n"
        "Recommendation: choose the lowest threshold that still keeps recall ≥ 80%, "
        "even if precision drops slightly, because missing churners is more costly than extra outreach."
    )


def main():
    X_train, X_test, y_train, y_test = load_data()

    model = train_rf(X_train, y_train)

    thresholds, p, r, f1 = evaluate_thresholds(model, X_test, y_test)

    plot_results(thresholds, p, r, f1)

    analyze(thresholds, p, r, f1)


if __name__ == "__main__":
    main()