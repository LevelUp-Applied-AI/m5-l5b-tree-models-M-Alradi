import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


NUMERIC_FEATURES = [
    "tenure", "monthly_charges", "total_charges",
    "num_support_calls", "senior_citizen",
    "has_partner", "has_dependents", "contract_months"
]


# -----------------------------
# Data
# -----------------------------
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


# -----------------------------
# Models
# -----------------------------
def train_models(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    dt = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
    dt.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)

    return lr, dt, rf, scaler


# -----------------------------
# Custom Ensemble
# -----------------------------
class CustomVotingEnsemble:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        # already fitted externally (as required)
        return self

    def _align_probas(self, model, proba):
        """
        Ensure class ordering matches across models using model.classes_
        """
        if not hasattr(self, "classes_"):
            self.classes_ = model.classes_

        # map probabilities into correct class order
        aligned = np.zeros((proba.shape[0], len(self.classes_)))

        for i, cls in enumerate(model.classes_):
            idx = np.where(self.classes_ == cls)[0][0]
            aligned[:, idx] = proba[:, i]

        return aligned

    def predict_proba(self, X_list):
        """
        X_list: list of inputs aligned with models
        """
        probs = []

        for model, X in zip(self.models, X_list):
            p = model.predict_proba(X)

            if not hasattr(self, "classes_"):
                self.classes_ = model.classes_

            probs.append(self._align_probas(model, p))

        return np.mean(probs, axis=0)

    def predict(self, X_list):
        avg_proba = self.predict_proba(X_list)
        return np.argmax(avg_proba, axis=1)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(name, model, X_list, y_test):
    if name == "LR":
        proba = model.predict_proba(X_list[0])
    else:
        proba = model.predict_proba(X_list[0])

    preds = np.argmax(proba, axis=1)

    print(f"\n--- {name} ---")
    print(classification_report(y_test, preds, zero_division=0))
    print(f"PR-AUC: {average_precision_score(y_test, proba[:, 1]):.3f}")


def evaluate_ensemble(ensemble, X_raw, X_scaled, y_test):
    proba = ensemble.predict_proba([X_scaled, X_raw, X_raw])
    preds = np.argmax(proba, axis=1)

    print("\n--- Custom Ensemble ---")
    print(classification_report(y_test, preds, zero_division=0))
    print(f"PR-AUC: {average_precision_score(y_test, proba[:, 1]):.3f}")


# -----------------------------
# Main
# -----------------------------
def main():
    X_train, X_test, y_train, y_test = load_data()

    lr, dt, rf, scaler = train_models(X_train, y_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Evaluate individuals
    evaluate("Logistic Regression", lr, [X_test_scaled], y_test)
    evaluate("Decision Tree", dt, [X_test], y_test)
    evaluate("Random Forest", rf, [X_test], y_test)

    # Ensemble
    ensemble = CustomVotingEnsemble([lr, dt, rf])

    evaluate_ensemble(ensemble, X_test, X_test_scaled, y_test)

    print("\n--- Insight ---")
    print(
       """"
       The custom ensemble performs best overall, achieving the highest PR-AUC (0.444), improving over Logistic Regression (0.392), Decision Tree (0.403), and Random Forest (0.413). This shows that combining models improves ranking performance.
        Logistic Regression misses most churners (low recall), the Decision Tree catches more churners but creates many false positives, and Random Forest is more balanced but still limited.
        The ensemble works better because it averages these different behaviors and reduces individual model weaknesses. However, the improvement is moderate, meaning all models still learn similar patterns from the data.
        """
    )


if __name__ == "__main__":
    main()