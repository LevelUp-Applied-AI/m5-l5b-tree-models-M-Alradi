import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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


def get_mdi_importance(model, feature_names):
    return dict(sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    ))


def get_permutation_importance(model, X_test, y_test, feature_names):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="f1"
    )

    return dict(sorted(
        zip(feature_names, result.importances_mean),
        key=lambda x: x[1],
        reverse=True
    ))


def plot_comparison(mdi, perm):
    os.makedirs("results", exist_ok=True)

    top_features = list(dict(list(mdi.items())[:10]).keys())

    mdi_vals = [mdi[f] for f in top_features]
    perm_vals = [perm.get(f, 0) for f in top_features]

    x = np.arange(len(top_features))

    plt.figure(figsize=(10, 5))

    plt.bar(x - 0.2, mdi_vals, width=0.4, label="MDI (Gini importance)")
    plt.bar(x + 0.2, perm_vals, width=0.4, label="Permutation importance")

    plt.xticks(x, top_features, rotation=45)
    plt.ylabel("Importance")
    plt.title("MDI vs Permutation Importance (Top 10 Features)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/permutation_vs_mdi.png")
    plt.close()


def explanation():
    print("\n--- Explanation ---")
    print(
        '''
        The disagreement between MDI (Gini importance) and Permutation importance occurs primarily because they measure different aspects of feature influence. 
        MDI is calculated during the training process based on how much each feature decreases impurity in tree nodes; it is inherently biased toward high-cardinality features (variables with many unique values) because the algorithm can easily "overfit" by splitting on them repeatedly. 
        In contrast, Permutation importance measures the drop in model performance when a feature's values are shuffled on unseen, validation data. 
        Consequently, MDI often overestimates the importance of features that allow for excessive splitting, 
        while Permutation importance provides a more realistic view of a feature’s actual predictive power on new data, even showing negative values if a feature is actually detrimental to the model's performance.'''
    )


def main():
    X_train, X_test, y_train, y_test = load_data()

    model = train_rf(X_train, y_train)

    mdi = get_mdi_importance(model, NUMERIC_FEATURES)
    perm = get_permutation_importance(model, X_test, y_test, NUMERIC_FEATURES)

    plot_comparison(mdi, perm)

    explanation()


if __name__ == "__main__":
    main()