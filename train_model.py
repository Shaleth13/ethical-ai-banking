from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report
import joblib

RANDOM_STATE = 42


def generate_synthetic_data(n_samples: int = 2000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)

    age = rng.randint(21, 70, size=n_samples)
    income = rng.normal(60000, 15000, size=n_samples).clip(20000, 200000)
    loan_amount = rng.normal(200000, 50000, size=n_samples).clip(50000, 500000)
    credit_score = rng.randint(300, 900, size=n_samples)
    employment_years = rng.randint(0, 30, size=n_samples)
    gender = rng.choice(["Male", "Female"], size=n_samples)
    marital_status = rng.choice(["Single", "Married", "Divorced"], size=n_samples)

    base_score = (
        0.03 * (credit_score - 600)
        + 0.0005 * (income - 50000)
        - 0.0004 * (loan_amount - 150000)
        + 0.7 * (employment_years > 2).astype(int)
    )

    gender_bias = np.where(gender == "Female", -0.5, 0.0)

    logits = (base_score / 20.0) + gender_bias
    probs = 1.0 / (1.0 + np.exp(-logits))
    approved = rng.binomial(1, probs)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income.astype(int),
            "loan_amount": loan_amount.astype(int),
            "credit_score": credit_score,
            "employment_years": employment_years,
            "gender": gender,
            "marital_status": marital_status,
            "approved": approved,
        }
    )

    return df


def train_and_save_model() -> None:
    base_path = Path(__file__).resolve().parent
    data_dir = base_path / "data"
    models_dir = base_path / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "credit_risk_data.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = generate_synthetic_data()
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to {csv_path}")

    feature_cols = [
        "age",
        "income",
        "loan_amount",
        "credit_score",
        "employment_years",
        "gender",
        "marital_status",
    ]
    target_col = "approved"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    numeric_features = ["age", "income", "loan_amount", "credit_score", "employment_years"]
    categorical_features = ["gender", "marital_status"]

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Evaluation on test set:")
    print(classification_report(y_test, y_pred))

    model_path = models_dir / "credit_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_and_save_model()
