import pandas as pd
import mlflow
import mlflow.sklearn
import optuna

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==============================
# FEATURE LISTS
# ==============================
NUMERIC_FEATURES = ["Age", "Income", "CreditScore", "LoanAmount", "LoanTerm"]
CATEGORICAL_FEATURES = ["EmploymentType", "ResidenceType", "PreviousDefault"]


# ==============================
# LOAD DATA
# ==============================
def load_data(path):
    df = pd.read_csv(path)
    X = df.drop("RiskCategory", axis=1)
    y = df["RiskCategory"]
    return X, y


# ==============================
# BUILD PREPROCESSOR
# ==============================
def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )


# ==============================
# BUILD MODEL PIPELINE
# ==============================
def build_model(preprocessor, params):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(**params, random_state=42))
    ])


# ==============================
# OPTUNA OBJECTIVE
# ==============================
def objective(trial, X, y, preprocessor):
    with mlflow.start_run(nested=True):

        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

        model = build_model(preprocessor, {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split
        })

        scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
        accuracy = scores.mean()

        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split
        })
        mlflow.log_metric("cv_accuracy", accuracy)

        return accuracy


# ==============================
# TRAIN FINAL MODEL
# ==============================
def train_best_model(X, y, best_params, preprocessor):
    model = build_model(preprocessor, best_params)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc


# ==============================
# FULL PIPELINE RUNNER
# ==============================
def run_pipeline(data_path):
    mlflow.set_experiment("Loan_Risk_Bayesian_Optimization")

    X, y = load_data(data_path)
    preprocessor = build_preprocessor()

    with mlflow.start_run(run_name="Bayesian_Optimization"):

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X, y, preprocessor), n_trials=20)

        best_params = study.best_params
        mlflow.log_params(best_params)

        final_model, test_accuracy = train_best_model(X, y, best_params, preprocessor)
        mlflow.log_metric("test_accuracy", test_accuracy)

        mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            registered_model_name="LoanRiskModel"
        )

        print(f"✅ Best model registered with test accuracy: {test_accuracy:.4f}")
