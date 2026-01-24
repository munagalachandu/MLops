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

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\muni8\Desktop\mlops\Data\loan_risk_data.csv")

X = df.drop("RiskCategory", axis=1)
y = df["RiskCategory"]

numeric_features = ["Age", "Income", "CreditScore", "LoanAmount", "LoanTerm"]
categorical_features = ["EmploymentType", "ResidenceType", "PreviousDefault"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# -----------------------------
# 2. Set MLflow Experiment
# -----------------------------
mlflow.set_experiment("Loan_Risk_Bayesian_Optimization")

# -----------------------------
# 3. Define Optuna Objective
# -----------------------------
def objective(trial):
    with mlflow.start_run(nested=True):  # 👈 Each trial logged

        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            ))
        ])

        scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
        accuracy = scores.mean()

        # Log trial data
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_metric("cv_accuracy", accuracy)

        return accuracy

# -----------------------------
# 4. Run Bayesian Optimization
# -----------------------------
with mlflow.start_run(run_name="Bayesian_Optimization"):

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    mlflow.log_params(best_params)

    print("Best Parameters Found:", best_params)

    # -----------------------------
    # 5. Train Final Best Model
    # -----------------------------
    final_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            **best_params,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    final_model.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, final_model.predict(X_test))

    mlflow.log_metric("test_accuracy", test_accuracy)

    # -----------------------------
    # 6. Register Best Model
    # -----------------------------
    mlflow.sklearn.log_model(
        final_model,
        artifact_path="model",
        registered_model_name="LoanRiskModel"
    )

    print(f"✅ Best model registered with test accuracy: {test_accuracy:.4f}")
