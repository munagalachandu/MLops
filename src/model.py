import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\muni8\Desktop\mlops\Data\loan_risk_data.csv")

# -----------------------------
# 2. Split Features & Target
# -----------------------------
X = df.drop("RiskCategory", axis=1)
y = df["RiskCategory"]

# -----------------------------
# 3. Column Types
# -----------------------------
numeric_features = ["Age", "Income", "CreditScore", "LoanAmount", "LoanTerm"]
categorical_features = ["EmploymentType", "ResidenceType", "PreviousDefault"]

# -----------------------------
# 4. Preprocessing
# -----------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -----------------------------
# 5. Model Pipeline
# -----------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

# -----------------------------
# 6. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 7. Train Model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 8. Evaluate
# -----------------------------
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))
# -----------------------------
# 9. Save Model
# -----------------------------
joblib.dump(model, "loan_risk_model.pkl")
print("Model saved as loan_risk_model.pkl")
