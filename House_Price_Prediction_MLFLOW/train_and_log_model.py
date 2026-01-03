import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import mlflow
import mlflow.sklearn

# Load dataset
df = pd.read_csv("dataset/train.csv")

X = df[["GrLivArea", "TotalBsmtSF", "OverallQual", "YearBuilt"]]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow run
with mlflow.start_run() as run:
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    # Log params & metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("r2_score", r2)

    # Log model
    mlflow.sklearn.log_model(
        model,
        artifact_path="house_price_model"
    )

    print("Model logged to MLflow")
    print("Run ID:", run.info.run_id)