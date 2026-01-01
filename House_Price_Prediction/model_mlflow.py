import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import gradio as gr
import mlflow
import mlflow.sklearn

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("dataset/train.csv")

# Select numeric features only
X = df[["GrLivArea", "TotalBsmtSF", "OverallQual", "YearBuilt"]]
y = df["SalePrice"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 2. MLflow tracking
# -------------------------
with mlflow.start_run():

    # Create & train Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,   # number of trees
        random_state=42
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    r2 = r2_score(y_test, y_pred)
    print("Model Performance (R²):", round(r2 * 100, 2), "%")

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("r2_score", r2)

    # Log the model
    mlflow.sklearn.log_model(model, "houese_price_model")

# -------------------------
# 3. Load model back from MLflow (optional, safe)
# -------------------------
# Get the latest run id from mlflow
# loaded_model = mlflow.sklearn.load_model(f"runs:/{mlflow.active_run().info.run_id}/random_forest_model")

# -------------------------
# 4. Gradio interface for predictions
# -------------------------
def predict_price(gr_liv_area, total_bsmt_sf, overall_qual, year_built):
    new_house = pd.DataFrame([{
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt_sf,
        "OverallQual": overall_qual,
        "YearBuilt": year_built
    }])
    price = model.predict(new_house)
    return round(price[0], 2)

interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Living Area (sq ft)"),
        gr.Number(label="Basement Area (sq ft)"),
        gr.Slider(1, 10, step=1, label="Overall Quality"),
        gr.Number(label="Year Built")
    ],
    outputs=gr.Number(label="Predicted House Price (USD)"),
    title="House Price Prediction",
    description="Enter house details to predict the price"
)

interface.launch()
