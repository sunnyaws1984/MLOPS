import os
import pandas as pd
import gradio as gr
import mlflow
import mlflow.sklearn

RUN_ID = os.getenv("RUN_ID")
if not RUN_ID:
    raise ValueError("RUN_ID environment variable is required")

MODEL_URI = f"runs:/{RUN_ID}/house_price_model"
model = mlflow.sklearn.load_model(MODEL_URI)

def predict_price(gr_liv_area, total_bsmt_sf, overall_qual, year_built):
    df = pd.DataFrame([{
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt_sf,
        "OverallQual": overall_qual,
        "YearBuilt": year_built
    }])
    return round(float(model.predict(df)[0]), 2)

gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Living Area (sq ft)"),
        gr.Number(label="Basement Area (sq ft)"),
        gr.Slider(1, 10, step=1, label="Overall Quality"),
        gr.Number(label="Year Built")
    ],
    outputs="number",
    title="House Price Prediction",
    description="Model loaded from MLflow"
).launch(server_name="0.0.0.0", server_port=7860)
