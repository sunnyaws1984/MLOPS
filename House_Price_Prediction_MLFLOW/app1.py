import os
import pandas as pd
import gradio as gr
import mlflow
import mlflow.sklearn

# Point to the mlruns folder
mlflow.set_tracking_uri("file:///app/mlruns") # Adjust path as needed

model_uri = "mlruns/0/models/m-9a432fc99a4f4574a04457b3e270c4c3/artifacts"     # Hardcoded model path
model = mlflow.sklearn.load_model(model_uri) # Load model from the specified path

print(f"Model loaded from: {model_uri}")

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
