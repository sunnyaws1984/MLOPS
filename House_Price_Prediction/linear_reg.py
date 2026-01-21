import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import gradio as gr

# 1. Load dataset (download train.csv from Kaggle)
df = pd.read_csv("dataset/train.csv")

# 2. Select numeric features only (simple, clean)
X = df[["GrLivArea", "TotalBsmtSF", "OverallQual", "YearBuilt"]]   # all numbers
y = df["SalePrice"]

# 3. Train-test split , random_state=42 fixes the randomness in Algorithm so you get the same results every time you run the code
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create & train model
model = LinearRegression() 
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

## 6. Evaluation
# R² Score - Think of R² as a marks score for your model
# Example
# R² = 1.0 → model is perfect (100/100) , R² = 0.8 → model is good (80/100) , R² = 0.5 → model is average (50/100) ,R² = 0.0 → model is bad (0/100)

print("Model Performance:", round(r2_score(y_test, y_pred) * 100, 2), "%")

# 7. Predict new house price

def predict_price(gr_liv_area, total_bsmt_sf, overall_qual, year_built):
    new_house = pd.DataFrame([{
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt_sf,
        "OverallQual": overall_qual,
        "YearBuilt": year_built
    }])

    price = model.predict(new_house)
    return round(price[0], 2) 

# 8. Gradio Interface

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
