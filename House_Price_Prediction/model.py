import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset (download train.csv from Kaggle)
df = pd.read_csv("dataset/train.csv")

# 2. Select numeric features only (simple, clean)
X = df[["GrLivArea", "TotalBsmtSF", "OverallQual", "YearBuilt"]]   # all numbers
y = df["SalePrice"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create & train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# 7. Predict new house price
new_house = pd.DataFrame([{
    "GrLivArea": 2000,
    "TotalBsmtSF": 900,
    "OverallQual": 7,
    "YearBuilt": 2005
}])

price = model.predict(new_house)
print("Predicted Price:", price[0])
