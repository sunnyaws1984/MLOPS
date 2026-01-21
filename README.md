# MLOPS
Sample House Predictions using Machine Learning

Below are the steps to perform:

1) Setup a Virtual env.
2) pip install pandas numpy scikit-learn gradio mlflow
3) python model.py

You’ll get a web UI like:

Living Area: 2000
Basement: 900
Quality: 7
Year Built: 2005
→ Predicted Price: 236,889

################################################################################################################################

📏 What is Linear Regression?

Linear Regression is a machine learning algorithm that predicts a numeric value by learning a straight-line relationship between input features and the target.

📌 In simple words

It uses a single formula to predict outputs:

Price = (w1 × Feature1) + (w2 × Feature2) + … + b
The model learns weights (w1, w2, …) and a base (b) from the training data
eg:
Price =
  (a × GrLivArea)
+ (b × TotalBsmtSF)
+ (c × OverallQual)
+ (d × YearBuilt)
+ base price

Each feature contributes proportionally to the prediction
Predictions are a linear combination of features
Works best when the relationship between inputs and output is roughly straight-line


🌳 What is Random Forest?

Random Forest is a machine learning algorithm that makes predictions by building many decision trees and combining their results.

📌 In simple words

It does not use a single rule or formula
It creates many decision trees, each trained slightly differently
Every tree gives its own prediction
The final answer is the average of all trees (for regression)
Each tree learns different patterns from the data

Eg:
Build Tree 1 → makes a prediction
Build Tree 2 → makes a prediction
… up to Tree 100

Final prediction = average of all 100 tree predictions

🔹 Key Differences
 	                Linear Regression	                                            Random Forest
Formula    	      Uses a single straight-line formula (weights + base)	      Uses many decision trees and averages their predictions
Relationship	  Assumes a linear relationship between features and target	  Can capture complex, non-linear relationships
Interpretability  Very easy to explain (weights show feature importance)	  Harder to explain (rules inside many trees)
Accuracy	      Works well for simple, linear problems	                  Usually more accurate for complex data
Training	      Fast	                                                      Slower (many trees to train)
Prediction Style  Direct calculation using formula	                          Voting / averaging across trees
