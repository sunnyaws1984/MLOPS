## What sklearn is used for

Data preprocessing (scaling, encoding, cleaning)
Machine learning models
Regression (Linear, Random Forest, etc.)
Classification (Logistic, SVM, KNN)
Model evaluation
Accuracy and other Metrics
Train-test splitting

## Important Notes

mlflow.sklearn.load_model(
    "runs:/35e680a0f5034befb2c84838ca3ce4b5/house_price_model"
)

MLflow does:
Read run_id → locate the run
Read artifact root from backend store
Load model.pkl (model)

mlflow.sklearn.load_model("runs:/RUN_ID/house_price_model")
Why this is powerful:
Same code works locally, cloud, Docker, Kubernetes
Artifact store can change (local → S3)
Backend DB can change (SQLite → Postgres)
Your code never change

## Dockerize the Application
```bash
python train_and_log_model.py 
export RUN_ID=<GET_RUN_ID_FROM_ABOVE>
python app.py







