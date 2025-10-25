import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def evaluate(model_path="../models/final_model.joblib"):
    model = joblib.load(model_path)
    df = pd.read_csv("../data/train.csv")
    y = np.log1p(df['SalePrice'])
    X = df.drop(['SalePrice','Id'], axis=1, errors='ignore')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Predicting on holdout set...")
    pred_log = model.predict(X_val)
    rmse_log = np.sqrt(mean_squared_error(y_val, pred_log))
    print("RMSE (log-target):", rmse_log)
    # back to original
    y_true = np.expm1(y_val)
    y_pred = np.expm1(pred_log)
    print("RMSE (original):", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("MAE (original):", mean_absolute_error(y_true, y_pred))
    print("R2 (log-target):", r2_score(y_val, pred_log))

if __name__ == "__main__":
    evaluate()
