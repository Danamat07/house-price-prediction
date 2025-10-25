import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from data_loader import load_data
from preprocess import build_preprocessor
from models import get_models

# Load data
train_df, test_df = load_data()

# Define features and target
y = np.log1p(train_df["SalePrice"])
X = train_df.drop(["SalePrice", "Id"], axis=1)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessor
preprocessor, numeric_cols, ordinal_cols, cat_cols = build_preprocessor(train_df)
joblib.dump(preprocessor, "../models/preprocessor.joblib")
print("✅ Saved preprocessor to ../models/preprocessor.joblib")

# Load models
models = get_models()
results = []

# Train and evaluate
for name, model in models.items():
    print(f"\n=== Training {name} ===")
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)

    # Limit extreme predictions to avoid overflow
    preds_safe = np.clip(preds, a_min=None, a_max=20)
    y_pred_orig = np.expm1(preds_safe)
    y_true_orig = np.expm1(y_val)

    # Metrics
    rmse_log = np.sqrt(mean_squared_error(y_val, preds))
    mae_log = mean_absolute_error(y_val, preds)
    r2_log = r2_score(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae = mean_absolute_error(y_true_orig, y_pred_orig)

    print(f"{name} log-RMSE: {rmse_log:.4f}")
    print(f"{name} log-MAE: {mae_log:.4f}")
    print(f"{name} log-R²: {r2_log:.4f}")
    print(f"{name} RMSE (original scale): {rmse:,.0f}")
    print(f"{name} MAE (original scale): {mae:,.0f}")

    results.append((name, rmse_log, pipe))

# Pick best model
best_model = sorted(results, key=lambda x: x[1])[0]
print(f"\n🏆 Best model: {best_model[0]} (log-RMSE={best_model[1]:.4f})")
joblib.dump(best_model[2], "../models/final_model.joblib")
print("✅ Saved best model to ../models/final_model.joblib")
