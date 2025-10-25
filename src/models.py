from sklearn.linear_model import LassoCV, Ridge
from sklearn.ensemble import RandomForestRegressor

def get_models():

    models = {
        "LassoCV": LassoCV(cv=5, random_state=42, n_alphas=50, max_iter=5000),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    }
    return models
