import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Feature engineering
class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_total_sf=True):
        self.add_total_sf = add_total_sf

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.add_total_sf:
            for col in ['TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea']:
                if col not in X.columns:
                    X[col] = 0
            X['TotalSF'] = X['TotalBsmtSF'].fillna(0) + X['1stFlrSF'].fillna(0) + X['2ndFlrSF'].fillna(0)
        if 'PoolArea' in X.columns:
            X['HasPool'] = (X['PoolArea'].fillna(0) > 0).astype(int)
        return X

# Build preprocessor
def build_preprocessor(df, numeric_impute='median', scale_numeric=True):
    df_columns = df.columns.tolist()
    to_drop = [c for c in ('Id','SalePrice') if c in df_columns]
    feature_cols = [c for c in df_columns if c not in to_drop]

    numeric_cols = df[feature_cols].select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()

    ordinal_mappings = {
        'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'BsmtQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'BsmtCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'FireplaceQu': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
    }
    ordinal_cols = [c for c in ordinal_mappings.keys() if c in categorical_cols]
    cat_nominal = [c for c in categorical_cols if c not in ordinal_cols]

    # Numeric pipeline
    num_steps = [("imputer", SimpleImputer(strategy=numeric_impute))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    # Ordinal pipeline
    if ordinal_cols:
        ordinal_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy='constant', fill_value='None')),
            ("ordinal", OrdinalEncoder(categories=[ordinal_mappings[c] for c in ordinal_cols], dtype=float))
        ])
    else:
        ordinal_transformer = None

    # Nominal categorical pipeline
    if cat_nominal:
        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy='constant', fill_value='None')),
            ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    else:
        cat_transformer = None

    # ColumnTransformer
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if ordinal_cols:
        transformers.append(("ord", ordinal_transformer, ordinal_cols))
    if cat_nominal:
        transformers.append(("cat", cat_transformer, cat_nominal))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Full pipeline
    full_pipeline = Pipeline(steps=[
        ("feat", FeatureAdder(add_total_sf=True)),
        ("pre", preprocessor)
    ])

    return full_pipeline, numeric_cols, ordinal_cols, cat_nominal

# Utility to get feature names
def get_feature_names_from_column_transformer(column_transformer, input_features):
    feature_names = []
    for name, trans, cols in column_transformer.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "named_steps"):
            last = list(trans.named_steps.items())[-1][1]
        else:
            last = trans
        if hasattr(last, "get_feature_names_out"):
            try:
                names = last.get_feature_names_out(cols)
            except:
                names = last.get_feature_names(cols)
            feature_names.extend(list(names))
        else:
            feature_names.extend(list(cols))
    return feature_names
