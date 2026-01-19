import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']
DROP_COLS = ['label', 'difficulty', 'is_attack']

def build_preprocessor(categorical_cols, numeric_cols):
    cat = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    num = StandardScaler()
    return ColumnTransformer(
        [('cat', cat, categorical_cols), ('num', num, numeric_cols)],
        remainder='drop'
    )

def preprocess_data(df_train, df_test):
    df_train_normal = df_train[df_train['label'] == 'normal']
    numeric_cols = [c for c in df_train.columns if c not in CATEGORICAL_COLS + DROP_COLS]
    preprocessor = build_preprocessor(CATEGORICAL_COLS, numeric_cols)
    preprocessor.fit(df_train_normal[CATEGORICAL_COLS + numeric_cols])
    X_train = preprocessor.transform(df_train[CATEGORICAL_COLS + numeric_cols])
    X_test = preprocessor.transform(df_test[CATEGORICAL_COLS + numeric_cols])
    return X_train, X_test, preprocessor