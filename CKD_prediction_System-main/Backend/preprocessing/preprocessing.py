import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_FILE = os.path.join(BASE_DIR, "kidney_disease.csv")


def load_raw_data():
    """Load raw CKD data for EDA."""
    df = pd.read_csv(CSV_FILE)
    df.replace('?', np.nan, inplace=True)
    df.columns = df.columns.str.strip()
    return df


def load_and_preprocess_data(test_size=0.2, random_state=42):
    """Load, clean, encode, split, and scale data for ML training."""
    df = pd.read_csv(CSV_FILE)
    df.replace('?', np.nan, inplace=True)
    df.columns = df.columns.str.strip()

    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)

    target_col = next((c for c in ['class', 'classification', 'CKD', 'status'] if c in df.columns), None)
    if target_col is None:
        raise ValueError("No target column found! Check CSV headers.")

    df[target_col] = df[target_col].astype(str).str.strip()
    df = df[df[target_col].notna()]

    counts = df[target_col].value_counts()
    valid_classes = counts[counts >= 2].index
    df = df[df[target_col].isin(valid_classes)]

    target_encoder = LabelEncoder()
    df[target_col] = target_encoder.fit_transform(df[target_col])

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Convert numeric columns to numeric, coerce errors to NaN
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Drop all-empty numeric columns
    empty_num_cols = X[num_cols].columns[X[num_cols].isna().all()].tolist()
    if empty_num_cols:
        print(f"Dropping completely empty numeric columns: {empty_num_cols}")
        X.drop(columns=empty_num_cols, inplace=True)
        num_cols = [col for col in num_cols if col not in empty_num_cols]

    if num_cols:
        num_imputer = SimpleImputer(strategy='median')
        X[num_cols] = num_imputer.fit_transform(X[num_cols])

    if cat_cols:
        empty_cat_cols = X[cat_cols].columns[X[cat_cols].isna().all()].tolist()
        if empty_cat_cols:
            print(f"Dropping completely empty categorical columns: {empty_cat_cols}")
            X.drop(columns=empty_cat_cols, inplace=True)
            cat_cols = [col for col in cat_cols if col not in empty_cat_cols]

        if cat_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
            for col in cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

    # Drop constant columns
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_cols:
        print(f"Dropping constant columns: {constant_cols}")
        X.drop(columns=constant_cols, inplace=True)

    # Stratified split
    stratify_y = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify_y,
        random_state=random_state
    )

    # Check for NaN or inf before scaling
    if X_train.isna().any().any() or X_test.isna().any().any() or y_train.isna().any() or y_test.isna().any():
        raise ValueError("NaN values found after preprocessing!")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert y to 1D numpy arrays
    y_train = y_train.values
    y_test = y_test.values

    return X_train, X_test, y_train, y_test, scaler, target_encoder, X.columns.tolist()


def get_stratified_kfold(n_splits=5):
    """Return StratifiedKFold object for cross-validation."""
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )
