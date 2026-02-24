import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and string values.
    """
    df = df.copy()

    # normalize columns
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    # normalize string values
    for col in df.select_dtypes(include="object"):
        df[col] = (
            df[col]
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", "_", regex=True)
        )

    return df


def median_impute(
    df: pd.DataFrame,
    columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Impute missing values using the median.
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()

    for col in columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    return df


def knn_impute(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    n_neighbors: int = 5
) -> pd.DataFrame:
    """
    Impute missing values using KNN.
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns])

    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights="uniform"
    )
    filled_scaled = imputer.fit_transform(scaled_data)

    filled_original = scaler.inverse_transform(filled_scaled)

    df[columns] = filled_original

    return df


def count_outliers(column: pd.Series) -> int:
    """
    Count the number of outliers in a column using the IQR method.
    """
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return (column < lower_bound).sum() + (column > upper_bound).sum()

def build_preprocessor_median(
    continuous_vars: list[str],
    categorical_vars: list[str],
    binary_vars: list[str]
) -> ColumnTransformer:
    """
    Build a preprocessor that imputes missing values using the median.
    """
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), continuous_vars),
            ("cat", OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False
            ), categorical_vars),
            ("bin", "passthrough", binary_vars)
        ],
        verbose_feature_names_out=False,
    )


def build_preprocessor_knn(
    continuous_vars: list[str],
    categorical_vars: list[str],
    binary_vars: list[str]
) -> ColumnTransformer:
    """
    Build a preprocessor that imputes missing values using KNN.
    """
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", KNNImputer(n_neighbors=5)),
                ("scaler", StandardScaler()),
            ]), continuous_vars),
            ("cat", OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False
            ), categorical_vars),
            ("bin", "passthrough", binary_vars)
        ],
        verbose_feature_names_out=False,
    )
