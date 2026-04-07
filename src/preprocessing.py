from typing import Any, Literal

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)


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


def build_pipeline(
    model: BaseEstimator,
    continuous_vars: list[str],
    categorical_vars: list[str],
    binary_vars: list[str],
    imputation_method: Literal["median", "knn"] = "median",
    apply_log: bool = True,
    log_variables: list[str] | None = None,
    sampler: Any | None = None,
) -> ImbPipeline:
    if imputation_method == "median":
        num_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    elif imputation_method == "knn":
        num_steps = [
            ("scaler", StandardScaler()),
            ("imputer", KNNImputer(n_neighbors=5, weights="uniform")),
        ]
    else:
        raise ValueError(f"invalid imputation method: {imputation_method}")

    if apply_log:
        if log_variables is None:
            log_variables = continuous_vars.copy()

        log_num_pipeline = Pipeline(steps=[
            ("log1p", FunctionTransformer(np.log1p, validate=False)),
            *num_steps
        ])

        std_num_pipeline = Pipeline(steps=num_steps)

        std_variables = [var for var in continuous_vars if var not in log_variables]

        num_transformers = [
            ("log_num", log_num_pipeline, log_variables),
            ("std_num", std_num_pipeline, std_variables),
        ]
    else:
        num_transformers = [
            ("std_num", Pipeline(num_steps), continuous_vars),
        ]

    preprocessor = ColumnTransformer(
        transformers=[
            *num_transformers,
            ("cat", OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False
            ), categorical_vars),
            ("bin", "passthrough", binary_vars)
        ],
        verbose_feature_names_out=False,
    )

    if sampler is not None:
        return ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("sampler", sampler),
            ("model", model)
        ])

    return ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
