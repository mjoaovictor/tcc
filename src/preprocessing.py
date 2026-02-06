import pandas as pd


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
