import pandas as pd
from sklearn.preprocessing import StandardScaler


def validate_dataframe(df, name="dataframe"):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    if df.empty:
        raise ValueError(f"{name} is empty.")

    if df.isna().sum().sum() > 0:
        raise ValueError(f"{name} still has missing values.")

    print(f"{name} is valid.")
    print("Shape:", df.shape)


def standardize_dataframe(df):
    validate_dataframe(df, "input dataframe")

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)

    standardized_df = pd.DataFrame(
        scaled_array,
        index=df.index,
        columns=df.columns
    )

    return standardized_df, scaler


def inverse_transform_dataframe(df_scaled, scaler):
    original_array = scaler.inverse_transform(df_scaled)

    original_df = pd.DataFrame(
        original_array,
        index=df_scaled.index,
        columns=df_scaled.columns
    )

    return original_df


def select_rows(df, row_names):
    missing_rows = [row for row in row_names if row not in df.index]

    if missing_rows:
        print("These rows were not found:")
        for row in missing_rows:
            print(f"- {row}")

    selected_rows = [row for row in row_names if row in df.index]

    if len(selected_rows) == 0:
        raise ValueError("None of the requested rows were found.")

    selected_df = df.loc[selected_rows].copy()
    return selected_df


def print_summary_stats(df, name="dataframe"):
    print(f"\nSummary for {name}")
    print("-" * 40)
    print("Mean (first 5 columns):")
    print(df.mean().head())
    print("\nStd (first 5 columns):")
    print(df.std().head())
    print("\nMin value:", df.min().min())
    print("Max value:", df.max().max())