from pathlib import Path

import kagglehub
import pandas as pd


CLIMATE_DATASET_NAME = "berkeleyearth/climate-change-earth-surface-temperature-data"
CLIMATE_FILE_NAME = "GlobalLandTemperaturesByCountry.csv"
OZONE_URL = "https://raw.githubusercontent.com/Sabaae/Dataset/main/ozone_air_pollution_by_station.csv"


def download_climate_dataset():
    path = kagglehub.dataset_download(CLIMATE_DATASET_NAME)
    return Path(path)


def load_climate_raw_data():
    dataset_path = download_climate_dataset()
    file_path = dataset_path / CLIMATE_FILE_NAME

    if not file_path.exists():
        raise FileNotFoundError(f"Climate file not found at: {file_path}")

    df = pd.read_csv(file_path)
    return df


def prepare_climate_data(start_year=1901, end_year=2012):
    df = load_climate_raw_data().copy()

    df["dt"] = pd.to_datetime(df["dt"])
    df["Year"] = df["dt"].dt.year
    df["YearMonth"] = df["dt"].dt.to_period("M")

    df = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)].copy()

    monthly_df = df.pivot(
        index="Country",
        columns="YearMonth",
        values="AverageTemperature"
    )

    monthly_df = monthly_df.dropna()
    monthly_df.columns = monthly_df.columns.to_timestamp()

    yearly_df = monthly_df.T.groupby(monthly_df.columns.year).mean().T
    yearly_df.columns.name = "Year"

    return yearly_df


def load_ozone_raw_data(url=OZONE_URL):
    df = pd.read_csv(url)
    return df


def prepare_ozone_data(start_year=1995, end_year=2022):
    df = load_ozone_raw_data().copy()

    df["Mean"] = pd.to_numeric(df["Mean"], errors="coerce")
    df = df[["Year", "Station Name", "Mean"]].copy()
    df = df.drop_duplicates(subset=["Year", "Station Name"], keep="first")

    ozone_df = df.pivot(
        index="Station Name",
        columns="Year",
        values="Mean"
    )

    ozone_df = ozone_df.loc[:, start_year:end_year]
    ozone_df = ozone_df.dropna()
    ozone_df.columns.name = "Year"

    return ozone_df


def check_dataframe(df, name="DataFrame"):
    print(f"\n{name}")
    print("-" * 40)
    print("Shape:", df.shape)
    print("Missing values:", df.isna().sum().sum())

    if df.shape[0] > 0 and df.shape[1] > 0:
        print(df.head(3))


def save_dataframe(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved: {output_path}")


def build_and_save_datasets(
    climate_output_path="data/processed/climate_yearly.csv",
    ozone_output_path="data/processed/ozone_yearly.csv"
):
    climate_df = prepare_climate_data()
    ozone_df = prepare_ozone_data()

    check_dataframe(climate_df, "Climate yearly data")
    check_dataframe(ozone_df, "Ozone yearly data")

    save_dataframe(climate_df, climate_output_path)
    save_dataframe(ozone_df, ozone_output_path)

    return climate_df, ozone_df


if __name__ == "__main__":
    climate_df, ozone_df = build_and_save_datasets()