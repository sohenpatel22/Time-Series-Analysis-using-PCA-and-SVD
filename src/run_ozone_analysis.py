import os
import numpy as np

from load_data import prepare_ozone_data
from preprocess import standardize_dataframe, inverse_transform_dataframe
from pca_utils import (
    run_pca,
    get_num_components_for_threshold,
    reconstruct_dataframe
)
from svd_utils import reconstruct_svd_dataframe
from plotting import (
    plot_time_series,
    plot_scree,
    plot_first_n_pcs,
    plot_incremental_reconstruction,
    plot_residuals,
    plot_rmse_curves,
    plot_comparison_residuals,
    plot_comparison_rmse
)


def get_pca_rmse_on_original_scale(df_original, df_standardized, scaler, eigenvectors, selected_rows, max_components=10):
    rmse_dict = {}
    scores = df_standardized.values @ eigenvectors

    for name in selected_rows:
        if name not in df_standardized.index:
            print(f"Skipping {name} because it is not in the dataframe.")
            continue

        original_row = df_original.loc[name].values.reshape(1, -1)
        row_index = df_standardized.index.get_loc(name)
        rmse_list = []

        for k in range(1, max_components + 1):
            reconstructed_std = scores[row_index, :k].reshape(1, -1) @ eigenvectors[:, :k].T
            reconstructed_original = scaler.inverse_transform(reconstructed_std)
            rmse = np.sqrt(np.mean((original_row - reconstructed_original) ** 2))
            rmse_list.append(rmse)

        rmse_dict[name] = rmse_list

    return rmse_dict


def get_svd_std_rmse_on_original_scale(df_original, df_standardized, scaler, selected_rows, max_components=10):
    rmse_dict = {}
    U, S, Vt = np.linalg.svd(df_standardized.values, full_matrices=False)

    for name in selected_rows:
        if name not in df_standardized.index:
            print(f"Skipping {name} because it is not in the dataframe.")
            continue

        row_index = df_standardized.index.get_loc(name)
        original_row = df_original.loc[name].values.reshape(1, -1)
        rmse_list = []

        for k in range(1, max_components + 1):
            U_k = U[:, :k]
            S_k = np.diag(S[:k])
            Vt_k = Vt[:k, :]
            reconstructed_std = U_k @ S_k @ Vt_k
            reconstructed_row_std = reconstructed_std[row_index].reshape(1, -1)
            reconstructed_row_original = scaler.inverse_transform(reconstructed_row_std)

            rmse = np.sqrt(np.mean((original_row - reconstructed_row_original) ** 2))
            rmse_list.append(rmse)

        rmse_dict[name] = rmse_list

    return rmse_dict


def main():
    os.makedirs("outputs/plots", exist_ok=True)

    ozone_df = prepare_ozone_data()
    ozone_std_df, scaler = standardize_dataframe(ozone_df)

    selected_stations = [
        "Dorset",
        "Kingston",
        "Sarnia",
        "Grand Bend",
        "North Bay",
        "Burlington"
    ]

    print("\nOzone data shape:", ozone_df.shape)
    print("Selected stations:", selected_stations)

    plot_time_series(
        ozone_df,
        selected_stations,
        title="Ozone Data: Original Time Series",
        y_label="Ozone Level",
        save_path="outputs/plots/ozone_time_series.png"
    )

    pca_results = run_pca(ozone_std_df)

    eigenvalues = pca_results["eigenvalues"]
    eigenvectors = pca_results["eigenvectors"]
    explained_variance_ratio = pca_results["explained_variance_ratio"]
    cumulative_variance = pca_results["cumulative_variance"]

    n_components_999 = get_num_components_for_threshold(cumulative_variance, threshold=0.999)

    print("\nPCA results")
    print("-" * 40)
    print("Covariance matrix shape:", pca_results["cov_matrix"].shape)
    print("Number of components for 99.9% variance:", n_components_999)

    plot_scree(
        eigenvalues,
        explained_variance_ratio,
        cumulative_variance,
        title="Ozone Data: Scree Plot",
        save_path="outputs/plots/ozone_scree.png"
    )

    plot_first_n_pcs(
        ozone_std_df.columns,
        eigenvectors,
        n_components=16,
        title="Ozone Data: First 16 Principal Components",
        save_path="outputs/plots/ozone_first_16_pcs.png"
    )

    pca_reconstructed_std = {
        1: reconstruct_dataframe(ozone_std_df, eigenvectors, 1),
        2: reconstruct_dataframe(ozone_std_df, eigenvectors, 2),
        4: reconstruct_dataframe(ozone_std_df, eigenvectors, 4),
        8: reconstruct_dataframe(ozone_std_df, eigenvectors, 8),
        16: reconstruct_dataframe(ozone_std_df, eigenvectors, 16),
    }

    pca_reconstructed_original = {}

    for k, df_std in pca_reconstructed_std.items():
        pca_reconstructed_original[k] = inverse_transform_dataframe(df_std, scaler)

    plot_incremental_reconstruction(
        ozone_df,
        pca_reconstructed_original,
        selected_stations,
        title="Ozone Data: Incremental PCA Reconstruction",
        y_label="Ozone Level",
        save_path="outputs/plots/ozone_incremental_reconstruction.png"
    )

    plot_residuals(
        ozone_df,
        pca_reconstructed_original[16],
        selected_stations,
        title="Ozone Data: Residuals for PCA Reconstruction (16 Components)",
        y_label="Residual",
        save_path="outputs/plots/ozone_pca_residuals_16.png"
    )

    pca_rmse = get_pca_rmse_on_original_scale(
        ozone_df,
        ozone_std_df,
        scaler,
        eigenvectors,
        selected_stations,
        max_components=10
    )

    plot_rmse_curves(
        pca_rmse,
        title="Ozone Data: PCA RMSE vs Number of Components",
        y_label="RMSE",
        save_path="outputs/plots/ozone_pca_rmse.png"
    )

    svd_std_reconstructed = reconstruct_svd_dataframe(ozone_std_df, 16)
    svd_std_reconstructed = inverse_transform_dataframe(svd_std_reconstructed, scaler)

    svd_original_reconstructed = reconstruct_svd_dataframe(ozone_df, 16)

    comparison_reconstructed = {
        "PCA": pca_reconstructed_original[16],
        "SVD Standardized": svd_std_reconstructed,
        "SVD Original": svd_original_reconstructed,
    }

    plot_comparison_residuals(
        ozone_df,
        comparison_reconstructed,
        selected_stations,
        title="Ozone Data: PCA vs SVD Residual Comparison",
        y_label="Residual",
        save_path="outputs/plots/ozone_pca_vs_svd_residuals.png"
    )

    svd_std_rmse = get_svd_std_rmse_on_original_scale(
        ozone_df,
        ozone_std_df,
        scaler,
        selected_stations,
        max_components=10
    )

    svd_original_rmse = {}
    for name in selected_stations:
        if name not in ozone_df.index:
            continue

        original_row = ozone_df.loc[name].values.reshape(1, -1)
        rmse_list = []

        for k in range(1, 11):
            reconstructed_df = reconstruct_svd_dataframe(ozone_df, k)
            reconstructed_row = reconstructed_df.loc[name].values.reshape(1, -1)
            rmse = np.sqrt(np.mean((original_row - reconstructed_row) ** 2))
            rmse_list.append(rmse)

        svd_original_rmse[name] = rmse_list

    comparison_rmse = {
        "PCA": pca_rmse,
        "SVD Standardized": svd_std_rmse,
        "SVD Original": svd_original_rmse,
    }

    plot_comparison_rmse(
        comparison_rmse,
        title="Ozone Data: PCA vs SVD RMSE Comparison",
        y_label="RMSE",
        save_path="outputs/plots/ozone_pca_vs_svd_rmse.png"
    )


if __name__ == "__main__":
    main()