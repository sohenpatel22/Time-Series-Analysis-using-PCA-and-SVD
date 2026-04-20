import numpy as np
import pandas as pd


def compute_covariance_matrix(df):
    return df.cov()


def get_sorted_eigen(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvalues, eigenvectors


def compute_explained_variance(eigenvalues):
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    return explained_variance_ratio, cumulative_variance


def get_num_components_for_threshold(cumulative_variance, threshold=0.999):
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    return n_components


def project_onto_pcs(df, eigenvectors):
    scores = df.values @ eigenvectors
    return scores


def reconstruct_from_pcs(scores, eigenvectors, k):
    reconstructed = scores[:, :k] @ eigenvectors[:, :k].T
    return reconstructed


def reconstruct_dataframe(df, eigenvectors, k):
    scores = project_onto_pcs(df, eigenvectors)
    reconstructed_array = reconstruct_from_pcs(scores, eigenvectors, k)

    reconstructed_df = pd.DataFrame(
        reconstructed_array,
        index=df.index,
        columns=df.columns
    )

    return reconstructed_df


def compute_rmse(original, reconstructed):
    original_array = np.array(original)
    reconstructed_array = np.array(reconstructed)

    rmse = np.sqrt(np.mean((original_array - reconstructed_array) ** 2))
    return rmse


def compute_rmse_by_row(original_df, reconstructed_df):
    rmse_values = {}

    for row_name in original_df.index:
        original_row = original_df.loc[row_name].values
        reconstructed_row = reconstructed_df.loc[row_name].values
        rmse_values[row_name] = compute_rmse(original_row, reconstructed_row)

    return pd.Series(rmse_values, name="RMSE")


def compute_rmse_curve(df, eigenvectors, max_components=10):
    rmse_list = []

    for k in range(1, max_components + 1):
        reconstructed_df = reconstruct_dataframe(df, eigenvectors, k)
        rmse = compute_rmse(df, reconstructed_df)
        rmse_list.append(rmse)

    return rmse_list


def compute_rmse_curve_for_selected_rows(df, eigenvectors, selected_rows, max_components=10):
    rmse_dict = {}
    scores = project_onto_pcs(df, eigenvectors)

    for row_name in selected_rows:
        if row_name not in df.index:
            print(f"Skipping {row_name} because it is not in the dataframe.")
            continue

        rmse_list = []
        original_row = df.loc[row_name].values.reshape(1, -1)
        row_index = df.index.get_loc(row_name)

        for k in range(1, max_components + 1):
            reconstructed_row = scores[row_index, :k].reshape(1, -1) @ eigenvectors[:, :k].T
            rmse = compute_rmse(original_row, reconstructed_row)
            rmse_list.append(rmse)

        rmse_dict[row_name] = rmse_list

    return rmse_dict


def run_pca(df):
    cov_matrix = compute_covariance_matrix(df)
    eigenvalues, eigenvectors = get_sorted_eigen(cov_matrix)
    explained_variance_ratio, cumulative_variance = compute_explained_variance(eigenvalues)
    scores = project_onto_pcs(df, eigenvectors)

    results = {
        "cov_matrix": cov_matrix,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance": cumulative_variance,
        "scores": scores,
    }

    return results