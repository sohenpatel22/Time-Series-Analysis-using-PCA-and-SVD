import numpy as np
import pandas as pd


def compute_svd(df):
    U, S, Vt = np.linalg.svd(df.values, full_matrices=False)
    return U, S, Vt


def get_svd_scores(U, S):
    scores = U * S
    return scores


def reconstruct_from_svd(U, S, Vt, k):
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]

    reconstructed = U_k @ S_k @ Vt_k
    return reconstructed


def reconstruct_svd_dataframe(df, k):
    U, S, Vt = compute_svd(df)
    reconstructed_array = reconstruct_from_svd(U, S, Vt, k)

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


def compute_rmse_curve_svd(df, max_components=10):
    U, S, Vt = compute_svd(df)
    rmse_list = []

    for k in range(1, max_components + 1):
        reconstructed_array = reconstruct_from_svd(U, S, Vt, k)
        rmse = compute_rmse(df.values, reconstructed_array)
        rmse_list.append(rmse)

    return rmse_list


def compute_rmse_curve_svd_for_selected_rows(df, selected_rows, max_components=10):
    U, S, Vt = compute_svd(df)
    rmse_dict = {}

    for row_name in selected_rows:
        if row_name not in df.index:
            print(f"Skipping {row_name} because it is not in the dataframe.")
            continue

        row_index = df.index.get_loc(row_name)
        original_row = df.loc[row_name].values.reshape(1, -1)
        rmse_list = []

        for k in range(1, max_components + 1):
            reconstructed_array = reconstruct_from_svd(U, S, Vt, k)
            reconstructed_row = reconstructed_array[row_index].reshape(1, -1)
            rmse = compute_rmse(original_row, reconstructed_row)
            rmse_list.append(rmse)

        rmse_dict[row_name] = rmse_list

    return rmse_dict


def run_svd(df):
    U, S, Vt = compute_svd(df)
    scores = get_svd_scores(U, S)

    results = {
        "U": U,
        "S": S,
        "Vt": Vt,
        "scores": scores,
    }

    return results