import matplotlib.pyplot as plt
import numpy as np


def plot_time_series(df, selected_names, title, y_label, save_path=None, show=True):
    plt.figure(figsize=(12, 5))

    for name in selected_names:
        if name in df.index:
            plt.plot(df.columns, df.loc[name], linewidth=2, label=name)

    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_scree(eigenvalues, explained_variance_ratio, cumulative_variance, title, save_path=None, show=True):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    x = np.arange(1, len(eigenvalues) + 1)

    ax1.bar(x, explained_variance_ratio * 100, alpha=0.6)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative_variance * 100, marker="o", linewidth=2, markersize=3)
    ax2.set_ylabel("Cumulative Variance (%)")

    for threshold in [90, 95, 99]:
        k = np.argmax(cumulative_variance >= threshold / 100) + 1
        ax2.axhline(y=threshold, linestyle="--", alpha=0.5)
        ax2.text(k, threshold, f"{threshold}% at PC{k}", fontsize=9)

    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_first_n_pcs(years, eigenvectors, n_components=16, title="Principal Components", save_path=None, show=True):
    fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True)
    axes = axes.flatten()

    for i in range(n_components):
        axes[i].plot(years, eigenvectors[:, i], linewidth=1.5)
        axes[i].axhline(0, linestyle="--", linewidth=0.8, alpha=0.5)
        axes[i].set_title(f"PC {i+1}")
        axes[i].grid(True, alpha=0.3)

    for i in range(n_components, len(axes)):
        axes[i].axis("off")

    for ax in axes[-4:]:
        ax.set_xlabel("Year")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_incremental_reconstruction(original_df, reconstructed_dfs, selected_names, title, y_label, save_path=None, show=True):
    n = len(selected_names)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True)

    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, selected_names):
        if name not in original_df.index:
            continue

        ax.plot(
            original_df.columns,
            original_df.loc[name],
            linestyle="--",
            linewidth=2.5,
            label="Original"
        )

        for k, recon_df in reconstructed_dfs.items():
            if name in recon_df.index:
                label = "PC1" if k == 1 else f"PC1-PC{k}"
                ax.plot(
                    recon_df.columns,
                    recon_df.loc[name],
                    linewidth=1.5,
                    label=label
                )

        ax.set_title(name)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Year")
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_residuals(original_df, reconstructed_df, selected_names, title, y_label, save_path=None, show=True):
    n = len(selected_names)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True)

    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, selected_names):
        if name not in original_df.index or name not in reconstructed_df.index:
            continue

        residual = original_df.loc[name] - reconstructed_df.loc[name]
        rmse = np.sqrt(np.mean(residual.values ** 2))

        ax.plot(original_df.columns, residual, linewidth=1.5, label=f"RMSE={rmse:.4f}")
        ax.fill_between(original_df.columns, residual, alpha=0.2)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Year")
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_rmse_curves(rmse_dict, title, y_label, save_path=None, show=True):
    plt.figure(figsize=(10, 5))

    for name, values in rmse_dict.items():
        x = range(1, len(values) + 1)
        plt.plot(x, values, marker="o", linewidth=1.8, label=name)

    plt.title(title)
    plt.xlabel("Number of Components")
    plt.ylabel(y_label)
    plt.xticks(range(1, len(next(iter(rmse_dict.values()))) + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison_residuals(original_df, reconstructed_dict, selected_names, title, y_label, save_path=None, show=True):
    n = len(selected_names)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True)

    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, selected_names):
        if name not in original_df.index:
            continue

        for method_name, recon_df in reconstructed_dict.items():
            if name not in recon_df.index:
                continue

            residual = original_df.loc[name] - recon_df.loc[name]
            rmse = np.sqrt(np.mean(residual.values ** 2))

            ax.plot(
                original_df.columns,
                residual,
                linewidth=1.5,
                label=f"{method_name} ({rmse:.4f})"
            )

        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Year")
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison_rmse(rmse_results, title, y_label, save_path=None, show=True):
    names = list(next(iter(rmse_results.values())).keys())
    n = len(names)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        for method_name, method_data in rmse_results.items():
            if name not in method_data:
                continue

            values = method_data[name]
            x = range(1, len(values) + 1)

            ax.plot(x, values, marker="o", linewidth=1.5, label=method_name)

        ax.set_title(name)
        ax.set_xlabel("Components")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()