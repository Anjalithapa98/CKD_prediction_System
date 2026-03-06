import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Backend.preprocessing.preprocessing import load_raw_data


def paginate_plot(df, plot_func, plot_name, output_dir,
                  n_cols=3, figsize_per_plot=(5, 4), max_plots_per_page=6):

    df = df.dropna(axis=1, how='all')
    total_cols = len(df.columns)
    if total_cols == 0:
        print(f"No columns available for {plot_name}. Skipping...")
        return

    pages = math.ceil(total_cols / max_plots_per_page)

    for page in range(pages):
        start = page * max_plots_per_page
        end = min(start + max_plots_per_page, total_cols)

        cols_chunk = df.columns[start:end]
        n_rows = math.ceil(len(cols_chunk) / n_cols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(figsize_per_plot[0] * n_cols,
                     figsize_per_plot[1] * n_rows)
        )

        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for i, col in enumerate(cols_chunk):
            plt.sca(axes[i])
            try:
                plot_func(df[col])
            except:
                axes[i].text(0.5, 0.5, 'Cannot plot', ha='center')
            axes[i].set_title(col, fontsize=10)

        for j in range(len(cols_chunk), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.suptitle(
            f"{plot_name} (Page {page+1})",
            fontsize=16,
            y=1.02
        )

        plt.savefig(
            os.path.join(
                output_dir,
                f"{plot_name.lower().replace(' ','_')}_page_{page+1}.png"
            )
        )
        plt.close()


def save_statistical_analysis(df, output_dir):
    if df.shape[1] == 0:
        print("No numeric columns available for statistical analysis. Skipping...")
        return

    stats = pd.DataFrame()
    stats["Mean"] = df.mean()
    stats["Median"] = df.median()
    stats["Std Dev"] = df.std()
    stats["Variance"] = df.var()
    stats["Skewness"] = df.skew()
    stats["Kurtosis"] = df.kurtosis()
    stats["Min"] = df.min()
    stats["Max"] = df.max()

    stats.to_csv(os.path.join(output_dir, "statistical_analysis.csv"))
    print("Statistical analysis saved.")


def correlation_analysis(df, output_dir):
    if df.shape[1] == 0:
        print("No numeric columns available for correlation heatmap. Skipping...")
        return

    corr = df.corr()
    if corr.empty:
        print("Correlation matrix is empty. Skipping heatmap...")
        return

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm"
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()

    corr.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))
    print("Correlation analysis saved.")


def class_distribution(df, output_dir):
    target_col = next((c for c in ['class', 'classification', 'CKD', 'status'] if c in df.columns), None)
    if target_col:
        plt.figure()
        df[target_col].value_counts().plot(kind="bar")
        plt.title("Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "class_distribution.png"))
        plt.close()
        print("Class distribution saved.")


def perform_eda(output_dir="eda_plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = load_raw_data()
    df.replace('?', np.nan, inplace=True)

    # Numeric columns for plotting
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

    # Drop empty or constant columns
    numeric_df = numeric_df.dropna(axis=1, how='all')
    numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

    print("\n========== EXPLORATORY DATA ANALYSIS ==========\n")

    if numeric_df.shape[1] == 0:
        print("No numeric columns available for plots and analysis. Skipping numeric EDA...")
    else:
        print("Saving Histograms...")
        paginate_plot(
            numeric_df,
            plot_func=lambda x: x.hist(bins=20, edgecolor="black"),
            plot_name="Histograms",
            output_dir=output_dir
        )

        print("Saving Box Plots...")
        paginate_plot(
            numeric_df,
            plot_func=lambda x: plt.boxplot(x.dropna().values, vert=True),
            plot_name="Box Plots",
            output_dir=output_dir
        )

        print("Saving Violin Plots...")
        paginate_plot(
            numeric_df,
            plot_func=lambda x: sns.violinplot(y=x.dropna().values, inner="quartile"),
            plot_name="Violin Plots",
            output_dir=output_dir
        )

        print("Saving Statistical Analysis...")
        save_statistical_analysis(numeric_df, output_dir)

        print("Saving Correlation Analysis...")
        correlation_analysis(numeric_df, output_dir)

    print("Saving Class Distribution...")
    class_distribution(df, output_dir)

    print(f"\nAll EDA outputs saved in '{output_dir}'")


if __name__ == "__main__":
    perform_eda()
