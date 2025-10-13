import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import matplotlib.font_manager as fm

def setup_cjk_font():
    """
    Intelligently tries to set a font that supports CJK characters and provides clear guidance.
    This function is kept in case the user wants to try enabling CJK support in the future.
    """
    font_candidates = [
        'SimHei', 'Microsoft YaHei',  # Windows
        'PingFang SC', 'STHeiti',     # macOS
        'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Source Han Sans SC'  # Linux
    ]
    
    print("Attempting to set a CJK-compatible font (optional)...")
    for font_name in font_candidates:
        try:
            fm.findfont(font_name, fallback_to_default=False)
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"Success! Font has been set to '{font_name}'.")
            return
        except:
            continue
    
    print("\nWarning: Could not find any recommended CJK fonts on your system.")
    print("Plots will be generated using the default English font.")


def analyze_and_plot_ivf_stats(csv_filepath):
    """
    Analyzes IVF partition statistics from a CSV file and generates insightful plots.

    Args:
        csv_filepath (str): The path to the _ivf_stats.csv file.
    """
    if not os.path.exists(csv_filepath):
        print(f"Error: File '{csv_filepath}' not found.")
        return

    # --- 1. Read and Calculate Statistics ---
    df = pd.read_csv(csv_filepath)
    
    if not {'partition_id', 'vector_count'}.issubset(df.columns):
        print(f"Error: CSV file '{csv_filepath}' must contain 'partition_id' and 'vector_count' columns.")
        return

    vector_counts = df['vector_count']
    
    # Core statistics (based on partition count)
    nlist = len(df)
    non_empty_partitions = len(vector_counts.dropna())
    total_vectors = vector_counts.sum()
    max_size = vector_counts.max()
    min_size = vector_counts.min()
    avg_size = vector_counts.mean()
    std_dev = vector_counts.std()
    median_size = vector_counts.median()
    q1 = vector_counts.quantile(0.25)
    q3 = vector_counts.quantile(0.75)

    # --- New Metrics: Based on Vector Distribution ---
    # Threshold for partitions holding the top 25% and 75% of vectors
    sorted_desc = df.sort_values(by='vector_count', ascending=False)
    sorted_desc['cumulative_vectors'] = sorted_desc['vector_count'].cumsum()
    
    target_top_25_vec = total_vectors * 0.25
    target_top_75_vec = total_vectors * 0.75
    
    top_25_idx = sorted_desc['cumulative_vectors'].searchsorted(target_top_25_vec)
    top_75_idx = sorted_desc['cumulative_vectors'].searchsorted(target_top_75_vec)
    
    top_25_threshold = sorted_desc.iloc[top_25_idx]['vector_count'] if top_25_idx < len(sorted_desc) else np.nan
    top_75_threshold = sorted_desc.iloc[top_75_idx]['vector_count'] if top_75_idx < len(sorted_desc) else np.nan

    # Threshold for partitions holding the bottom 25% of vectors
    sorted_asc = df.sort_values(by='vector_count', ascending=True)
    sorted_asc['cumulative_vectors'] = sorted_asc['vector_count'].cumsum()
    
    target_bottom_25_vec = total_vectors * 0.25
    bottom_25_idx = sorted_asc['cumulative_vectors'].searchsorted(target_bottom_25_vec)
    bottom_25_threshold = sorted_asc.iloc[bottom_25_idx]['vector_count'] if bottom_25_idx < len(sorted_asc) else np.nan

    # --- 2. Prepare for Output ---
    output_dir = os.path.splitext(csv_filepath)[0] + "_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nPlots will be saved to directory: '{output_dir}'")

    # Prepare summary text for plot annotations
    stats_summary = (
        f"IVF Partition Statistics Summary\n\n"
        f"Basic Information\n"
        f"  - Total Partitions (nlist): {nlist}\n"
        f"  - Non-Empty Partitions: {non_empty_partitions}\n"
        f"  - Total Vectors: {int(total_vectors)}\n"
        f"------------------------------------\n"
        f"Statistics by Partition Count (Original Q1/Q3)\n"
        f"  - Average Size: {avg_size:.2f}\n"
        f"  - Median Size: {int(median_size)}\n"
        f"  - Max / Min Size: {int(max_size)} / {int(min_size)}\n"
        f"  - Std Deviation: {std_dev:.2f}\n"
        f"  - Partition Size Quartiles (Q1/Q3): {int(q1)} / {int(q3)}\n"
        f"------------------------------------\n"
        f"Statistics by Vector Count (New Metrics)\n"
        f"  - Top 25% of vectors are in partitions of size >= {int(top_25_threshold)}\n"
        f"  - Top 75% of vectors are in partitions of size >= {int(top_75_threshold)}\n"
        f"  - Bottom 25% of vectors are in partitions of size <= {int(bottom_25_threshold)}"
    )
    print("\n" + stats_summary)

    # --- 3. Generate Plots ---
    
    # Plot 1: Sorted Partition Sizes (Bar Chart)
    plt.figure(figsize=(15, 8))
    sorted_df = df.sort_values(by='vector_count', ascending=False)
    plt.bar(range(len(sorted_df)), sorted_df['vector_count'], color='skyblue')
    plt.title('Sorted IVF Partition Sizes (Largest to Smallest)', fontsize=16)
    plt.xlabel('Partition Index (Sorted)', fontsize=12)
    plt.ylabel('Number of Vectors in Partition', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.text(0.95, 0.95, stats_summary, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "1_sorted_partition_sizes.png")
    plt.savefig(plot1_path)
    plt.close()
    print(f"Plot '1_sorted_partition_sizes.png' has been saved.")

    # Plot 2: Partition Size Distribution (Histogram)
    bins = [0, 1, 16, 32, 64, 128, 256, 512, 1024, 2048, np.inf]
    bins = [b for b in bins if b <= max_size]
    if max_size not in bins:
        bins.append(max_size)
    dynamic_labels = []
    for i in range(len(bins) - 1):
        start = int(bins[i] + 1)
        end = int(bins[i+1])
        if i == 0 and bins[i] == 0:
            start = 1
        dynamic_labels.append(f"{start}-{end}")

    df['size_bin'] = pd.cut(df['vector_count'], bins=bins, labels=dynamic_labels, right=True, include_lowest=True)
    distribution = df['size_bin'].value_counts().sort_index()

    plt.figure(figsize=(15, 8))
    distribution.plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.title('IVF Partition Size Distribution Histogram', fontsize=16)
    plt.xlabel('Vector Count Range in Partition', fontsize=12)
    plt.ylabel('Number of Partitions', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(distribution):
        plt.text(i, v + distribution.max()*0.01, str(v), ha='center', va='bottom')
    plt.text(0.95, 0.95, stats_summary, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "2_partition_size_distribution.png")
    plt.savefig(plot2_path)
    plt.close()
    print(f"Plot '2_partition_size_distribution.png' has been saved.")

    # Plot 3: Box Plot of Partition Sizes
    plt.figure(figsize=(10, 8))
    plt.boxplot(vector_counts.dropna(), vert=False, patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red', linewidth=2))
    plt.title('IVF Partition Size Box Plot', fontsize=16)
    plt.xlabel('Number of Vectors in Partition', fontsize=12)
    plt.yticks([])
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.text(0.95, 0.95, stats_summary, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, "3_partition_size_boxplot.png")
    plt.savefig(plot3_path)
    plt.close()
    print(f"Plot '3_partition_size_boxplot.png' has been saved.")
    print("\nAnalysis and plotting complete!")


def create_dummy_csv(filename="dummy_ivf_stats.csv", num_partitions=4096):
    """Creates a dummy CSV file for demonstration."""
    print(f"Creating a dummy CSV file for demonstration: '{filename}'")
    with open(filename, 'w') as f:
        f.write("partition_id,vector_count\n")
        for i in range(num_partitions):
            if random.random() > 0.1:
                size = int(np.random.lognormal(mean=4, sigma=1.5)) + 1
                if size > 0:
                    f.write(f"{i},{size}\n")
    print("Dummy CSV file created successfully.")


if __name__ == '__main__':
    # --- Example Usage ---
    
    # 1. (Optional) Try to set up a CJK font. If it fails, the script will continue with English defaults.
    # setup_cjk_font()

    # 2. Define your CSV filename.
    #    Replace 'your_ivf_stats.csv' with your actual filename.
    stats_filename = "./gist/base_d960_nlist15625_HNSW32_IVFFlat_ivf_stats.csv"

    # 3. (Optional) If your statistics file doesn't exist, a dummy file will be created.
    if not os.path.exists(stats_filename):
        create_dummy_csv(stats_filename, num_partitions=4096)

    # 4. Run the analysis and plotting function.
    analyze_and_plot_ivf_stats(stats_filename)