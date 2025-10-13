import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import matplotlib.font_manager as fm

def setup_cjk_font():
    """
    Intelligently tries to set a font that supports CJK characters and provides clear guidance.
    """
    # A list of common CJK fonts for different operating systems
    font_candidates = [
        'SimHei', 'Microsoft YaHei',  # Windows
        'PingFang SC', 'STHeiti',     # macOS
        'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Source Han Sans SC'  # Linux
    ]
    
    print("Attempting to set a CJK-compatible font...")
    for font_name in font_candidates:
        try:
            # findfont will validate if the font exists, raising an exception if not
            fm.findfont(font_name, fallback_to_default=False)
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False # This line is important for displaying minus signs correctly
            print(f"Success! Font has been set to '{font_name}'.")
            return
        except:
            # If not found, continue to the next candidate
            continue
    
    # If the loop completes and no font was found
    print("\nWarning: Could not find any recommended CJK fonts on your system.")
    print("Chinese, Japanese, or Korean characters in the plots may not display correctly.")
    print("To fix this, please install a CJK font. For example, on Linux:")
    print("  - For Debian/Ubuntu systems: sudo apt-get update && sudo apt-get install -y fonts-wqy-zenhei")
    print("  - For CentOS/RHEL systems: sudo yum install -y wqy-zenhei-fonts")
    print("After installing a font, please restart the script.\n")


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
    
    # Ensure column names are correct
    if not {'partition_id', 'vector_count'}.issubset(df.columns):
        print(f"Error: CSV file '{csv_filepath}' must contain 'partition_id' and 'vector_count' columns.")
        return

    vector_counts = df['vector_count']
    
    # Core statistics
    nlist = len(df) # Assuming the CSV contains all partitions
    non_empty_partitions = len(vector_counts.dropna())
    total_vectors = vector_counts.sum()
    max_size = vector_counts.max()
    min_size = vector_counts.min()
    avg_size = vector_counts.mean()
    std_dev = vector_counts.std()
    variance = vector_counts.var()
    median_size = vector_counts.median()
    q1 = vector_counts.quantile(0.25)
    q3 = vector_counts.quantile(0.75)

    # --- 2. Prepare for Output ---
    # Create a directory to save plots
    output_dir = os.path.splitext(csv_filepath)[0] + "_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nPlots will be saved to directory: '{output_dir}'")

    # Prepare summary text for annotations on plots
    stats_summary = (
        f"IVF Partition Statistics Summary\n\n"
        f"Total Partitions (nlist): {nlist} (Estimated)\n"
        f"Non-Empty Partitions: {non_empty_partitions}\n"
        f"Total Vectors: {total_vectors}\n"
        f"------------------------------------\n"
        f"Average Partition Size: {avg_size:.2f}\n"
        f"Standard Deviation (Std Dev): {std_dev:.2f}\n"
        f"Variance: {variance:.2f}\n"
        f"Median: {median_size}\n"
        f"------------------------------------\n"
        f"Max Partition Size: {max_size}\n"
        f"Min Partition Size: {min_size}\n"
        f"Quartiles: Q1={q1}, Q3={q3}"
    )
    print("\n" + stats_summary)

    # --- 3. Generate Plots ---
    
    # Plot 1: Sorted Partition Sizes (Bar Chart)
    plt.figure(figsize=(15, 8))
    sorted_df = df.sort_values(by='vector_count', ascending=False)
    plt.bar(range(non_empty_partitions), sorted_df['vector_count'], color='skyblue')
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
    
    # Adjust bins based on max_size to avoid empty high-range bins
    # This keeps the bins relevant to the actual data range
    bins = [b for b in bins if b <= max_size]
    if max_size not in bins:
        bins.append(max_size)
    
    # Generate labels dynamically based on the adjusted bins
    dynamic_labels = []
    for i in range(len(bins) - 1):
        start = int(bins[i] + 1)
        end = int(bins[i+1])
        if i == 0 and bins[i] == 0:
             start = 1 # The first bin is for a single item
        if end == np.inf:
            dynamic_labels.append(f">{int(bins[i])}")
        else:
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
    # Add count labels on top of each bar
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
    plt.boxplot(vector_counts.dropna(), vert=False, patch_artist=True, # Use dropna() for robustness
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red', linewidth=2))
    plt.title('IVF Partition Size Box Plot', fontsize=16)
    plt.xlabel('Number of Vectors in Partition', fontsize=12)
    plt.yticks([]) # Hide y-axis ticks as they are not meaningful here
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
        # Create a distribution that is somewhat skewed, typical for IVF indexes
        for i in range(num_partitions):
            if random.random() > 0.1: # 90% chance of being non-empty
                # Use a log-normal-like distribution for sizes
                size = int(np.random.lognormal(mean=4, sigma=1.5)) + 1
                if size > 0:
                    f.write(f"{i},{size}\n")
    print("Dummy CSV file created successfully.")


if __name__ == '__main__':
    # --- Example Usage ---
    
    # 1. Set up matplotlib to handle non-ASCII characters in plots if needed.
    setup_cjk_font()

    # 2. Define your CSV filename.
    #    Replace 'your_ivf_stats.csv' with your actual filename.
    #    For example: stats_filename = "faiss_index_ivf_stats.csv"
    stats_filename = "./gist/base_d960_nlist15625_HNSW32_IVFFlat_ivf_stats.csv"

    # 3. (Optional) If your statistics file doesn't exist, the code below
    #    will create a dummy file for demonstration purposes.
    if not os.path.exists(stats_filename):
        create_dummy_csv(stats_filename, num_partitions=4096)

    # 4. Run the analysis and plotting function.
    analyze_and_plot_ivf_stats(stats_filename)