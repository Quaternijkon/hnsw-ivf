import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy import stats

# ==============================================================================
# 0. Configuration
# ==============================================================================
# Paths for data file and output directory
# Assuming this script is in the same directory as the previous one, 
# and the data file is in the './gist' subdirectory
DATA_DIR = "./gist"
STATS_FILE = os.path.join(DATA_DIR, "search_partition_ratios.txt")
OUTPUT_DIR = os.path.join(DATA_DIR, "analysis_charts")


# ==============================================================================
# 1. Core Analysis and Plotting Functions
# ==============================================================================

def plot_histogram(data, output_path):
    """Generate and save a histogram of the search ratios."""
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Plot histogram and Kernel Density Estimate (KDE) using Seaborn
    ax = sns.histplot(data, kde=True, bins=50)
    
    # Calculate and annotate mean and median
    mean_val = np.mean(data)
    median_val = np.median(data)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}%')
    plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}%')
    
    plt.title('Distribution Histogram of Search Partition Coverage Ratio', fontsize=16)
    plt.xlabel('Ratio of Points in Hit Partition to Total Points (%)', fontsize=12)
    plt.ylabel('Number of Queries', fontsize=12)
    plt.legend()
    
    # Save the figure
    plt.savefig(output_path)
    plt.close()
    print(f"Histogram saved to: {output_path}")

def plot_boxplot(data, output_path):
    """Generate and save a boxplot of the search ratios."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    sns.boxplot(x=data)
    
    plt.title('Boxplot of Search Partition Coverage Ratio', fontsize=16)
    plt.xlabel('Ratio of Points in Hit Partition to Total Points (%)', fontsize=12)
    
    # Save the figure
    plt.savefig(output_path)
    plt.close()
    print(f"Boxplot saved to: {output_path}")

def plot_cdf(data, output_path):
    """Generate and save a Cumulative Distribution Function (CDF) plot of the search ratios."""
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Sort the data
    sorted_data = np.sort(data)
    # Calculate cumulative probabilities
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
    
    plt.plot(sorted_data, yvals, marker='.', linestyle='none', label='Data Points')

    # --- 新增代码：开始 ---
    # 定义硬编码的值并转换为百分比
    specific_value_ratio = 64 / 15625
    specific_value_percent = specific_value_ratio * 100

    # 计算该特定值的百分位数 (数据中小于或等于该值的点的比例)
    # 使用 atexit.register(stats.percentileofscore) 也可以, 但为了减少依赖直接计算
    percentile_of_value = np.sum(data <= specific_value_percent) / len(data)

    # 绘制特定值的垂直和水平分位线
    plt.axvline(specific_value_percent, color='darkviolet', linestyle=':', linewidth=2, 
                label=f'Quantile for {specific_value_percent:.4f}%: {percentile_of_value:.2%}')
    plt.axhline(percentile_of_value, color='darkviolet', linestyle=':', linewidth=2)
    # --- 新增代码：结束 ---
    
    # Annotate key percentile points
    p25 = np.percentile(data, 25)
    p50 = np.median(data)
    p90 = np.percentile(data, 90)
    p99 = np.percentile(data, 99)

    # 绘制25%分位线
    plt.axvline(p25, color='purple', linestyle='--', linewidth=1.5, label=f'25th Percentile: {p25:.2f}%')
    plt.axhline(0.25, color='purple', linestyle='--', linewidth=1.5)

    plt.axvline(p50, color='green', linestyle='--', linewidth=1.5, label=f'50th Percentile (Median): {p50:.2f}%')
    plt.axhline(0.5, color='green', linestyle='--', linewidth=1.5)
    
    plt.axvline(p90, color='orange', linestyle='--', linewidth=1.5, label=f'90th Percentile: {p90:.2f}%')
    plt.axhline(0.9, color='orange', linestyle='--', linewidth=1.5)
    
    plt.axvline(p99, color='red', linestyle='--', linewidth=1.5, label=f'99th Percentile: {p99:.2f}%')
    plt.axhline(0.99, color='red', linestyle='--', linewidth=1.5)

    plt.title('CDF of Search Partition Coverage Ratio', fontsize=16)
    plt.xlabel('Ratio of Points in Hit Partition to Total Points (%)', fontsize=12)
    plt.ylabel('Cumulative Probability (Proportion of Queries)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save the figure
    plt.savefig(output_path)
    plt.close()
    print(f"CDF plot saved to: {output_path}")

# ==============================================================================
# 2. Main Function
# ==============================================================================

def main():
    """Main execution function."""
    print("--- Starting Analysis of Search Partition Coverage Ratio Data ---")
    
    # Step 1: Check if the input file exists
    if not os.path.exists(STATS_FILE):
        print(f"Error: Statistics file not found! Please ensure '{STATS_FILE}' exists.")
        print("Please run the previous script to generate this file first.")
        # Create dummy data for demonstration if file doesn't exist
        print("Creating dummy data for demonstration purposes...")
        os.makedirs(DATA_DIR, exist_ok=True)
        # Generate data that looks somewhat like a ratio distribution
        dummy_data = np.random.beta(a=2, b=10, size=1000) * 0.2 
        np.savetxt(STATS_FILE, dummy_data)
        print(f"Dummy data saved to '{STATS_FILE}'.")

    # Step 2: Create the output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Charts will be saved to the directory: {OUTPUT_DIR}")
    
    # Step 3: Load the data
    try:
        ratios = np.loadtxt(STATS_FILE)
        # Convert ratios to percentages for easier interpretation
        ratios_percent = ratios * 100
        print(f"Successfully loaded {len(ratios_percent)} query records.")
    except Exception as e:
        print(f"Error: Failed to load file '{STATS_FILE}'. Reason: {e}")
        sys.exit(1)

    # Step 4: Calculate and print descriptive statistics
    print("\n" + "="*50)
    print("Descriptive Statistics for Search Coverage Ratios:")
    print(f"  Total Queries:   {len(ratios_percent)}")
    print(f"  Mean:            {np.mean(ratios_percent):.4f}%")
    print(f"  Median:          {np.median(ratios_percent):.4f}%")
    print(f"  Std Dev:         {np.std(ratios_percent):.4f}%")
    print(f"  Min:             {np.min(ratios_percent):.4f}%")
    print(f"  Max:             {np.max(ratios_percent):.4f}%")
    print(f"  25th Percentile: {np.percentile(ratios_percent, 25):.4f}%")
    print(f"  75th Percentile: {np.percentile(ratios_percent, 75):.4f}%")
    print(f"  95th Percentile: {np.percentile(ratios_percent, 95):.4f}%")
    print("="*50 + "\n")
    
    # Step 5: Generate and save the charts
    plot_histogram(ratios_percent, os.path.join(OUTPUT_DIR, "ratio_distribution_histogram.png"))
    plot_boxplot(ratios_percent, os.path.join(OUTPUT_DIR, "ratio_distribution_boxplot.png"))
    plot_cdf(ratios_percent, os.path.join(OUTPUT_DIR, "ratio_distribution_cdf.png"))
    
    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()
