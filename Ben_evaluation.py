import numpy as np
import matplotlib.pyplot as plt

# Load all cumulative rewards
file_paths = [f"cumulative_rewards/cumulative_rewards_run_{i}.npy" for i in range(1, 11)]
cumulative_rewards = [np.load(path) for path in file_paths]

def rolling_average(arr, window_size=500):
    cumsum = np.cumsum(arr)
    cumsum[window_size:] -= cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

# Calculate rolling averages
rolling_averages = [rolling_average(rewards) for rewards in cumulative_rewards]

def plot_avg_std(arrays):
    data = np.array(arrays)
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mean, label='Mean', color='blue', lw=2)
    plt.fill_between(range(len(mean)), mean - std_dev, mean + std_dev, color='blue', alpha=0.2, label='Standard Deviation')
    
    # Customize labels and ticks
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Classification Accuracy (%)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Add grid, legend, and minor ticks
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', length=4, color='gray', labelsize=14)
    plt.tick_params(axis='both', which='major', length=6, color='black', labelsize=14)
    plt.tick_params(top=True, right=True, direction='in', length=6)
    plt.tick_params(which='minor', top=True, right=True, direction='in', length=4)

# Adjust layout for better appearance
plt.tight_layout()

plot_avg_std(rolling_averages)