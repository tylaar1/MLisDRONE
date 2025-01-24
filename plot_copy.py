import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data
epochs = np.load('./epochs.npy')
cumreward_df = pd.read_csv('./cumulative_rewards.csv')
cumreward_df = cumreward_df.iloc[:,2:]

def rolling_average(arr, ROLLING_WINDOW_SIZE):
    cumsum = arr.cumsum()
    cumsum[ROLLING_WINDOW_SIZE:] -= cumsum[:-ROLLING_WINDOW_SIZE]
    return (cumsum[ROLLING_WINDOW_SIZE - 1:] / ROLLING_WINDOW_SIZE)

def plot_cumreward(epochs, mean, std, ROLLING_WINDOW_SIZE, all_rewards=None):
    plt.figure(figsize=(10, 6))
    
    # Plot all individual rewards if provided
    if all_rewards is not None:
        for i, reward in enumerate(all_rewards.T):  # Transpose to loop through columns
            plt.plot(epochs, reward, label=f'Run {i+1}')
    
    # Plot the mean
    plt.plot(epochs, mean, color='r', label='Mean Cumulative Reward', linewidth=2)
    # Plot error bands
    plt.fill_between(epochs, mean - std, mean + std, color='r', alpha=0.2, label='Std Dev')
    
    # Labels and styling
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Cumulative Reward', fontsize=14)
    plt.title(f'Rolling Average = {ROLLING_WINDOW_SIZE}', fontsize=16)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', length=6, color='black', labelsize=14, direction='in', top=True, right=True)
    plt.tick_params(which='minor', length=4, color='gray')
    plt.legend(fontsize=10)
    plt.show()

ROLLING_WINDOW_SIZE = 700

# Compute rolling averages for each individual run
individual_rollavgs = []
for col in cumreward_df.columns:
    col_array = cumreward_df[col].to_numpy()
    col_rollavg = rolling_average(col_array, ROLLING_WINDOW_SIZE)
    individual_rollavgs.append(col_rollavg)

# Convert the list of individual rolling averages back to a 2D array
individual_rollavgs = np.stack(individual_rollavgs, axis=1)

# Compute mean and std from rolling averages
cumreward_mean_rollavg = individual_rollavgs.mean(axis=1)
cumreward_std_rollavg = individual_rollavgs.std(axis=1)

# Adjust epochs
epochs_adj = epochs[ROLLING_WINDOW_SIZE - 1:]

# Plot the results
plot_cumreward(epochs_adj, cumreward_mean_rollavg, cumreward_std_rollavg, ROLLING_WINDOW_SIZE, all_rewards=individual_rollavgs)
