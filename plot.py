import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

epochs = np.load('./epochs.npy')
cumreward_df = pd.read_csv('./cumulative_rewards.csv')
cumreward_df = cumreward_df.iloc[:,2:10] # gets rid of epochs column

def rolling_average(arr, ROLLING_WINDOW_SIZE):
    cumsum = arr.cumsum()
    cumsum[ROLLING_WINDOW_SIZE:] -= cumsum[:-ROLLING_WINDOW_SIZE]
    return (cumsum[ROLLING_WINDOW_SIZE - 1:] / ROLLING_WINDOW_SIZE)

def plot_cumreward(epochs, mean, std, ROLLING_WINDOW_SIZE):
    # Plot the mean
    plt.plot(epochs, mean, color='r', label='Mean Cumulative Reward')
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
cumreward_mean = cumreward_df.to_numpy().mean(axis=1)
cumreward_std = cumreward_df.to_numpy().std(axis=1)

cumreward_mean_rollavg = rolling_average(cumreward_mean, ROLLING_WINDOW_SIZE)
cumreward_std_rollavg = rolling_average(cumreward_std, ROLLING_WINDOW_SIZE)


epochs_adj = epochs[ROLLING_WINDOW_SIZE - 1:]

plot_cumreward(epochs_adj, cumreward_mean_rollavg, cumreward_std_rollavg, ROLLING_WINDOW_SIZE)
