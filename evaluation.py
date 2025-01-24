import numpy as np
import matplotlib.pyplot as plt
file_path1 = "cumulative_rewards/cumulative_rewards_1.npy"
file_path2 = "cumulative_rewards/cumulative_rewards_2.npy"
file_path3 = "cumulative_rewards/cumulative_rewards_3.npy"
file_path4 = "cumulative_rewards/cumulative_rewards_4.npy"
file_path5 = "cumulative_rewards/cumulative_rewards_5.npy"
file_path6 = "cumulative_rewards/cumulative_rewards_6.npy"
file_path7 = "cumulative_rewards/cumulative_rewards_7.npy"
file_path8 = "cumulative_rewards/cumulative_rewards_8.npy"
file_path9  = "cumulative_rewards/cumulative_rewards_9.npy"
file_path10 = "cumulative_rewards/cumulative_rewards_10.npy"




# Load the array
cumulative_rewards1 = np.load(file_path1)
cumulative_rewards2 = np.load(file_path2)
cumulative_rewards3 = np.load(file_path3)
cumulative_rewards4 = np.load(file_path4)
cumulative_rewards5 = np.load(file_path5)
cumulative_rewards6 = np.load(file_path6)
cumulative_rewards7 = np.load(file_path7)
cumulative_rewards8 = np.load(file_path8)
cumulative_rewards9 = np.load(file_path9)
cumulative_rewards10 = np.load(file_path10)

def rolling_average(arr, ROLLING_WINDOW_SIZE):
    cumsum = arr.cumsum()
    cumsum[ROLLING_WINDOW_SIZE:] -= cumsum[:-ROLLING_WINDOW_SIZE]
    return np.array((cumsum[ROLLING_WINDOW_SIZE - 1:] / ROLLING_WINDOW_SIZE))


def plot_avg_std(*arrays):

    data = np.array(arrays)
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    

    iteration = np.arange(len(mean))
    plt.figure(figsize=(10, 6))
    plt.plot(iteration, mean, label='Mean', color='blue', lw=2)
    plt.fill_between(iteration, mean - std_dev, mean + std_dev, color='blue', alpha=0.2, label='Standard Deviation')
    
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


rolling_average1 = rolling_average(cumulative_rewards1, 500)
rolling_average2 = rolling_average(cumulative_rewards2, 500)    
rolling_average3 = rolling_average(cumulative_rewards3, 500)
rolling_average4 = rolling_average(cumulative_rewards4, 500)
rolling_average5 = rolling_average(cumulative_rewards5, 500)
rolling_average6 = rolling_average(cumulative_rewards6, 500)
rolling_average7 = rolling_average(cumulative_rewards7, 500)
rolling_average8 = rolling_average(cumulative_rewards8, 500)
rolling_average9 = rolling_average(cumulative_rewards9, 500)
rolling_average10 = rolling_average(cumulative_rewards10, 500)
plot_avg_std(rolling_average1, rolling_average2, rolling_average3, rolling_average4, rolling_average5, rolling_average6, rolling_average7, rolling_average8, rolling_average9, rolling_average10)