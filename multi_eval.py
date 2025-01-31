import numpy as np
import matplotlib.pyplot as plt


def rolling_average(arr, ROLLING_WINDOW_SIZE):
    cumsum = arr.cumsum()
    cumsum[ROLLING_WINDOW_SIZE:] -= cumsum[:-ROLLING_WINDOW_SIZE]
    return np.array((cumsum[ROLLING_WINDOW_SIZE - 1:] / ROLLING_WINDOW_SIZE))


    
def plot_avg_std(color,array,start,end,alpha):
    
    data = np.array(array[start:end])
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    
    iteration = np.arange(len(mean))
    
    plt.plot(iteration, mean, label=r'$\gamma = $' + f'{alpha}', color=color, lw=2)
    #plt.fill_between(iteration, mean - std_dev, mean + std_dev, color=color, alpha=0.2)
    plt.ylim(-200,250)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Cumulative Reward', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14, loc='lower right')
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', length=4, color='gray', labelsize=14)
    plt.tick_params(axis='both', which='major', length=6, color='black', labelsize=14)
    plt.tick_params(top=True, right=True, direction='in', length=6)
    plt.tick_params(which='minor', top=True, right=True, direction='in', length=4)
    plt.savefig('sd_corrected_cumrewardgraph.pdf')
    
   
    
sample_size=10
runs=range(1,sample_size+1)
alphas=[0.8,0.9,0.99]

file_paths = [f"cumulative_rewards/cumulative_rewards_{i}_alpha_{j}.npy" for i in runs for j in alphas]
cumulative_rewards = [np.load(path) for path in file_paths]
cumulative_rewards = np.array(cumulative_rewards)  
print(cumulative_rewards.shape)
window_size = 500
rolling_averages = [rolling_average(rewards,window_size) for rewards in cumulative_rewards]
rolling_averages = np.array(rolling_averages)
print(rolling_averages.shape)




plt.figure(figsize=(10, 6))
total_files = len(file_paths)
start_vals = range(0,total_files,sample_size)
plot_avg_std('blue',rolling_averages,start_vals[0],start_vals[0]+sample_size,alphas[0])
plot_avg_std('red',rolling_averages,start_vals[1],start_vals[1]+sample_size,alphas[1])
plot_avg_std('green',rolling_averages,start_vals[2],start_vals[2]+sample_size,alphas[2])


plt.grid(True)
plt.show()