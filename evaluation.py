import numpy as np
file_path = "cumulative_rewards/cumulative_rewards_1.npy"
file_path = "cumulative_rewards/cumulative_rewards_2.npy"
file_path = "cumulative_rewards/cumulative_rewards_3.npy"

# Load the array
cumulative_rewards = np.load(file_path)

print("Loaded cumulative rewards:", cumulative_rewards)

#functionality to load all files and plot average and standard deviation
#just need to do maybe 10 or so iterations and change cumulative rewards_1 to cumulative rewards_2 etc
#its a bit hacky sorry but means we wont have issue of q vals not reseting git branch branch_name
