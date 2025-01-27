from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import os

class Ben_MCController(FlightController):

    def __init__(self):
        self.alpha=0.05
        self.gamma=0.9
        self.epsilon=1
        self.epsilon_decay=0.001
        self.epsilon_min=0.1
        self.runs = 10  # Number of total runs

        self.actions = [ 
            (round(thrust_left,2), round(thrust_right,2))
            for thrust_left in np.arange(0, 1.25, 0.25) 
            for thrust_right in np.arange(0, 1.25, 0.25)
        ]
        self.state_size=10
        self.final_q_values = []
        self.all_runs_results = []

    def discretize_state(self, drone:Drone):
        x_target,y_target=drone.get_next_target()
        x_dist=drone.x-x_target
        y_dist=drone.y-y_target
        targets_remaining = len(drone.target_coordinates)
        state=np.array([int(x_dist*10),int(y_dist*10),targets_remaining])
        
        return tuple(np.clip(state, 1-self.state_size, self.state_size - 1))

    def distance(self, drone: Drone):
        distance_array = self.discretize_state(drone)
        distance = (distance_array[0]**2+distance_array[1]**2)**0.5
        return distance

    # def reward(self,drone: Drone):
    #     distance = self.distance(drone)
    #     if drone.has_reached_target_last_update: 
    #         return 100
    #     if distance > 9:
    #         return -50  
    #     else:
    #         return 10-distance
    def reward(self, drone: Drone):
        distance = self.distance(drone)
        
        # Calculate angle to target
        x_target, y_target = drone.get_next_target()
        angle_to_target = np.arctan2(y_target - drone.y, x_target - drone.x)
        
        # Get current drone angle
        current_angle = drone.pitch
        
        # Calculate angle difference
        angle_diff = abs(angle_to_target - current_angle)
        
        # Normalize angle difference to be between 0 and pi
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
        
        # Reward components
        distance_reward = 10 - distance
        # Scale down angle reward significantly
        angle_reward = (np.pi - angle_diff) * 0.1  # Reduced by factor of 10 and multiplied by 0.1
        
        if drone.has_reached_target_last_update: 
            return 100
        if distance > 9:
            return -50  
        else:
            # Reduce angle reward contribution further
            return distance_reward + 0.1 * angle_reward  # Changed from 0.5 to 0.1

    def update_q_vals(self, episode): 
        G = 0 
        visited = set()
        for state, index, reward in reversed(episode):
            G = reward + self.gamma * G
            if (state, index) not in visited:
                visited.add((state, index))  
                if state not in self.q_values: 
                    self.q_values[state] = np.full(len(self.actions), 0.1)
                self.q_values[state][index] += self.alpha * (G - self.q_values[state][index])

    def get_thrusts(self, drone: Drone, training=False) -> Tuple[float, float]:
        state=self.discretize_state(drone)
        if state not in self.q_values:
            self.q_values[state]=np.full(len(self.actions), 0.1)
        
        if training:
            if np.random.rand() < self.epsilon:
                index=np.random.randint(len(self.actions))
                self.epsilon *= (1 - self.epsilon_decay)
                self.epsilon=max(self.epsilon,self.epsilon_min)
                return self.actions[index],index
            else:
                q_vals= self.q_values[state]
                max_q_val = np.max(q_vals)
                max_q_indices = np.flatnonzero(q_vals == max_q_val)
                index = np.random.choice(max_q_indices)
                return self.actions[index],index
        else:
            q_vals= self.q_values[state]
            max_q_val = np.max(q_vals)
            max_q_indices = np.flatnonzero(q_vals == max_q_val)
            index = np.random.choice(max_q_indices)
            return self.actions[index],index

    def train(self, drone: Drone):
        run_cumulative_rewards = []
        
        for run in range(self.runs):
            print(f"Processing Run {run+1}")
            
            # Reset for each run
            self.q_values = {}
            self.epsilon = 1
            cumulative_rewards = []
            
            epochs = 50000
            for i in range(epochs):
                drone = self.init_drone()
                actions = self.get_max_simulation_steps()
                delta_time = 10 * self.get_time_interval()
                episode = []
                cumulative_reward = 0
                
                for step in range(actions):
                    state = self.discretize_state(drone)
                    thrusts = self.get_thrusts(drone, training=True)
                    index = thrusts[1]
                    
                    drone.set_thrust(thrusts)
                    drone.step_simulation(delta_time=delta_time)
                    
                    reward = self.reward(drone)
                    cumulative_reward += reward
                    episode.append((state, index, reward))
                    
                    # if drone.has_reached_target_last_update:
                    #     break
                    
                    if self.distance(drone) > 10:
                        break
                
                self.update_q_vals(episode)
                cumulative_rewards.append(cumulative_reward)
            
            # Save results for this run
            directory = "cumulative_rewards"
            os.makedirs(directory, exist_ok=True)
            np.save(os.path.join(directory, f"anglecumulative_rewards_run_{run+1}.npy"), cumulative_rewards)
            
            # Store Q-values and rewards
            run_cumulative_rewards.append(cumulative_rewards)
            self.final_q_values.append(self.q_values)
        
        self.all_runs_results = run_cumulative_rewards
        
        # Plot results
        plt.figure(figsize=(10,6))
        for rewards in run_cumulative_rewards:
            plt.plot(rewards)
        plt.title('Cumulative Rewards Across Runs')
        plt.xlabel('Epochs')
        plt.ylabel('Cumulative Reward')
        plt.show()

    def load(self, filename="q_values.npy"):
        q_values_list = np.load(filename, allow_pickle=True)
        self.q_values = {state: q_vals for state, q_vals in q_values_list}
        print('Q-values loaded from', filename) 

    def save(self, filename="q_values.npy"):
        q_values_list = [(state, q_vals) for state, q_vals in self.q_values.items()]
        np.save(filename, q_values_list, allow_pickle=True)
        print('Q-values saved to', filename)