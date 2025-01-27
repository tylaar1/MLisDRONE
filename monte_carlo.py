from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import os


class MCController(FlightController):

    def __init__(self):
        self.alpha=0.05 #later these should have ways of varying these parameters to compare results
        self.gamma=0.9
        self.epsilon=1
        self.epsilon_decay=0.001
        self.epsilon_min=0.1

        self.actions = [ 
            (round(thrust_left,2), round(thrust_right,2)) #round due to floating points
            for thrust_left in np.arange(0, 1.25, 0.25) 
            for thrust_right in np.arange(0, 1.25, 0.25)
        ]
        self.q_values={}
        self.state_size=10 #maximum state space is 10 from target
        '''should this be based around centre of screen or drone?'''
    def discretize_state(self, drone:Drone):
        x_target,y_target=drone.get_next_target()
        x_dist=drone.x-x_target #this should be acceptable regardless of which way around it is as learns same pattern
        y_dist=drone.y-y_target
        targets_remaining = len(drone.target_coordinates)
        state=np.array([int(x_dist*10),int(y_dist*10),targets_remaining]) #*10 means round to nearest .1 rather than 1, int works by truncating rather than traditional rounding
        
        return tuple(np.clip(state, 1-self.state_size, self.state_size - 1)) #state can be no more than 9 away from target in either direction
    #room to expand state to include velocity,pitch
    def distance(self, drone: Drone): #distance from target
        distance_array = self.discretize_state(drone)
        distance = (distance_array[0]**2+distance_array[1]**2)**0.5
        return distance
    def reward(self,drone: Drone):
        distance = self.distance(drone)
        if drone.has_reached_target_last_update: 
            return 100
        if distance > 9: #if drone goes too far from target enforce large punishment
            return -50  
        else:
            return 10-distance #reward for getting closer punishes for getting further away
    def update_q_vals(self,episode): 
        G = 0 
        visited = set() #set unordered
        for state, index, reward in reversed(episode): #unpack in reverse as we are working backwards
            G = reward + self.gamma * G  #g continually updated according to MC formula
            if (state, index) not in visited: #state-action pair only visited once per episode 
                visited.add((state, index))  
                if state not in self.q_values: 
                    self.q_values[state] = np.full(len(self.actions), 0.1) #small initial value
                self.q_values[state][index] += self.alpha * (G - self.q_values[state][index])
   
    def train(self,drone: Drone):
        epochs = 10000 #number of training loops
        cumulative_rewards=[] 
        for i in range(epochs): 
            drone = self.init_drone() #reset the drone
            actions=self.get_max_simulation_steps() #actions and delta time are set to equal what they are in pygame simulation
            delta_time= 10*self.get_time_interval() #it is rare to actually leave a state in a given step so check every 10 steps
            episode=[]
            cumulative_reward=0 
            for i in range(actions):
                state=self.discretize_state(drone)
                thrusts = self.get_thrusts(drone,training=True) 
                index = thrusts[1] 
                  
                drone.set_thrust(thrusts) #take action
                drone.step_simulation(delta_time=delta_time) #update drone state
               
                reward = self.reward(drone)
                cumulative_reward += reward
                episode.append((state,index,reward)) #collect together for use in update_q_vals
               
                
                if drone.has_reached_target_last_update: #this makes simulation stop after target reached, for all 4 targets comment this out.
                    #print(f"Target reached at step {i}")
                    
                    break
                
                if self.distance(drone) > 10: #has already recieved large punishment for this to discourage behaviour
                    #print(f"Drone has gone too far from the target at step {i}")
                    break
            self.update_q_vals(episode)
            cumulative_rewards.append(cumulative_reward)
            #print(self.q_values) 
        # plt.plot(range(1,epochs+1),cumulative_rewards)
        # plt.xlabel('Epochs')
        # plt.ylabel('Cumulative Reward')
        # plt.show()
        #we should do this multiple times and get average and standard deviation for plotting purposes
        #print(self.q_values)  
        #print('cumulative reward array:',cumulative_rewards)
        directory = "cumulative_rewards"
        file_path = os.path.join(directory, "cumulative_rewards_4.npy")
        np.save(file_path, cumulative_rewards)
        print(f"Cumulative rewards saved to {file_path}")
        
    def get_thrusts(self, drone: Drone,training=False) -> Tuple[float, float]:
        state=self.discretize_state(drone)
        if state not in self.q_values:
            self.q_values[state]=np.full(len(self.actions), 0.1)#small initial value to encourage exploration
        if training:
            if np.random.rand() < self.epsilon: #epsilon greedy movement strategy - happy this works as expected
                index=np.random.randint(len(self.actions))
                self.epsilon *= (1 - self.epsilon_decay)
                self.epsilon=max(self.epsilon,self.epsilon_min) #currently only decreasing after a random action taken but can modify so that its after any action
                return self.actions[index],index #automatically return these values as index same shape as expected output
            else:
                q_vals= self.q_values[state]
                max_q_val = np.max(q_vals)
                max_q_indices = np.flatnonzero(q_vals == max_q_val) #indexes of max q vals
                index = np.random.choice(max_q_indices)  #for when 2 q vals have same max
                return self.actions[index],index
        else:
            q_vals= self.q_values[state]
            max_q_val = np.max(q_vals)
            max_q_indices = np.flatnonzero(q_vals == max_q_val) #indexes of max q vals
            index = np.random.choice(max_q_indices)  #for when 2 q vals have same max
            return self.actions[index],index
    def load(self,filename="q_values.npy"):
        q_values_list = np.load(filename, allow_pickle=True)
        self.q_values = {state: q_vals for state, q_vals in q_values_list}
        print('Q-values loaded from', filename) 
    def save(self,filename="q_values.npy"):
        q_values_list = [(state, q_vals) for state, q_vals in self.q_values.items()]
        np.save(filename, q_values_list,allow_pickle=True)
        print('Q-values saved to', filename) 