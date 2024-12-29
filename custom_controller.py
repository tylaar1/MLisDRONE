from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import os

'''to do list 
1. figure out why drone.reached target since last update works in main but not training loop (potentially to do with editing
shape of data to include index)
2. we overwrite the same q values every time need system to store different saves based on input parameters eg alpha values for 
comparison in report
3. expand state and see if can find all four targets - this is basic model now fully complete
4. experiment with best parameters (i.e loop to try alpha =0.01, 0.05, 0.1 etc)
5. look at extensions on readme
'''

'''
pseudocode i found on git hub for reward based epsilon decay that could be implimented later - wont necessaraly improve model but 
good experiment
if EPSILON > MINIMUM_EPSILON and LAST_REWARD >= REWARD_THRESHOLD:    
    EPSILON = DECAY_EPSILON(EPSILON)    
    REWARD_THRESHOLD = INCREMENT_REWARD(REWARD_THRESHOLD)
'''
class CustomController(FlightController):

    def __init__(self):
        self.alpha=0.01 #later these should have ways of varying these parameters to compare results
        self.gamma=0.99
        self.epsilon=1
        self.epsilon_decay=0.01
        self.epsilon_min=0.1
        #later should add epsilon decay for more exploration at start more exploitation at end

        self.actions = [ #as there are many options here we may get curse of dimensionality
            (round(thrust_left,2), round(thrust_right,2)) #round due to floating points
            for thrust_left in np.arange(0, 1.25, 0.25) 
            for thrust_right in np.arange(0, 1.25, 0.25) 
        ]
        #print(len(self.actions))
        self.q_values={}
        self.state_size=10
    def discretize_state(self, drone:Drone):
        x_target,y_target=drone.get_next_target()
        x_dist=drone.x-x_target #this should be acceptable regardless of which way around it is as learns same pattern
        y_dist=drone.y-y_target
        state=np.array([int(x_dist*10),int(y_dist*10)]) #*10 means round to nearest .1 rather than 1, int works by truncating rather than traditional rounding
        return tuple(np.clip(state, 1-self.state_size, self.state_size - 1))#this doesnt acc confine to screen just means state cant be further than 1 screen away no control over dynamics
    #room to expand state to include velocity,pitch
    def distance(self, drone: Drone):
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
    def update_q_vals(self, state, index, reward, new_state):
        if state not in self.q_values:
            self.q_values[state] = np.full(len(self.actions), 0.1)  #initialise q values to 0.1
        if new_state not in self.q_values:
            self.q_values[new_state] = np.full(len(self.actions), 0.1)

        max_q_new_state = np.max(self.q_values[new_state])
        q_current = self.q_values[state][index]
        self.q_values[state][index] = q_current + self.alpha * (reward + self.gamma * max_q_new_state - q_current)         
   
          
    def train(self,drone: Drone):
        epochs = 10000 #number of training loops
        cumulative_rewards=[] 
        for i in range(epochs): 
            drone = self.init_drone() #reset the drone
            actions=self.get_max_simulation_steps() #actions and delta time are set to equal what they are in pygame simulation
            delta_time= 10*self.get_time_interval()  
         
            cumulative_reward=0 
            for i in range(actions):
                state=self.discretize_state(drone)
                thrusts = self.get_thrusts(drone,training=True) 
                index = thrusts[1] 
            
                drone.set_thrust(thrusts) #take action
                drone.step_simulation(delta_time=delta_time) #update drone state
                new_state=self.discretize_state(drone) 
                reward = self.reward(drone)
                cumulative_reward += reward
                #print(f"Drone Position: ({drone.x}, {drone.y}), Velocity: ({drone.velocity_x}, {drone.velocity_y})")

                if new_state not in self.q_values:
                    self.q_values[new_state]=np.full(len(self.actions), 0.1)
                self.update_q_vals(state, index, reward, new_state)
                
                
                if drone.has_reached_target_last_update:
                    #print(f"Target reached at step {i}")
                    cumulative_rewards.append(cumulative_reward)
                    break
                if self.distance(drone) > 10:
                    #print(f"Drone has gone too far from the target at step {i}")
                    break
            cumulative_rewards.append(cumulative_reward)
            #print(self.q_values) 
        print(self.q_values)  
        print('cumulative reward array:',cumulative_rewards)
         
    def get_thrusts(self, drone: Drone,training=False) -> Tuple[float, float]:
        state=self.discretize_state(drone)
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