from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import os

'''to do list 
1. make it so epsilon greedy policy exists in training loop rather than get thrusts + figure out why drone.reached target since 
last update works in main but not training loop (potentially to do with editing shape of data to include index)
2. we overwrite the same q values every time need system to store different saves based on input parameters eg alpha values for 
comparison in report
3. expand state and see if can find all four targets - this is basic model now fully complete
4. experiment with adding in epsilon decay and best parameters (i.e loop to try alpha =0.01, 0.05, 0.1 etc)
5. look at extensions on readme
'''
class CustomController(FlightController):

    def __init__(self):
        self.alpha=0.1 #later these should have ways of varying these parameters to compare results
        self.gamma=0.9
        self.epsilon=0.2
        #later should add epsilon decay for more exploration at start more exploitation at end

        self.actions = [ #as there are many options here we may get curse of dimensionality
            (round(thrust_left, 1), round(thrust_right, 1)) #round due to floating points
            for thrust_left in np.arange(0, 1.1, 0.2) #changing to 0.2 would quater curse dimensionality - experiment for when model working
            for thrust_right in np.arange(0, 1.1, 0.2)
        ]
        #print(len(self.actions))
        self.q_values={}
        self.state_size=64
    def discretize_state(self, drone:Drone):
        x_target,y_target=drone.get_next_target()
        distance=np.sqrt((drone.x-x_target)**2+(drone.y-y_target)**2) # BEN: DISTANCE FROM DRONE TO TARGET
        '''
        this will learn to perform the same action every time the distance away is the same 
        regardless of specific x,y coords and other stuff like velocity and pitch
        - will obvs need updating but ok for now as just trying to get it to learn to move to first target
        '''
        state=int(distance*10) #*10 means round to nearest .1 rather than 1
        return min(state,self.state_size-1) #this doesnt acc confine to screen just means state cant be further than 1 screen away no control over dynamics
    #room to expand state to include velocity,pitch
    def reward(self,drone: Drone):
        distance = self.discretize_state(drone) #this only works for now as states categorised by distance only, will need updating when discretize state updated
        #if drone.has_reached_target_last_update: #this is obviously something to be played with
        #    return 100
        if distance == 0:
            return 100 #drone.has_reached_target_last_update doesnt seem to give any rewards
        else:
            return -distance #experimenting with positive reward +1 to avoid divide by 0 error at target
    def update_q_vals(self, drone: Drone, state, index,reward, new_state):
        max_q_new_state = np.max(self.q_values[new_state])
        q_current =self.q_values[state][index]
        new_q = q_current + self.alpha * (reward + self.gamma * max_q_new_state - q_current)
        #print('new_q:',new_q)
        return new_q
          
    def train(self,drone: Drone):
        epochs = 10 #number of training loops
        cumulative_rewards=[]
        for i in range(epochs):
            actions=100000#max number action per loop 
            cumulative_reward=0
            for i in range(actions):
                thrusts = self.get_thrusts(drone) 
                index = thrusts[1]
                #print(thrusts)
                #print(index)
                state=self.discretize_state(drone) 
                #reward = self.reward(drone)
                #cumulative_reward += reward
                #print(thrusts) #doesnt appear to currently be 
                drone.set_thrust(thrusts) #take action
                drone.step_simulation(delta_time=0.01)
                new_state=self.discretize_state(drone) 
                reward = self.reward(drone)
                cumulative_reward += reward
                #print(f"Drone Position: ({drone.x}, {drone.y}), Velocity: ({drone.velocity_x}, {drone.velocity_y})")

                if new_state not in self.q_values:
                    self.q_values[new_state]=np.zeros(len(self.actions))
                self.q_values[new_state][index]=self.update_q_vals(drone, state, index, reward, new_state)
                cumulative_reward += reward
                
                #if drone.has_reached_target_last_update: #only aiming for first target for now 
                 #   cumulative_rewards.append(cumulative_reward)   
                  #  break 
            cumulative_rewards.append(cumulative_reward)
            #print(self.q_values) 
        print(self.q_values)  
        print('cumulative reward array:',cumulative_rewards)
         
    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        state=self.discretize_state(drone)
        if state not in self.q_values:
            self.q_values[state]=np.zeros(len(self.actions))#one q value per action 
        if np.random.rand() < self.epsilon: #epsilon greedy movement strategy - happy this works as expected
            index=np.random.randint(len(self.actions))
            return self.actions[index],index #automatically return these values as index same shape as expected output
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