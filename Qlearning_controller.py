from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np

class CustomController(FlightController):

    def __init__(self):
        self.alpha = 0.1  
        self.gamma = 0.9 
        self.epsilon = 0.2 

        self.actions = [ #as there are many options here we may get curse of dimensionality
            (round(thrust_left, 1), round(thrust_right, 1))
            for thrust_left in np.arange(0.0, 1.1, 0.1) #changing to 0.2 would quater curse dimensionality - experiment for when model working
            for thrust_right in np.arange(0.0, 1.1, 0.1)
        ]
        self.q_table = {}
        self.state_space_size = 100
    def discretize_state(self,x,y,target_x,target_y): #as far as i am aware need discreet space for q learning
        distance_to_target = np.sqrt((x - target_x)**2 + (y - target_y)**2)

        state= int(distance_to_target * 10)  #multiply by 10 means we essentially round to nearest 0.1 rathrer than nearest 1
        state= min(state, self.state_space_size - 1)  #make sure drone stays in screen 
        #important for q learning as if space infinate drone can fly in wrong direction forever in name of exploration

        return state  
    def get_reward(self, distance_to_target):
        if distance_to_target < 0.1:  #reward for hitting target
            reward = 100
        else:
            reward = -distance_to_target #directly encourage movement towards target as target is known
            #can explore adding non linearity here once basic model working
        return reward 
       
        
    def train(self, drone, episodes, delta_t):
       
        for episode in range(episodes):
        
            drone.x, drone.y = 0, 0 #initialise coords- i think this is centre
            total_reward = 0 #initialise reward

            for step in range(10000):  #takes 10000 steps to find target(s)
                
                thrusts = self.get_thrusts(drone)
                drone.set_thrust(thrusts)

                distance_to_target = drone.step_simulation(delta_t)
                reward = self.get_reward(distance_to_target)

                # Discretize the current state
                state = self.discretize_state(drone.x, drone.y, *drone.get_next_target())
                # Next state
                next_state = self.discretize_state(drone.x, drone.y, *drone.get_next_target())
                
                if state not in self.q_table: #initialise q value table each time a new state is entered
                    self.q_table[state] = np.zeros(len(self.actions))
                if next_state not in self.q_table:
                    self.q_table[next_state] = np.zeros(len(self.actions))


                action_idx = self.actions.index(thrusts)
                q_update = reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action_idx]
                self.q_table[state][action_idx] += self.alpha * q_update


                total_reward += reward

            
    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        state = self.discretize_state(drone.x, drone.y, *drone.get_next_target())
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if np.random.rand() < self.epsilon: #epsilon greedy movement strategy
            return self.actions[np.random.randint(len(self.actions))]
        else:
            q_values = self.q_table[state]
            max_q_value = np.max(q_values)
            max_q_indices = np.flatnonzero(q_values == max_q_value)  # Indices of max Q-values
            chosen_idx = np.random.choice(max_q_indices)  # Random tie-breaking
            return self.actions[chosen_idx]
            

    def load(self):
        pass
    def save(self):
        pass