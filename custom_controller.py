from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np

class CustomController(FlightController):

    def __init__(self):
        self.alpha = 0.1  
        self.gamma = 0.9 
        self.epsilon = 1.0  #initialise epsilon
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01 #so that episilon cant get too low
        

        self.actions = [ #as there are many options here we may get curse of dimensionality
            (round(thrust_left, 1), round(thrust_right, 1))
            for thrust_left in np.arange(0.0, 1.1, 0.1)
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
        return reward     
    def train(self):
        pass    
    #epsilon greedy training policy to explore different routes
    #targets appear in same locations each time so only have to learn the route rather than to fly specifically to target
    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        return (0.5, 0.5) # Replace this with your custom algorithm
    def load(self):
        pass
    def save(self):
        pass