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
            for thrust_left in np.arange(0.0, 1.1, 0.1)
            for thrust_right in np.arange(0.0, 1.1, 0.1)
        ]
        self.q_table = {}
        self.state_space_size = 100
        
    # def get_reward(self):
       
        
    def train(self):
        pass    
    #epsilon greedy training policy to explore different routes
    #targets appear in same locations each time so only have to learn the route rather than to fly specifically to target
    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        return (0.9, 0.45) # Replace this with your custom algorithm
    def load(self):
        pass
    def save(self):
        pass