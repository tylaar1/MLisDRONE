from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np

class CustomController(FlightController):

    def __init__(self):
        self.alpha = 0.1  
        self.gamma = 0.9 
        self.epsilon = 1.0  
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        

        self.actions = [ #as there are many options here we may get curse of dimensionality
            (round(thrust_left, 1), round(thrust_right, 1))
            for thrust_left in np.arange(0.0, 1.1, 0.1)
            for thrust_right in np.arange(0.0, 1.1, 0.1)
        ]
        self.q_table = {}
        self.state_space_size = 100
        
    def get_reward(self):
       
        
    def train(self):
        pass    
    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        return (0.5, 0.5) # Replace this with your custom algorithm
    def load(self):
        pass
    def save(self):
        pass