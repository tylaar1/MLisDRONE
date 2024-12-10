from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np

class CustomController(FlightController):

    def __init__(self):
        return state  
    def train(self):
        pass    
    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        return (0.5, 0.5) # Replace this with your custom algorithm
    def load(self):
        pass
    def save(self):
        pass