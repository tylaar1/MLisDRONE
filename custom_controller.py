from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

class CustomController(FlightController):

    def __init__(self):
        self.alpha=0.1 #later these should have ways of varying them to compare results
        self.gamma=0.9
        self.epsilon=0.2
        #later should add epsilon decay for more exploration at start more exploitation at end
        self.actions = [ #as there are many options here we may get curse of dimensionality
            (round(thrust_left, 1), round(thrust_right, 1))
            for thrust_left in np.arange(0.0, 1.1, 0.1) #changing to 0.2 would quater curse dimensionality - experiment for when model working
            for thrust_right in np.arange(0.0, 1.1, 0.1)
        ]
        print(len(self.actions))
        self.q_values={}
        self.state_size=64
    def discretize_state(self, drone:Drone):
        x_target,y_target=drone.get_next_target()
        distance=np.sqrt((drone.x-x_target)**2+(drone.y-y_target)**2)
        '''
        this will learn to perform the same action every time the distance away is the same 
        regardless of specific x,y coords and other stuff like velocity and pitch
        - will obvs need updating but ok for now as just trying to get it to learn to move to first target
        '''
        state=int(distance*10) #*10 means round to nearest .1 rather than 1
        return min(state,self.state_size-1) #this doesnt acc confine to screen just means state cant be further than 1 screen away no control over dynamics
    #room to expand state to include velocity,pitch
    def train(self,drone: Drone):
        '''
        Steps:  - Initialise drone - drone=Drone() or somthing to that effect
                - take action
                - update q table according to q learning formula
                - cumulative reward += step reward
                - Loop through these actions
        File "main.py", line 120, in <module>
        controller.train()
        TypeError: train() missing 1 required positional argument: 'drone'
        '''
        pass    
    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        state=self.discretize_state(drone)
        if state not in self.q_values:
            self.q_values[state]=np.zeros(len(self.actions))#one q value per action
        print(self.q_values) #we know this succesfully initiallises - just need way of acc updating q vals 
        if np.random.rand() < self.epsilon: #epsilon greedy movement strategy - happy this works as expected
            return self.actions[np.random.randint(len(self.actions))] #automatically return these values as index same shape as expected output
        else:
            '''
            some logic to pick the max q value from table 
            '''
            left_thrust=0.55
            right_thrust=0.5
        return (left_thrust, right_thrust) # Replace this with your custom algorithm
    def load(self):
        pass
    def save(self):
        pass