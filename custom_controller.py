from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

'''current issues to fix - in the order i think is most important
1. updating q value table - this currently updates the whole table for the given space not just the action, its the commented out line
at the bottom of train
2. need a few lines of code in get_thrusts to chose the action with the highest q value - should include random choice for tiebreaks
3. actually impliment the q learning formula instead of returning one - no point doing this until q table updating properly
4. get save + load functionality working
'''
class CustomController(FlightController):

    def __init__(self):
        self.alpha=0.1 #later these should have ways of varying these parameters to compare results
        self.gamma=0.9
        self.epsilon=0.2
        #later should add epsilon decay for more exploration at start more exploitation at end
        self.alpha = 0.1  
        self.gamma = 0.9 
        self.epsilon = 1.0  #initialise epsilon 

        self.actions = [ #as there are many options here we may get curse of dimensionality
            (round(thrust_left, 1), round(thrust_right, 1)) #round due to floating points
            for thrust_left in np.arange(0.0, 1.1, 0.1) #changing to 0.2 would quater curse dimensionality - experiment for when model working
            for thrust_right in np.arange(0.0, 1.1, 0.1)
        ]
        #print(len(self.actions))
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
    def reward(self,drone: Drone):
        if drone.has_reached_target_last_update:
            return 100
        else:
            return - self.discretize_state(drone) #this only works for now as states categorised by distance only, will need updating when discretize state updated
        
    def update_q_vals(self,drone: Drone):
        '''
        q vals update formula
        '''
        return 1 #check that code works up to here should see ones in dictionary corresponding to correct action and state  
    def train(self,drone: Drone):
        epochs = 10 #number of training loops
        cumulative_rewards=[]
        for i in range(epochs):
            actions=10 #max number action per loop (will add logic to stop loop once drone reaches target)
            drone=Drone()
            cumulative_reward=0
            for i in range(actions):
                thrusts = self.get_thrusts(drone) #want as percentage
                state=self.discretize_state(drone) #could return from get thrusts but then have to modify drone.py which i am reluctant to do
                print(thrusts) #doesnt appear to currently be 
                drone.set_thrust(thrusts) #take action
                new_state=self.discretize_state(drone) 
                if new_state not in self.q_values:
                    self.q_values[new_state]=np.zeros(len(self.actions))
                 
                #self.q_values[state]=self.update_q_vals(drone)
                if drone.has_reached_target_last_update: #only aiming for first target for now 
                    break 
              
       
                #cumulative_reward += step_reward
            #cumulative_rewards.append(cumulative_reward)
        #print(cumulative_rewards)
                
                
                
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
    
        thrustleft = 0.5
        thrustright = 0.5
        return (thrustleft, thrustright) # Replace this with your custom algorithm

    def load(self):
        pass
    def save(self):
        pass