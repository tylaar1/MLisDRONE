from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np

class CustomController(FlightController,Drone):
    episodes=2
    def __init__(self):
        super().__init__()
        self.alpha = 0.1  
        self.gamma = 0.9 
        self.epsilon = 0.2 
        self.delta_time=0.01
        self.actions = [ #as there are many options here we may get curse of dimensionality
            (round(thrust_left, 1), round(thrust_right, 1))
            for thrust_left in np.arange(0.0, 1.1, 0.1) #changing to 0.2 would quater curse dimensionality - experiment for when model working
            for thrust_right in np.arange(0.0, 1.1, 0.1)
        ]
        self.q_table = {}
        self.state_space_size = 100
    def discretize_state(self,x,y,target_x,target_y,velocity_x,velocity_y,pitch,pitch_velocity): 
        distance_to_target = np.sqrt((x - target_x)**2 + (y - target_y)**2)

        dist_state= int(distance_to_target * 10)  #multiply by 10 means we essentially round to nearest 0.1 rathrer than nearest 1
        dist_state= min(dist_state, self.state_space_size - 1)  #make sure drone stays in screen 
        #important for q learning as if space infinate drone can fly in wrong direction forever in name of exploration
        
        #adding velocity and pitch info so we can make decisions based off these as well - more curse of dimensionality but more info
        v_state_x = int(velocity_x * 10) #again multiply by 10
        v_state_y = int(velocity_y * 10)
        
        pitch_state = int(pitch * 10)
        v_pitch_state = int(pitch_velocity*10)
        state = (dist_state, v_state_x, v_state_y, pitch_state,v_pitch_state)
        return state
  
    def get_reward(self, distance_to_target):
        if distance_to_target < 0.1:  #reward for hitting target
            reward = 100
        else:
            reward = -distance_to_target #directly encourage movement towards target as target is known
            #can explore adding non linearity here once basic model working
        return reward 
    #will be implimented in train later
    '''   
    def q_learning_formula(self, current_state, action, reward, next_state,):
        if state not in self.q_table: #initialise q value table each time a new state is entered
            self.q_table[state] = np.zeros(len(self.actions))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))
        current_q = self.q_table[current_state + (action,)]

        max_next_q = np.max(self.q_table[next_state])

        #Q-learning formula
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[current_state + (action,)] = new_q   
    ''' 
                
    def train(self, drone=Drone, episodes=2, delta_time=0.01): #i dont think these are actually doing anything idk why episode not working
        for episode in range(episodes):
        
            drone.x, drone.y = 0, 0 #initialise coords- i think this is centre
            total_reward = 0 #initialise reward

            for step in range(10000):  #this also dont seem to be working it gets initialised by flight controller class instead
                
                thrusts = self.get_thrusts(drone)
                self.set_thrust(thrusts)

                distance_to_target = self.step_simulation()
                reward = self.get_reward(distance_to_target)

                #making the current state discreet
                state = self.discretize_state(drone.x, drone.y,drone.get_next_target()[0],drone.get_next_target()[1],drone.velocity_x,drone.velocity_y,drone.pitch,drone.pitch_velocity)
                next_state = self.discretize_state(drone.x, drone.y,drone.get_next_target()[0],drone.get_next_target()[1],drone.velocity_x,drone.velocity_y,drone.pitch,drone.pitch_velocity)

                action_idx = self.actions.index(thrusts)
                q_update = reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action_idx]
                self.q_table[state][action_idx] += self.alpha * q_update

                total_reward += reward

          
    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        state = self.discretize_state(drone.x, drone.y,self.get_next_target()[0],self.get_next_target()[1],self.velocity_x,self.velocity_y,self.pitch,self.pitch_velocity)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if np.random.rand() < self.epsilon: #epsilon greedy movement strategy
            return self.actions[np.random.randint(len(self.actions))]
        else:
            q_values = self.q_table[state]
            max_q_value = np.max(q_values)
            max_q_indices = np.flatnonzero(q_values == max_q_value) #indexes of max q vals
            chosen_idx = np.random.choice(max_q_indices)  #for when 2 q vals have same max
            return self.actions[chosen_idx]
           

    def load(self):
        filename = (f"./results/q_vals_")
        self.q_values = np.load(filename)
        pass
    def save(self):
        filename = (f"./results/q_vals_")
        np.save(filename, self.q_values)
            