from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import pickle

class MrChatGPTCustomController(FlightController):

    def __init__(self):
        # Q-learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate (starts high for exploration)

        # Actions (thrust combinations)
        self.actions = [
            (round(thrust_left, 1), round(thrust_right, 1))  # Round to avoid floating point precision issues
            for thrust_left in np.arange(0.0, 1.1, 0.1)
            for thrust_right in np.arange(0.0, 1.1, 0.1)
        ]
        self.q_values = {}  # Q-table initialized as an empty dictionary
        self.state_size = 64  # Max state index for discretized states

    def discretize_state(self, drone: Drone):
        """Discretize the state based on the distance to the target."""
        x_target, y_target = drone.get_next_target()
        distance = np.sqrt((drone.x - x_target)**2 + (drone.y - y_target)**2)
        state = int(distance * 10)  # Discretize by rounding to nearest 0.1
        return min(state, self.state_size - 1)

    def reward(self, drone: Drone):
        """Calculate reward based on distance to target."""
        if drone.has_reached_target_last_update:
            return 100
        else:
            return -self.discretize_state(drone)  # Negative reward for distance

    def update_q_vals(self, state, action_index, reward, next_state):
        """Update Q-values using the Q-learning formula."""
        if next_state not in self.q_values:
            self.q_values[next_state] = np.zeros(len(self.actions))  # Initialize Q-values for unseen state

        # Q-learning formula
        max_next_q = np.max(self.q_values[next_state])  # Maximum Q-value for the next state
        current_q = self.q_values[state][action_index]
        self.q_values[state][action_index] += self.alpha * (reward + self.gamma * max_next_q - current_q)

    def train(self):
        """Train the controller using Q-learning."""
        epochs = 100  # Number of training epochs
        max_steps_per_epoch = 500  # Max steps per episode

        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}")
            drone = Drone()  # Reset the drone for each epoch
            cumulative_reward = 0

            for step in range(max_steps_per_epoch):
                state = self.discretize_state(drone)

                # Ensure the Q-values for this state are initialized
                if state not in self.q_values:
                    self.q_values[state] = np.zeros(len(self.actions))

                # ε-greedy action selection with tie-breaking
                if np.random.rand() < self.epsilon:
                    action_index = np.random.randint(len(self.actions))  # Explore: Random action
                else:
                    best_actions = np.flatnonzero(self.q_values[state] == np.max(self.q_values[state]))
                    action_index = np.random.choice(best_actions)  # Exploit: Random tie-breaking

                thrusts = self.actions[action_index]
                drone.set_thrust(thrusts)
                drone.step_simulation(0.1)  # Simulate a step

                reward = self.reward(drone)  # Calculate reward
                next_state = self.discretize_state(drone)

                # Update Q-values
                self.update_q_vals(state, action_index, reward, next_state)
                cumulative_reward += reward

                # Debugging information
                # print(f"Step {step+1}: State: {state}, Action: {self.actions[action_index]}, Reward: {reward}, Next State: {next_state}")
                # print(f"Q-values: {self.q_values[state]}")

                # Stop if target is reached
                if drone.has_reached_target_last_update:
                    print(f"Target reached in {step+1} steps")
                    break

            # Epsilon decay
            self.epsilon = max(0.1, self.epsilon * 0.99)  # Decay epsilon, minimum value of 0.1
            print(f"Epoch {epoch+1} finished with cumulative reward: {cumulative_reward}, epsilon: {self.epsilon}")

    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        """Select the best action based on Q-values using ε-greedy strategy."""
        state = self.discretize_state(drone)

        # Initialize Q-values for unseen states
        if state not in self.q_values:
            self.q_values[state] = np.zeros(len(self.actions))

        # ε-greedy strategy with tie-breaking
        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(len(self.actions))  # Explore: Random action
        else:
            best_actions = np.flatnonzero(self.q_values[state] == np.max(self.q_values[state]))
            action_index = np.random.choice(best_actions)  # Exploit: Random tie-breaking

        # Debugging output
        # print(f"Choosing action {self.actions[action_index]} for state {state}")
        return self.actions[action_index]

    def save(self):
        """Save Q-values to a file."""
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(self.q_values, f)
        print("Q-table saved to q_table.pkl")

    def load(self):
        """Load Q-values from a file."""
        try:
            with open('q_table.pkl', 'rb') as f:
                self.q_values = pickle.load(f)
            print("Q-table loaded from q_table.pkl")
        except FileNotFoundError:
            print("No saved Q-table found. Starting fresh.")
            self.q_values = {}

