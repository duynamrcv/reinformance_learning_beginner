import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from env import final_states

# Class for Q-learning table
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        # Create full Q table for all cells
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        # Create Q table for cells in final route
        self.q_table_final = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # Choose the action for agent
    def choose_action(self, observation):
        # Check if the state exists in the table
        self.check_state_exist(observation)
        # Choose the best action
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # Choose the random action
            action = np.random.choice(self.actions)
        return action
    
    # Learning and update Q-table
    def learn(self, state, action, reward, next_state):
        # Check if the next step exists in the Q-table
        self.check_state_exist(next_state)
        
        # Current state in the current position 
        q_predict = self.q_table.loc[state, action]

        # Check if next state is free or obstacle or goal
        if next_state != 'goal' or next_state != 'obstacle':
            q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()
        else:
            q_target = reward

        # Update Q-table
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

        return self.q_table.loc[state, action]

    # Add state to Q-table
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state)
            )

    
    # Print Q-table with states
    def print_q_table(self):
        # Get the coordinates of final route in env.py()
        e = final_states()

        # Compare indexes with coordinates and write in the new Q-table value
        for i in range(len(e)):
            state = str(e[i])
            for j in range(len(self.q_table.index)):
                if self.q_table.index[j] == state:
                    self.q_table_final.loc[state, :] = self.q_table.loc[state, :]

        print()
        print('Length of final Q-table =', len(self.q_table_final.index))
        print('Final Q-table with values from the final route:')
        print(self.q_table_final)

        print()
        print('Length of full Q-table =', len(self.q_table.index))
        print('Full Q-table:')
        print(self.q_table)
    
    # Plotting the results
    def plot_results(self, steps, cost):
        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        
        ax1.plot(np.arange(len(steps)), steps, 'b')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title('Episode via steps')

        ax2.plot(np.arange(len(cost)), cost, 'r')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost')
        ax2.set_title('Episode via cost')

        plt.tight_layout()  # Function to make distance between figures

        plt.figure()
        plt.plot(np.arange(len(steps)), steps, 'b')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        plt.figure()
        plt.plot(np.arange(len(cost)), cost, 'r')
        plt.title('Episode via cost')
        plt.xlabel('Episode')
        plt.ylabel('Cost')

        # Showing the plots
        plt.show()
