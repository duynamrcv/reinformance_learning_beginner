from env import Environment
from agent_q_learning import QLearningTable

def update():
    steps = []
    all_costs = []

    for epsion in range(1000):
        observation = env.reset()
        i = 0; cost = 0
        while True:
            # Refresh the environment
            env.render()

            # RL chooses action 
            action = RL.choose_action(str(observation))
            
            # RL take actions and get the next observation and reward
            observation_, reward, done = env.step(action)

            # RL learns from this transition and calculate the cost
            cost += RL.learn(str(observation), action, reward, str(observation_))

            observation = observation_
            i += 1

            if done:
                steps += [i]
                all_costs += [cost]
                break
    
    # Showing the final route
    env.final()

    # Showing the Q-table with values for each action
    RL.print_q_table()

    # Plotting the results
    RL.plot_results(steps, all_costs)

if __name__ == "__main__":
    env = Environment()
    # Main algorithm
    RL = QLearningTable(actions=list(range(env.n_actions)))
    # Update
    env.after(100, update)
    env.mainloop()