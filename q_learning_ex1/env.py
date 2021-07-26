# Import library
from tkinter.constants import ANCHOR
import numpy as np
import tkinter as tk
import time
from PIL import ImageTk, Image

# Setting the sizes for environment
pixels = 40
width = 9
height = 9

# Final route
a = {}

class Environment(tk.Tk, object):
    def __init__(self):
        super(Environment, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.title("RL Q-learning Ex1")
        self.geometry('{}x{}'.format(width*pixels, height*pixels))
        self.build_environment()

        # Draw the final route
        self.d = {}
        self.f = {}

        # Key
        self.i = 0

        # Writing the final at first time
        self.c = True

        # Show the steps for longest route
        self.longest = 0

        # Show the steps for shortest route
        self.shortest = 0

    # Build environment
    def build_environment(self):
        self.canvas_widget = tk.Canvas(self, bg='white',
                                        width = width*pixels,
                                        height = height*pixels)
        # Create grid lines
        for col in range(0, width*pixels, pixels):
            x0, y0, x1, y1 = col, 0, col, width*pixels
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='grey')
        for row in range(0, height*pixels, pixels):
            x0, y0, x1, y1 = 0, row, height*pixels, row
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='grey')

        # Create obstacles
        obstacle_img = Image.open("images/road_closed.png")
        self.obstacle_obj = ImageTk.PhotoImage(obstacle_img)

        self.obstacle1 = self.canvas_widget.create_image(pixels*0, pixels*2,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle2 = self.canvas_widget.create_image(pixels*0, pixels*6,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle3 = self.canvas_widget.create_image(pixels*1, pixels*2,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle4 = self.canvas_widget.create_image(pixels*1, pixels*4,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle5 = self.canvas_widget.create_image(pixels*1, pixels*8,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle6 = self.canvas_widget.create_image(pixels*2, pixels*0,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle7 = self.canvas_widget.create_image(pixels*3, pixels*2,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle8 = self.canvas_widget.create_image(pixels*3, pixels*3,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle9 = self.canvas_widget.create_image(pixels*3, pixels*4,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle10 = self.canvas_widget.create_image(pixels*3, pixels*6,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle11 = self.canvas_widget.create_image(pixels*4, pixels*2,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle12 = self.canvas_widget.create_image(pixels*4, pixels*8,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle13 = self.canvas_widget.create_image(pixels*5, pixels*1,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle14 = self.canvas_widget.create_image(pixels*6, pixels*6,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle15 = self.canvas_widget.create_image(pixels*6, pixels*7,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle16 = self.canvas_widget.create_image(pixels*6, pixels*8,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle17 = self.canvas_widget.create_image(pixels*7, pixels*3,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle18 = self.canvas_widget.create_image(pixels*7, pixels*5,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle19 = self.canvas_widget.create_image(pixels*8, pixels*0,
                                                        anchor='nw', image=self.obstacle_obj)
        self.obstacle20 = self.canvas_widget.create_image(pixels*7, pixels*2,
                                                        anchor='nw', image=self.obstacle_obj)
        
        # Final point
        goal_img = Image.open("images/flag.png")
        self.goal_obj = ImageTk.PhotoImage(goal_img)
        self.goal = self.canvas_widget.create_image(pixels*7, pixels*7,
                                                    anchor="nw", image=self.goal_obj)

        # agent
        agent_img = Image.open("images/agent.png")
        self.agent_obj = ImageTk.PhotoImage(agent_img)
        self.agent = self.canvas_widget.create_image(0, 0, anchor="nw", image=self.agent_obj)

        # Packing all together
        self.canvas_widget.pack()
    
    # Reset environment and start new Episode
    def reset(self):
        self.update()

        # Update agent
        self.canvas_widget.delete(self.agent)
        self.agent = self.canvas_widget.create_image(0, 0, anchor="nw", image=self.agent_obj)

        # Clear dic and i
        self.d = {}
        self.i = 0

        # Return observation
        return self.canvas_widget.coords(self.agent)
    
    # Get the next observation and reward by doing next step
    def step(self, action):
        # Current state of agent
        state = self.canvas_widget.coords(self.agent)
        base_action = np.array([0,0])

        # Update next state
        if action == 0:     # up
            if state[1] >= pixels:
                base_action[1] -= pixels
        elif action == 1:   # down
            if state[1] < (height - 1)*pixels:
                base_action[1] += pixels
        elif action == 2:   # right
            if state[0] < (width - 1)*pixels:
                base_action[0] += pixels
        elif action == 3:   # left
            if state[0] >= pixels:
                base_action[0] -= pixels
        
        # Moving agent
        self.canvas_widget.move(self.agent, base_action[0], base_action[1])

        # Write to the dictionary
        self.d[self.i] = self.canvas_widget.coords(self.agent)

        # Udpate next state
        next_state = self.d[self.i]
        self.i += 1

        # Calculating the rewward
        if next_state == self.canvas_widget.coords(self.goal):
            reward = 1
            done = True
            next_state = 'goal'

            # Fill the dictionary first time
            if self.c == True:
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]
                self.c = False
                self.longest = len(self.d)
                self.shortest = len(self.d)
            
            # Check if the currentlt found route is shorter
            if len(self.d) < len(self.f):
                # Save the number of steps for the shortest route
                self.shortest = len(self.d)

                # Clear the dictionary for final route
                self.f = {}

                # Reassigning
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]

            # Save the number of steps for the longest route
            if len(self.d) > self.longest:
                self.longest = len(self.d)

        elif next_state in [self.canvas_widget.coords(self.obstacle1),
                            self.canvas_widget.coords(self.obstacle2),
                            self.canvas_widget.coords(self.obstacle3),
                            self.canvas_widget.coords(self.obstacle4),
                            self.canvas_widget.coords(self.obstacle5),
                            self.canvas_widget.coords(self.obstacle6),
                            self.canvas_widget.coords(self.obstacle7),
                            self.canvas_widget.coords(self.obstacle8),
                            self.canvas_widget.coords(self.obstacle9),
                            self.canvas_widget.coords(self.obstacle10),
                            self.canvas_widget.coords(self.obstacle11),
                            self.canvas_widget.coords(self.obstacle12),
                            self.canvas_widget.coords(self.obstacle13),
                            self.canvas_widget.coords(self.obstacle14),
                            self.canvas_widget.coords(self.obstacle15),
                            self.canvas_widget.coords(self.obstacle16),
                            self.canvas_widget.coords(self.obstacle17),
                            self.canvas_widget.coords(self.obstacle18),
                            self.canvas_widget.coords(self.obstacle19),
                            self.canvas_widget.coords(self.obstacle20)]:
            reward = -1
            done = True
            next_state = 'obstacle'

            # Clear the dictionary
            self.d = {}
            self.i = 0
        
        else:
            reward = 0
            done = False
        
        return next_state, reward, done

    # Refresh environment
    def render(self):
        self.update()

    # Show the route
    def final(self):
        # Delete the agent at the end
        self.canvas_widget.delete(self.agent)

        # Show number of steps
        print("Shortest route: {} steps".format(self.shortest))
        print("Longest route: {} steps".format(self.longest))

        # Create the initial point
        origin = np.array([20, 20])
        self.initial_point = self.canvas_widget.create_oval(
            origin[0] - 5, origin[1] - 5,
            origin[0] + 5, origin[1] + 5,
            fill='blue', outline='black'
        )

        # Fill the route
        for j in range(len(self.f)):
            # Show the coordinates of the final route
            print(self.f[j])
            self.track = self.canvas_widget.create_oval(
                self.f[j][0] + origin[0] - 5, self.f[j][1] + origin[1] - 5,
                self.f[j][0] + origin[0] + 5, self.f[j][1] + origin[1] + 5,
                fill='blue', outline='black'
            )
            # Write the final route
            a[j] = self.f[j]

def final_states():
    return a

# if __name__== "__main__":
#     env = Environment()
