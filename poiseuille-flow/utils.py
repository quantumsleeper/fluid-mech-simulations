from turtle import color
import matplotlib.pyplot as plt
import numpy as np

# Plotting 
def visualize(x, y, velocity):
    plt.contourf(x, y, velocity, levels=50)
    plt.colorbar()
    plt.quiver(x, y, velocity, np.zeros_like(velocity))
    plt.xlabel("Position along the pipe axis")
    plt.ylabel("Position perpendicular to pipe axis")
    
    plt.twiny()
    plt.plot(velocity[:, 1], y[:, 1], color="white")
    plt.xlabel("Flow Velocity")

    plt.draw()    # To re-draw the current figure
    plt.pause(0.05)
    plt.clf()