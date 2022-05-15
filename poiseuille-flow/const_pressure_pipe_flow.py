import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import *
import utils

class Operators:
    def __init__(self, field, element_length) -> None:
        self.field = field    # velocity field (u)
        self.element_length = element_length 

    def central_diff_x_periodic(self):
        diff = ((
            np.roll(self.field, shift=1, axis=1)    # wrap-around, mod-N
            -
            np.roll(self.field, shift=-1, axis=1)
        ) / (2*self.element_length))
        
        return diff 
    
    def laplace_periodic(self):
        # 5-point stencil
        diff = ((
            np.roll(self.field, shift=1, axis=1) +
            np.roll(self.field, shift=1, axis=0) +
            np.roll(self.field, shift=-1, axis=1) +
            np.roll(self.field, shift=-1, axis=0) -
            4*self.field
        ) / (self.element_length)**2)

        return diff


def main():
    # Creating mesh
    element_length = 1.0/(N_POINTS-1)   # spatial discretization length dx
    x_range = np.linspace(0.0, 1.0, N_POINTS)
    y_range = np.linspace(0.0, 1.0, N_POINTS)

    x_coordinates, y_coordinates = np.meshgrid(x_range, y_range)

    # Define initial conditions 
    u_previous = np.ones((N_POINTS, N_POINTS))   # N points in x & y-directions
    
    # Define boundary conditions (no-slip) at the walls
    u_previous[0, :] = 0.0
    u_previous[-1, :] = 0.0

    # Start the time loop 
    for iter in tqdm(range(N_TIME_STEPS)):
        # Convection term (u ∂u/dx)
        diff_operator = Operators(u_previous, element_length)
        convection_x = u_previous*diff_operator.central_diff_x_periodic()

        # Diffusion term (ν ∇²u)
        diffusion_x = KINEMATIC_VISCOSITY*diff_operator.laplace_periodic()

        # Explicit Euler step to update velocity 
        u_next = u_previous + \
            TIME_STEP*(-PRESSURE_GRADIENT[0] + diffusion_x - convection_x)
        
        # Enforce wall B.C. on the updated velocity as well
        u_next[0, :] = 0.0
        u_next[-1, :] = 0.0
            
        # Step forward in time     
        u_previous = u_next

        utils.visualize(x_coordinates, y_coordinates, u_next)


if __name__ == '__main__':
    main()
