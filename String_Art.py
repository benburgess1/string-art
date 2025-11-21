import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Process_Image import Image

class Pin:
    # Class for the Pin object, between which connections can be created to
    # form the representation of the image
    def __init__(self, theta):
        self.theta = theta
        self.r = np.array([np.cos(self.theta), np.sin(self.theta)])
        self.x = self.r[0]
        self.y = self.r[1]
        self.connections = []

    def connect(self, other):
        # Add connection between self and other
        if other not in self.connections:
            self.connections.append(other)
            other.connections.append(self)
    
    def deconnect(self, other):
        # Remove connection between self and other
        if other in self.connections:
            self.connections.pop(other)
            other.connections.pop(self)


class Board:
    # Board class: a set of pins and pixels, and methods for updating connections
    # until an acceptable accuracy to a given image is reached 
    def __init__(self, N_pins, image):
        self.N_pins = N_pins
        self.pins = []
        thetas = np.linspace(0, 2*np.pi, N_pins+1)[:-1]
        for theta in thetas:
            self.pins.append(Pin(theta))
        self.connections = []
        self.connection_paths = self.calc_connection_paths()
        self.image = image
        self.pixel_x = np.copy(self.image.x)
        self.pixel_y = np.copy(self.image.y)
        self.z = np.ones_like(self.pixel_x)
        self.cost = self.calc_cost()

    def calc_cost(self):
        # Calculate the current total cost function, as the square error between the current
        # pixel z_i values, and the target image z_i values
        return np.sum(self.z - self.image.bitmap)**2

    def calc_connection_paths(self):
        # Create dictionary of (ij):[k1, k2, ...] listing the pixels y_k intersected
        # by the connection between pins i and j
        pass

    def random_step(self):
        # Pick a random connection; flip it; test whether cost is improved or not; 
        # if improved, keep the change
        pass

    def optimise(self, N_steps):
        # Optimise connections by repeatedly making random steps
        for i in range(N_steps):
            print(f'Evaluating step {i+1} out of {N_steps}...' + 10*' ', end='\r')
            self.random_step()

    def show_state(self, ax=None, plot=True, color='k', ms=3, lw=1,
                   mark_pixels=False):
        # Plot all pins and connections; optionally plot the positions of pixels (for testing)
        if ax is None:
            fig, ax = plt.subplots() 

        circ = patches.Circle(xy=(0,0), radius=1, ec=color, fc=(0,0,0,0), lw=lw)
        ax.add_patch(circ)

        for pin in self.pins:
            if len(pin.connections) > 0:
                for other_pin in pin.connections:
                    ax.plot([pin.x, other_pin.x], [pin.y, other_pin.y], 
                            color=color, marker='o', ms=ms, lw=lw)
            else:
                ax.plot(*pin.r, color=color, marker='o', ms=ms, lw=lw)

        if mark_pixels:
            ax.plot(self.pixel_x.flatten(), self.pixel_y.flatten(), 
                    color='r', ms=2, marker='x', ls='')
                
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        
        if plot == True:
            plt.show()


    
if __name__ == '__main__':
    image = Image('Borat.png', downsize_pixels=111, L_pixels=101, y_offset=-2, x_offset=5)
    board = Board(N_pins=100, image=image)
    print(len(board.pins))
    pin1 = board.pins[0]
    print(pin1.r)
    pin1.connect(board.pins[60])
    board.show_state(mark_pixels=True)