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
    def __init__(self, N_pins, image, connection_paths=None, progress=False):
        self.N_pins = N_pins
        self.pins = []
        thetas = np.linspace(0, 2*np.pi, N_pins+1)[:-1]
        for theta in thetas:
            self.pins.append(Pin(theta))
        self.connections = []
        self.image = image
        self.pixel_x = np.copy(self.image.x)
        self.pixel_y = np.copy(self.image.y)
        self.dx = np.copy(self.image.dx)
        self.dy = np.copy(self.image.dy)
        self.Nx = self.pixel_x.shape[0]
        self.Ny = self.pixel_y.shape[1]
        self.z = np.ones_like(self.pixel_x)
        if connection_paths is not None:
            self.connection_paths = connection_paths
        else:
            self.connection_paths = self.calc_all_connection_paths()
        self.cost = self.calc_cost()
        self.progress = progress

    def calc_cost(self):
        # Calculate the current total cost function, as the square error between the current
        # pixel z_i values, and the target image z_i values
        return np.sum(self.z - self.image.bitmap)**2

    def calc_all_connection_paths(self):
        # Create nested list connection_paths
        # connection_paths[i][j] = [(k1,l1), (k2,l2), ...] is an array of tuples of
        # indices of the pixels (l,k) intersected by the connection between pins i and j
        connection_paths = [[[] for _ in range(self.N_pins)] for _ in range(self.N_pins)]
        count = 1
        if self.progress:
            print('Evaluating Connection Paths')
        for i in range(self.N_pins):
            for j in range(i+1, self.N_pins):
                if self.progress:
                    print(f'Evaluating pair {count} out of {0.5*self.N_pins*(self.N_pins-1)}', end='\r')
                path_pixels = self.calc_connection_path(i, j)
                connection_paths[i][j] = path_pixels
                connection_paths[j][i] = path_pixels
                count += 1
        print('')
        return connection_paths


    def find_closest_pixel(self, r):
        # Find the pixel that a point r is contained within, assuming an even grid of pixels
        # If the closest pixel has its centre outside the circular window, return None
        mask = (
            (np.abs(self.pixel_x - r[0])<self.dx/2) &
            (np.abs(self.pixel_y - r[1])<self.dy/2) &
            self.image.inside_mask
        )
        idxs = np.where(mask)
        if len(idxs[0]) == 0:
            return None
        elif len(idxs[0]) == 1:
            return (idxs[0][0], idxs[1][0])
        else:
            print('Warning: multiple acceptance pixels. Check pixel grid.')


    def calc_connection_path(self, i, j):
        # Calculate the array of tuples of indices (l,k) of pixels intersected by
        # the path connecting pins i and j
        path_pixels = []
        r_i = self.pins[i].r
        r_j = self.pins[j].r
        dr = np.linalg.norm(r_j - r_i)
        ts = np.linspace(0, 1, self.Nx)
        for t in ts:
            r = r_i * (1-t) + r_j * t
            idx = self.find_closest_pixel(r)
            if idx is not None:
                if idx not in path_pixels:
                    path_pixels.append(idx)
        return path_pixels
    

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
                   mark_pixels=False, highlight_pixels=None):
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
            mask = self.image.inside_mask
            print(self.pixel_x[mask].size)
            ax.plot(self.pixel_x[mask].flatten(), self.pixel_y[mask].flatten(), 
                    color='r', ms=2, marker='x', ls='')
            
        if highlight_pixels is not None:
            for idx in highlight_pixels:
                ax.plot(self.pixel_x[*idx], self.pixel_y[*idx], 
                        color='gold', ms=2, marker='x', ls='')
                
                
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        
        if plot == True:
            plt.show()


    
if __name__ == '__main__':
    image = Image('Borat.png', downsize_pixels=111, L_pixels=101, y_offset=-2, x_offset=5)
    board = Board(N_pins=100, image=image, progress=True)
    # print(len(board.pins))
    # pin1 = board.pins[0]
    # print(pin1.r)
    # pin1.connect(board.pins[60])
    pin25 = board.pins[24]
    pin56 = board.pins[55]
    pin25.connect(pin56)
    board.show_state(mark_pixels=True, highlight_pixels=board.connection_paths[24][55])