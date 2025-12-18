import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Process_Image import Image
import pickle

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
            self.connections.remove(other)
            other.connections.remove(self)


class Board:
    # Board class: a set of pins and pixels, and methods for updating connections
    # until an acceptable accuracy to a given image is reached 
    def __init__(self, N_pins, image, connection_paths=None, progress=False, sigma=5, Nbar=4,
                 s1=3, s2=15, c=0.5):
        self.N_pins = N_pins
        self.sigma = sigma
        self.s1 = s1
        self.s2 = s2
        self.c = c
        self.Nbar = Nbar
        self.progress = progress
        self.pins = []
        thetas = np.linspace(0, 2*np.pi, N_pins+1)[:-1]
        for theta in thetas:
            self.pins.append(Pin(theta))
        self.connections = []
        self.image = image
        self.pixel_intersections = np.zeros_like(self.image.x)
        self.pixel_x = np.copy(self.image.x)
        self.pixel_y = np.copy(self.image.y)
        self.dx = np.copy(self.image.dx)
        self.dy = np.copy(self.image.dy)
        self.Nx = self.pixel_x.shape[0]
        self.Ny = self.pixel_y.shape[1]
        self.z = 255 * np.ones_like(self.pixel_x)
        if connection_paths is not None:
            self.connection_paths = connection_paths
        else:
            self.connection_paths = self.calc_all_connection_paths()
        self.cost = self.calc_cost()
        self.N_strings = np.zeros(self.image.x.shape)
        self.stuck_counter = 0
        self.critical_stuck = 20

    def add_connection(self, i, j):
        # Add a connection between pin i and pin j, and update the relevant pixels information
        # for number of intersecting strings (N_strings) and intensity (z)
        if self.pins[i] not in self.pins[j].connections:
            self.pins[i].connect(self.pins[j])
            pixel_idxs = self.connection_paths[i][j]
            rows, cols = zip(*pixel_idxs)
            self.N_strings[rows, cols] += 1
            self.calc_z(pixel_idxs)

    def remove_connection(self, i, j):
        # Add a connection between pin i and pin j, and update the relevant pixels information
        # for number of intersecting strings (N_strings) and intensity (z)
        if self.pins[i] in self.pins[j].connections:
            self.pins[i].deconnect(self.pins[j])
            pixel_idxs = self.connection_paths[i][j]
            rows, cols = zip(*pixel_idxs)
            self.N_strings[rows, cols] -= 1
            self.calc_z(pixel_idxs)

    def calc_z(self, idxs=None):
        # Calculate intensity z of a pixel from the number N_strings of strings intersecting it
        if idxs is not None:
            rows, cols = zip(*idxs)
            # self.z[rows, cols] = 255 * (1 - np.tanh((self.N_strings[rows, cols] - self.Nbar)/self.sigma)) / (1 - np.tanh(-self.Nbar/self.sigma))
            # self.z[rows, cols] = 255 /(self.N_strings[rows, cols]/self.sigma + 1)
            # self.z[rows, cols] = 255 * np.exp(-self.N_strings[rows, cols]/self.sigma)
            # self.z[rows, cols] = 255 * np.exp(-self.N_strings[rows, cols]**2 / self.sigma**2)
            self.z[rows, cols] = 255 * (self.c * np.exp(-self.N_strings[rows, cols]**2 / self.s1**2) + 
                                        (1 - self.c) * np.exp(-self.N_strings[rows, cols]**2 / self.s2**2))
        else:
            # self.z = 255 * (1 - np.tanh((self.N_strings - self.Nbar)/self.sigma)) / (1 - np.tanh(-self.Nbar/self.sigma))
            # self.z = 255 /(self.N_strings/self.sigma + 1)
            # self.z = 255 * np.exp(-self.N_strings/self.sigma)
            # self.z = 255 * np.exp(-self.N_strings**2 / self.sigma**2)
            self.z = 255 * (self.c * np.exp(-self.N_strings[rows, cols]**2 / self.s1**2) + 
                            (1 - self.c) * np.exp(-self.N_strings[rows, cols]**2 / self.s2**2))

    def calc_cost(self, idxs=None):
        # Calculate the current total cost function, as the square error between the current
        # pixel z_i values, and the target image z_i values
        if idxs is not None:
            rows, cols = zip(*idxs)
            return np.sum((self.z[rows, cols] - self.image.bitmap[rows, cols])**2)
        else:
            self.cost = np.sum((self.z - self.image.bitmap)**2)
            return self.cost

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
                    print(f'Evaluating pair {count} out of {int(0.5*self.N_pins*(self.N_pins-1))}', end='\r')
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
        i = np.random.randint(0, self.N_pins)
        j = np.random.randint(0, self.N_pins)
        while j == i:
            j = np.random.randint(0, self.N_pins)
        # print(f'Evaluating pair {i}, {j}')
        pixel_idxs = self.connection_paths[i][j]
        if len(pixel_idxs) == 0:
            return
        current_cost = self.calc_cost(pixel_idxs)
        # print(f'Current cost: {current_cost}')
        if self.pins[i] not in self.pins[j].connections:
            # print('Not currently connected')
            self.add_connection(i, j)
            # print(self.pins[i] in self.pins[j].connections)
            pixel_idxs = self.connection_paths[i][j]
            (l,k) = pixel_idxs[0]
            # print(self.N_strings[l,k])
            # print(self.z[l,k])
            # print(self.image.bitmap[l,k])
            new_cost = self.calc_cost(pixel_idxs)
            # print(f'New cost: {current_cost}')
            if new_cost > current_cost: 
                # Revert change
                self.remove_connection(i, j)
            # else:
            #     print('Added connection')
        elif self.pins[i] in self.pins[j].connections:
            # print('Already connected')
            self.remove_connection(i, j)
            new_cost = self.calc_cost(pixel_idxs)
            if new_cost > current_cost:
                # Revert change; look for new pin to connect to
                self.add_connection(i, j)
            # else:
            #     print('Removed connection')


    def optimise(self, N_steps=None, record_cost=False):
        # Optimise connections by repeatedly making random steps
        if N_steps is None:
            N_steps = self.N_pins**2
        cost_list = np.zeros(N_steps)
        for i in range(N_steps):
            print(f'Evaluating step {i+1} out of {N_steps}...' + 10*' ', end='\r')
            self.random_step()
            if record_cost:
                cost_list[i] = self.calc_cost()
        print('\nDone')
        return cost_list

    def show_state(self, ax=None, plot=True, color='k', ms=3, lw=1,
                   mark_pixels=False, highlight_pixels=None, params_in_title=True):
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

        if params_in_title:
            title_str = ''
            title_str += r'$N_{pin} = $' + str(int(self.N_pins))
            title_str += ', ' + r'$N_{pix} = $' + str(int(self.Nx))
            # title_str += ', ' + r'$\sigma = $' + str(np.round(self.sigma,3))
            title_str += ', ' + r'$s_1 = $' + str(np.round(self.s1,3))
            title_str += ', ' + r'$s_2 = $' + str(np.round(self.s2,3))
            title_str += ', ' + r'$c = $' + str(np.round(self.c,3))
            ax.set_title(title_str)

        if plot == True:
            plt.show()

    def show_z(self, params_in_title=True):
        fig, ax = plt.subplots()
        ax.imshow(self.z, cmap='gray')
        rad = (self.Nx - 1)/2
        circ = patches.Circle(xy=(rad, rad), radius=rad, ec='r', fc=(0,0,0,0), lw=3)
        ax.add_patch(circ)
        if params_in_title:
            title_str = ''
            title_str += r'$N_{pin} = $' + str(int(self.N_pins))
            title_str += ', ' + r'$N_{pix} = $' + str(int(self.Nx))
            # title_str += ', ' + r'$\sigma = $' + str(np.round(self.sigma,3))
            title_str += ', ' + r'$s_1 = $' + str(np.round(self.s1,3))
            title_str += ', ' + r'$s_2 = $' + str(np.round(self.s2,3))
            title_str += ', ' + r'$c = $' + str(np.round(self.c,3))
            ax.set_title(title_str)
        plt.show()

    def calc_string_lenth(self):
        length = 0
        for pin in self.pins:
            for other in pin.connections:
                dr = other.r - pin.r
                length += np.linalg.norm(dr)
        length /= 2
        return length
    
    def calc_num_connections(self):
        total = 0
        for pin in self.pins:
            total += len(pin.connections)
        total /= 2
        return total



def plot_cost_record(cost_record, logx=False, logy=False):
    fig, ax = plt.subplots()
    x = np.arange(1, cost_record.size+1)
    ax.plot(x, cost_record, color='b', ls='-', marker=None)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cost')
    ax.set_title('Cost during optimisation procedure')
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    plt.show()


def generate_connection_paths(filename=None, L_pixels=101, N_pins=100):
    # Construct a board and save it's calculated connection_paths array
    image = Image('Einstein.png', downsize_pixels=2*L_pixels, L_pixels=L_pixels, y_offset=0, x_offset=0)     # 'Dummy' image
    board = Board(N_pins=N_pins, image=image, progress=True, connection_paths=None,
                  sigma=12.5)
    connection_paths = board.connection_paths
    if filename is None:
        filename = 'Connection_Paths_Npix' + str(L_pixels) + '_Npin' + str(N_pins) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(connection_paths, f)

    
if __name__ == '__main__':
    # generate_connection_paths(L_pixels=151, N_pins=250)
    # generate_connection_paths(L_pixels=151, N_pins=300)
    with open('Connection_Paths_Npix151_Npin250.pkl', 'rb') as f:
        # print('Loading connection_paths...')
        connection_paths = pickle.load(f)
        # print('Done')
    # image = Image('Borat.png', downsize_pixels=111, L_pixels=101, y_offset=-2, x_offset=5)
    # image = Image('Einstein.png', downsize_pixels=111, L_pixels=101, y_offset=15, x_offset=5)
    # image = Image('Einstein.png', downsize_pixels=161, L_pixels=151, y_offset=25, x_offset=0)
    # image = Image('selfie.jpg', downsize_pixels=221, L_pixels=151, y_offset=-10, x_offset=0)
    image = Image('selfie2.jpg', downsize_pixels=251, L_pixels=151, y_offset=-40, x_offset=5)
    # # board = Board(N_pins=100, image=image, progress=True, connection_paths=connection_paths,
    # #               sigma=12.5, Nbar=7)
    board = Board(N_pins=250, image=image, progress=True, connection_paths=connection_paths,
                  sigma=12.5, s1=8, s2=30, c=0.35)

    cost_record = board.optimise(N_steps=100000, record_cost=True)
    length = board.calc_string_lenth()
    num_connections = board.calc_num_connections()
    print(f'Length = {np.round(length * 0.25, 3)}')
    print(f'Number of connections = {np.round(num_connections, 3)}')
    board.show_state(mark_pixels=False, lw=0.025, params_in_title=True)
    board.show_z(params_in_title=True)
    plot_cost_record(cost_record, logx=True)