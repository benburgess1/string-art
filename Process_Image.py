from PIL import Image as PIL_Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# class Pixel:
#     # Class for the Pixel object, which gives the greyscale intensity at each
#     # point in the image
#     def __init__(self, r, z=0.):
#         self.r = r
#         self.x = self.r[0]
#         self.y = self.r[1]
#         self.z = z
#         self.N_connections = 0

#     def update_z(self):
#         # Calculate pixel darkness, as a function of N_connections
#         self.z = 1 / (0.3*self.N_connections + 1)


class Image:
    # Image class - essentially a list of pixels, their positions, and their greyscale intensities
    # Loads a generic image using the PIL library
    # Then downsizes into a square array of N x N
    # Marks pixels according to inside vs outside, but keeps the image as a square for 
    # convenience with manipulating square arrays later
    # Don't need a Pixel class?
    def __init__(self, filename, downsize_pixels=101, L_pixels=51, x_offset=0, y_offset=0):
        self.downsize_pixels = downsize_pixels
        self.L_pixels = L_pixels
        self.rad = (L_pixels - 1) / 2
        self.x_offset = x_offset
        self.y_offset = y_offset
        if np.abs(self.x_offset) > (self.downsize_pixels - self.L_pixels)/2:
            print('Warning: large x_offset; truncation errors may occur')
        if np.abs(self.y_offset) > (self.downsize_pixels - self.L_pixels)/2:
            print('Warning: large y_offset; truncation errors may occur')
        self.raw_img = PIL_Image.open(filename).convert('L')
        self.raw_bitmap = np.array(self.raw_img)
        self.img = self.proportional_downsize()
        self.bitmap = np.array(self.img)
        self.square_truncate()
        self.x, self.y = self.calc_positions()
        self.dx = self.x[0,1] - self.x[0,0]
        self.dy = self.dx
        self.inside_mask = self.calc_inside_mask()

    def proportional_downsize(self):
        # Assumes portrait picture; will add functionality for landscape later
        [Ly, Lx] = self.raw_bitmap.shape
        return self.raw_img.resize((self.downsize_pixels,int(Ly*self.downsize_pixels/Lx)), PIL_Image.BICUBIC)

    def square_truncate(self):
        Ly = self.bitmap.shape[0]
        i_min = int(Ly/2 - self.L_pixels/2 - self.y_offset)
        Lx = self.bitmap.shape[1]
        j_min = int(Lx/2 - self.L_pixels/2 + self.x_offset)
        self.bitmap = self.bitmap[i_min:i_min+self.L_pixels, j_min:j_min+self.L_pixels]

    def calc_positions(self):
        x = (np.arange(self.L_pixels) - self.rad) / self.rad
        y = np.copy(x)
        xx, yy = np.meshgrid(x, y, indexing='xy')
        yy = yy[::-1,:]
        return xx, yy
    
    def calc_inside_mask(self):
        return self.x**2 + self.y**2 <= 1

    def preview(self):
        fig, ax = plt.subplots()
        ax.imshow(self.bitmap, cmap='gray')
        circ = patches.Circle(xy=(self.rad, self.rad), radius=self.rad, ec='r', fc=(0,0,0,0), lw=3)
        ax.add_patch(circ)
        plt.show()


if __name__ == '__main__': 
    # image = Image('Borat.png', downsize_pixels=111, L_pixels=101, y_offset=-2, x_offset=5)
    # image = Image('Einstein.png', downsize_pixels=161, L_pixels=151, y_offset=25, x_offset=0)
    # image = Image('selfie.jpg', downsize_pixels=221, L_pixels=151, y_offset=-10, x_offset=0)
    image = Image('selfie2.jpg', downsize_pixels=251, L_pixels=151, y_offset=-40, x_offset=5)
    image.preview()