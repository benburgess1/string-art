from PIL import Image as PIL_Image
import numpy as np
import matplotlib.pyplot as plt

def import_image(filename):
    # Import and convert to grayscale
    img = Image.open(filename).convert('L')
    return img


class Pixel:
    # Class for the Pixel object, which gives the greyscale intensity at each
    # point in the image
    def __init__(self, r, z=0.):
        self.r = r
        self.x = self.r[0]
        self.y = self.r[1]
        self.z = z
        self.N_connections = 0

    def update_z(self):
        # Calculate pixel darkness, as a function of N_connections
        self.z = 1 / (0.3*self.N_connections + 1)


class Image:
    # Image class - essentially a list of pixels, their positions, and their greyscale intensities
    def __init__(self, img):
        self.img = img
        self.pixels = []
        self.circular_truncate()
        
    def circular_truncate(self):
        if self.img.shape[0] % 2 == 0:
            self.img = self.img[:-1,:]
        if self.img.shape[1] % 2 == 0:
            self.img = self.img[:,:-1]
        rad = (min(*self.img.shape) - 1) / 2
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                if (i - rad)**2 + (j-rad)**2 <= rad**2:
                    self.pixels.append(Pixel(r=np.array([(j-rad)/rad,(rad-i)/rad]), z=self.img[i,j]))






if __name__ == '__main__': 
    img = PIL_Image.open('Borat.png').convert('L')
    bitmap = np.array(img)

    img_small = img.resize((128,128), PIL_Image.BICUBIC)
    bitmap_small = np.array(img_small)

    print(bitmap.shape)
    print(bitmap.size)
    print(bitmap_small.shape)
    print(bitmap_small.size)

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(img, cmap='gray')
    axs[1].imshow(bitmap_small, cmap='gray')
    axs[0].set_title('Original')
    axs[1].set_title('Downsized')
    # for ax in axs:
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    plt.show()