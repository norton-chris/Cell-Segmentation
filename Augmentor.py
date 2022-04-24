import random
from skimage.transform import rotate

class Augmentor:
    def __init__(self,
                 img,
                 lab,
                 mode = 'reflect'):
        self.img = img
        self.lab = lab
        self.mode = mode

    def rotate(self):
        deg = random.randint(-180, 180)
        img = rotate(self.img, deg, mode=self.mode)
        lab = rotate(self.lab, deg, mode=self.mode)
        return img, lab
