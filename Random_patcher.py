import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from Augmentor import Augmentor

class Random_patcher:
    def __init__(self,
                 img,
                 lab,
                 batch_size = 1,
                 input_shape = (512, 512, 1),
                 step = 512,
                 num_classes = 1,
                 image = True):
        self.img = img
        self.lab = lab
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.step = step
        self.num_classes = num_classes
        self.image = image

    def __getitem__(self, item):
        return self.patch_image()

    def patch_image(self):
        max_x_pixel = self.img.shape
        #max_y_pixel = self.img.shape

        i = 0
        images = np.zeros((self.batch_size, self.step, self.step, self.num_classes))
        masks = np.zeros((self.batch_size, self.step, self.step, self.num_classes), dtype=bool)
        while i < self.batch_size:
            # Get random integer from 0 to the image size
            rand_int = random.randrange(0, max_x_pixel[0])
            if max_x_pixel[0] == self.step:
                rand_int = 0
            elif rand_int + self.step > max_x_pixel[0]:
                rand_int = random.randrange(0, max_x_pixel[0]-self.step)



            img_crop = self.img[rand_int:rand_int+self.step,rand_int:rand_int+self.step]
            lab_crop = self.lab[rand_int:rand_int+self.step,rand_int:rand_int+self.step]
            lab_crop = np.array(lab_crop, dtype=bool)

            augment = Augmentor(img_crop, lab_crop)
            img_crop, lab_crop = augment.rotate()

            images[i] = img_crop.reshape(*self.input_shape)
            masks[i] = lab_crop.reshape(*self.input_shape)
            i += 1

        return images, masks

