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
                 image = True,
                 augment = False):
        self.img = img
        self.lab = lab
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.step = step
        self.num_classes = num_classes
        self.image = image
        self.augment = augment

    def __getitem__(self, item):
        return self.patch_image()

    def patch_image(self):
        def normalize_image(input_block):
            block = input_block.copy()
            vol_max, vol_min = block.max(), block.min()
            if not vol_max == vol_min:  # run a check. otherwise error when divide by 0
                for i in range(block.shape[-1]):
                    block[:, :, i] = (block[:, :, i] - vol_min) / (vol_max - vol_min)
            return block
        max_x_pixel = 0
        #max_y_pixel = self.img.shape

        i = 0
        images = np.zeros((self.step, self.step, self.num_classes), dtype="float32")
        masks = np.zeros((self.step, self.step), dtype=bool)
        while True:
            # Get random integer from 0 to the image
            if self.img.shape[0] > self.img.shape[1]:
                max_x_pixel = self.img.shape[1]
            else:
                max_x_pixel = self.img.shape[0]
            rand_int = random.randrange(0, max_x_pixel)
            if max_x_pixel <= self.step:
                rand_int = 0
            elif rand_int + self.step > max_x_pixel:
               rand_int = random.randrange(0, max_x_pixel-self.step)
            else:
                rand_int = 0

            img_crop = self.img[rand_int:rand_int+self.step,rand_int:rand_int+self.step]
            lab_crop = self.lab[rand_int:rand_int+self.step,rand_int:rand_int+self.step]

            if lab_crop.max() == 0:
                continue
            # print(rand_int)
            # lab_crop = np.array(lab_crop, dtype=bool)
            #img_crop = np.array(img_crop, dtype="float32")
            #
            # fig = plt.figure(figsize=(10, 7))
            #
            # fig.add_subplot(1, 2, 1)
            # plt.imshow(img_crop)
            #
            # fig.add_subplot(1, 2, 2)
            # plt.imshow(lab_crop)
            #
            # plt.show()
            if self.augment:
                augment = Augmentor(img_crop, lab_crop)
                img_crop, lab_crop = augment.rotate()
            img_crop = normalize_image(img_crop.reshape(*self.input_shape))
            # plt.imshow(img_crop)
            # plt.title("patcher")
            # plt.show()
            images = img_crop
            masks = lab_crop.reshape((self.input_shape[0], self.input_shape[1]))
            #i += 1
            break
        # plt.imshow(images)
        # plt.title("patcher1")
        # plt.show()
        return images, masks

