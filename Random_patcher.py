#----------------------------------------------------------------------------
# Created By  : Chris Norton
# ---------------------------------------------------------------------------
"""
Randomly takes a patch from a full size image and mask and
returns an image and masks
"""
# ---------------------------------------------------------------------------

# 3rd Party Libs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# Owned
from Augmentor import Augmentor
__author__ = "Chris Norton"
__maintainer__ = "Chris Norton"
__email__ = "cnorton@mtu.edu"
__status__ = "Dev"

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
            # Ensure input_block is at least 3D
            if input_block.ndim == 2:
                input_block = input_block[:, :, np.newaxis]

            block = input_block.copy()
            for i in range(block.shape[-1]):
                channel = block[:, :, i]
                vol_max, vol_min = channel.max(), channel.min()
                if vol_max > vol_min:  # Avoid division by zero
                    block[:, :, i] = (channel - vol_min) / (vol_max - vol_min)
            return block
        max_x_pixel = 0
        #max_y_pixel = self.img.shape

        i = 0
        images = np.zeros((self.step, self.step, self.num_classes), dtype="float32")
        masks = np.zeros((self.step, self.step), dtype=bool)
        while True:
            # Determine the maximum x pixel for cropping
            max_x_pixel = min(self.img.shape[:2])

            # Generate a random starting point for the crop
            rand_int = random.randrange(0, max_x_pixel - self.step) if max_x_pixel > self.step else 0

            # Extract the crop from the image and label
            img_crop = self.img[rand_int:rand_int + self.step, rand_int:rand_int + self.step]
            lab_crop = self.lab[rand_int:rand_int + self.step, rand_int:rand_int + self.step]

            # Skip the crop if the label is all zeros
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
            # Apply augmentation if needed
            if self.augment:
                augment = Augmentor(img_crop, lab_crop)
                img_crop, lab_crop = augment.rotate()

            # Normalize and reshape the image crop
            img_crop = normalize_image(img_crop)
            img_crop = img_crop.reshape((img_crop.shape[0], img_crop.shape[1], 1))
            # plt.imshow(img_crop)
            # plt.title("patcher")
            # plt.show()
            images = img_crop
            masks = lab_crop
            #i += 1
            break
        # plt.imshow(images)
        # plt.title("patcher1")
        # plt.show()
        return images, masks

