import os
import numpy as np
import random
from patchify import patchify
import cv2
import keras

import Patcher
from Patcher import Patcher
from PIL import Image


class BatchLoadTest(keras.utils.all_utils.Sequence):
    def __init__(self,
                 paths,
                 batch_size=1,
                 dim=(128, 128, 1),
                 step=128,
                 num_classes=1):
        self.paths = paths
        self.batch_size = batch_size
        self.dim = dim
        self.num_classes = num_classes
        self.i = 0
        self.step = step

    def __len__(self):
        return int(np.floor(len(self.paths) / self.batch_size))

    def __data_generation(self, batch_paths):
        # Initialization
        images = np.empty((self.batch_size, *self.dim), dtype=int)  # define the numpy array for the batch
        i = 0
        num_of_images = 0
        for path in batch_paths:
            print("loop", self.paths + path)
            img = cv2.imread(self.paths + path)

            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except:
                print("error in cvtcolor")
                continue
            # img = np.array(img)
            # lab = np.array(lab)

            max_x_pixel = img[0].shape
            max_y_pixel = img[1].shape
            batch_size = int((float(max_x_pixel[0]) / float(self.step)) * int(float(max_y_pixel[0]) / float(self.step)))
            num_of_images += batch_size
            images = np.resize(images, [num_of_images, *self.dim])

            input_size = self.dim
            # image_patches = patchify(img, (input_size, input_size), step=512)
            # label_patches = patchify(lab, (input_size, input_size), step=512)
            patcher_img = Patcher(img, batch_size= batch_size, step=self.step)
            image_patches, actual_batch_size = patcher_img.patch_image()

            k = 0

            while k < actual_batch_size:
                images[i] = image_patches[k]
                if images[i].max() == 0:
                    print("image array shape:", images.shape)
                    print("all pixels 0")
                i += 1
                k += 1
                #print("image", i)

            # if images[i].max() == 0:
            #     print("image array shape:", images.shape)
            #     print("all pixels 0")

        return images

    def __getitem__(self, index):
        print(self.paths)
        paths_temp = os.listdir(self.paths)[
                     index * self.batch_size:(index + 1) * self.batch_size]

        # can augment here
        images = self.__data_generation(paths_temp)

        return images
