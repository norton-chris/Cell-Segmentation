import os
import numpy as np
import random
from patchify import patchify
import cv2
import keras
import matplotlib.pyplot as plt
import tensorflow as tf


import Patcher
from Patcher import Patcher
from Random_patcher import Random_patcher
from PIL import Image


class BatchLoad(keras.utils.all_utils.Sequence):
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
        self.paths_temp = os.listdir(self.paths + "Images/")


    def __len__(self):
        return int(np.floor(len(self.paths_temp) / (self.batch_size)))

    def __data_generation(self, batch_paths):
        # Initialization
        images = np.zeros((self.batch_size, *self.dim), dtype=int)  # define the numpy array for the batch
        masks = np.zeros((self.batch_size, self.dim[0], self.dim[1]), dtype=bool)
        i = 0
        num_of_images = 0
        image_path = self.paths + "Images/"
        label_path = self.paths + "Labels/"
        for path in batch_paths:
            #print("loop:", image_path + path)
            img = cv2.imread(image_path + path, -1)
            lab = cv2.imread(label_path + path, -1)

            #  img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
            # lab = cv2.GaussianBlur(lab, (5, 5), cv2.BORDER_DEFAULT)

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
            # try:
            #     #print("try img shape:", img.shape)
            #     #print("try lab shape:", lab.shape)
            # img = np.expand_dims(img, axis=2)
            # lab = np.expand_dims(lab, axis=2)
            # except Exception as e:
            #     print(e)
            #     #print("img shape:", img.shape)
            #     #print("lab shape:", lab.shape)
            #     continue

            # plt.imshow(img)
            # plt.show()
            # plt.imshow(lab)
            # plt.show()

            patcher_img = Random_patcher(img, lab, batch_size= 1, input_shape = self.dim, step=self.step)
            #patcher_lab = Random_patcher(lab, batch_size= self.batch_size, step=self.step, image = False)
            images, masks = patcher_img.patch_image()
            #masks = patcher_lab.patch_image()

            # k = 0
            # # plt.imshow(patcher_img[0])
            # # plt.show()
            # # plt.imshow(patcher_lab[0])
            # # plt.show()
            # while k < self.batch_size:
            #     images[i] = image_patches[k]
            #     masks[i] = label_patches[k]
            #     # plt.imshow(masks[i])
            #     # plt.show()
            #     if masks[i].max() == 0.00:
            #         # print("image array shape:", images.shape)
            #         # print("all pixels 0")
            #         k += 1
            #         continue # don't include black images
            #     i += 1
            #     k += 1
                #print("image", i)

            # if images[i].max() == 0:
            #     print("image array shape:", images.shape)
            #     print("all pixels 0")
            #i += 1

        return images, masks

    def __getitem__(self, index):
        # print(self.paths)

        # can augment here
        images, masks = self.__data_generation(self.paths_temp)

        return images, masks
