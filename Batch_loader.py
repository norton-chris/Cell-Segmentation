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

    def __len__(self):
        return int(np.floor(len(self.paths) / self.batch_size))

    def __data_generation(self, batch_paths):
        # Initialization
        images = np.empty((self.batch_size, *self.dim), dtype=int)  # define the numpy array for the batch
        masks = np.empty((self.batch_size, self.dim[0], self.dim[1], self.num_classes), dtype=bool)
        i = 0
        num_of_images = 0
        image_path = self.paths + "Image_out/"
        label_path = self.paths + "Label_out/"
        for path in batch_paths:
            print("loop:", image_path + path)
            img = cv2.imread(image_path + path, cv2.IMREAD_COLOR)
            lab = cv2.imread(label_path + path, cv2.IMREAD_COLOR)
            print(type(img))
            tf.image.adjust_contrast(
                img, 2
            )
            plt.imshow(img)
            plt.show()


            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
            except:
                print("error in cvtcolor")
                continue
            plt.imshow(img)
            plt.show()
            # img = np.array(img)
            # lab = np.array(lab)



            max_x_pixel = img[0].shape
            max_y_pixel = img[1].shape
            batch_size = int((float(max_x_pixel[0]) / float(self.step)) * int(float(max_y_pixel[0]) / float(self.step)))
            num_of_images += batch_size
            images = np.resize(images, [num_of_images, *self.dim])
            masks = np.resize(images, [num_of_images, self.dim[0], self.dim[1], self.num_classes])

            input_size = self.dim
            # image_patches = patchify(img, (input_size, input_size), step=512)
            # label_patches = patchify(lab, (input_size, input_size), step=512)
            patcher_img = Patcher(img, batch_size= batch_size, step=self.step)
            patcher_lab = Patcher(lab, batch_size= batch_size, step=self.step, image = False)
            image_patches, actual_batch_size = patcher_img.patch_image()
            label_patches, actual_batch_size = patcher_lab.patch_image()
            # num_patch = len(image_patches) - 1
            # rand_patch = random.randint(0, num_patch)
            # image_patch = image_patches[rand_patch]
            # label_patch = label_patches[rand_patch][0]
            # print("image shape before cvtcolor:", image_patch.shape)
            # image_patch = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
            # label_patch = cv2.cvtColor(label_patch, cv2.COLOR_BGR2GRAY)
            # print("reshape:", image_patches[rand_patch].reshape(512, 512, -1).shape)

            k = 0

            while k < actual_batch_size:
                images[i] = image_patches[k]
                masks[i] = label_patches[k]
                if images[i].max() == 0.00:
                    print("image array shape:", images.shape)
                    print("all pixels 0")
                    k += 1
                    continue # don't include black images
                i += 1
                k += 1
                #print("image", i)

            # if images[i].max() == 0:
            #     print("image array shape:", images.shape)
            #     print("all pixels 0")



        return images, masks

    def __getitem__(self, index):
        print(self.paths)
        paths_temp = os.listdir(self.paths + "Image_out/")[
                     index * self.batch_size:(index + 1) * self.batch_size]

        # can augment here
        images, masks = self.__data_generation(paths_temp)

        return images, masks
