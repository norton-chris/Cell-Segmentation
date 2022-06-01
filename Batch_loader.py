import os
import numpy as np
import random
import cv2
import keras
import matplotlib.pyplot as plt
import tensorflow as tf

import Patcher
from Patcher import Patcher
from Random_patcher import Random_patcher
from PIL import Image
from Thread_Batch_loader import thread_batch_loader

import ray


class BatchLoad(keras.utils.all_utils.Sequence):
    def __init__(self,
                 paths,
                 batch_size=1,
                 dim=(128, 128, 1),
                 step=128,
                 num_classes=1,
                 patching = True,
                 augment = False,
                 validate = False):
        self.paths = paths
        self.batch_size = batch_size
        self.dim = dim
        self.num_classes = num_classes
        self.i = 0
        self.step = step
        self.paths_temp = os.listdir(self.paths + "Images/")
        self.patching = patching
        self.augment = augment
        self.validate = validate

    def __len__(self):
        return int(np.floor(len(self.paths_temp) / (self.batch_size)))



    def __data_generation(self, batch_paths):
        def normalize_image(input_block):
            block = input_block.copy()
            vol_max, vol_min = block.max(), block.min()
            if not vol_max == vol_min:  # run a check. otherwise error when divide by 0
                for i in range(block.shape[-1]):
                    block[:, :, i] = (block[:, :, i] - vol_min) / (vol_max - vol_min)
            return block
        # Initialization
        images = np.zeros((self.batch_size, *self.dim), dtype="float32")  # define the numpy array for the batch
        masks = np.zeros((self.batch_size, self.dim[0], self.dim[1]), dtype=bool)
        image_path = self.paths + "Images/"
        label_path = self.paths + "Labels/"
        thread_ids = []
        if not self.validate:
            for i in range(0, self.batch_size):
                thread_ids.append(thread_batch_loader.remote(batch_paths=batch_paths,
                                                                image_path=image_path,
                                                                label_path=label_path,
                                                                patching=self.patching,
                                                                dim=self.dim,
                                                                step=self.step,
                                                                augment=self.augment))
        else:
            for i in range(0, self.batch_size):
                thread_ids.append(thread_batch_loader.remote(batch_paths=batch_paths,
                                                                image_path=image_path,
                                                                label_path=label_path,
                                                                patching=self.patching,
                                                                dim=self.dim,
                                                                step=self.step,
                                                                augment = False))
        for i in range(self.batch_size):
            images[i], masks[i] = ray.get(thread_ids[i])



        return images, masks

    def __getitem__(self, item):
        # print(self.paths)

        # can augment here
        images, masks = self.__data_generation(self.paths_temp)

        return images, masks

