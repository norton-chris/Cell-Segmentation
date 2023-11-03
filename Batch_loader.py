#----------------------------------------------------------------------------
# Created By  : Chris Norton
# ---------------------------------------------------------------------------
"""
Loads batches of images into model.fit function.
It's multithreaded using ray, so every image is run in parallel.
"""
# ---------------------------------------------------------------------------

import os
from collections import defaultdict

import numpy as np
import random
import cv2
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import ray

import Patcher
from Patcher import Patcher
from Random_patcher import Random_patcher
from PIL import Image
from Thread_Batch_loader import thread_batch_loader



class BatchLoad(keras.utils.all_utils.Sequence):
    def __init__(self,
                 paths,
                 batch_size=1,
                 dim=(128, 128, 1),
                 step=128,
                 num_classes=1,
                 patching=True,
                 augment=False,
                 validate=False,
                 multiple_sizes=False):
        self.paths = paths
        self.batch_size = batch_size
        self.dim = dim
        self.num_classes = num_classes
        self.step = step
        self.paths_temp = os.listdir(self.paths + "Images/")
        self.patching = patching
        self.augment = augment
        self.validate = validate
        self.multiple_sizes = multiple_sizes  # Store the new parameter

        if self.multiple_sizes:
            self.resolutions = [(256, 256), (512, 512), (768, 768), (1024, 1024)]
            # self.size_to_paths = defaultdict(list)
            # for img_path in self.paths_temp:
            #     img = cv2.imread(os.path.join(self.paths + "Images/", img_path), 0)
            #     h, w = img.shape[:2]
            #     self.size_to_paths[(h, w)].append(img_path)

    def __len__(self):
        return int(np.floor(len(self.paths_temp) / (self.batch_size)))

    @staticmethod
    def is_divisible_by_pooling_factor(dim, levels=4, factor=2):
        h, w, _ = dim
        for _ in range(levels):
            if h % factor != 0 or w % factor != 0:
                return False
            h //= factor
            w //= factor
        return True

    def __data_generation(self, batch_paths):
        # Initialization
        images = []  # This will be a list to hold arrays of varying sizes
        masks = []  # This too
        image_path = self.paths + "Images/"
        label_path = self.paths + "Labels/"
        thread_ids = []

        # Loop over each path in the batch to process images individually
        for path in batch_paths:
            # If handling multiple sizes, determine the size dynamically
            if self.multiple_sizes:
                img = cv2.imread(os.path.join(image_path, path), 0)
                h, w = img.shape[:2]
                if not BatchLoad.is_divisible_by_pooling_factor((h, w, 1)):
                    raise ValueError(
                        f"Image dimensions must be divisible by the pooling factor. Problem with image: {path}")
                # Append an array of zeros with the correct shape to the images and masks lists
                images.append(np.zeros((h, w, 1), dtype="float32"))
                masks.append(np.zeros((h, w), dtype=bool))  # Assuming masks are 2D
            else:
                # If not handling multiple sizes, use the predefined `self.dim`
                images.append(np.zeros(self.dim, dtype="float32"))
                masks.append(np.zeros((self.dim[0], self.dim[1]), dtype=bool))

            # Append the task to the thread pool
            if not self.validate:
                thread_ids.append(thread_batch_loader.remote(batch_paths=path,
                                                             image_path=image_path,
                                                             label_path=label_path,
                                                             patching=self.patching,
                                                             dim=self.dim if not self.multiple_sizes else (h, w, 1),
                                                             step=self.step,
                                                             augment=self.augment))
            else:
                thread_ids.append(thread_batch_loader.remote(batch_paths=path,
                                                             image_path=image_path,
                                                             label_path=label_path,
                                                             patching=self.patching,
                                                             dim=self.dim if not self.multiple_sizes else (h, w, 1),
                                                             step=self.step,
                                                             augment=False))

        # Retrieve the results from the threads and assign to the appropriate arrays
        for i, thread_id in enumerate(thread_ids):
            img, mask = ray.get(thread_id)
            if self.multiple_sizes:
                # If multiple sizes, replace the zero arrays with the actual images and masks
                images[i] = img
                masks[i] = mask
            else:
                # If not multiple sizes, the images and masks are already correctly shaped
                images[i][:] = img
                masks[i][:] = mask

        # Print the sizes of the loaded images
        #for i, img in enumerate(images):
        #    print(f"Loaded image {i} with size: {img.shape}")

        # Convert lists to numpy arrays for batch processing
        images = np.array(images, dtype=object)
        masks = np.array(masks, dtype=object)

        return images, masks

    def __getitem__(self, item):
        # Choose a random dimension from the list of resolutions if handling multiple sizes
        if self.multiple_sizes:
            chosen_resolution = random.choice(self.resolutions)
            self.dim = (chosen_resolution[0], chosen_resolution[1], 1)  # Assuming square images
            #print("Chosen resolution for batch:", self.dim)

        # Initialize an empty list for valid batch paths
        valid_batch_paths = []

        # Attempt to collect a batch of valid paths
        while len(valid_batch_paths) < self.batch_size:
            # Randomly select a path if handling multiple sizes, otherwise use the next path in sequence
            path = random.choice(self.paths_temp) if self.multiple_sizes else self.paths_temp[
                item * self.batch_size + len(valid_batch_paths)]

            # Ensure the file extension is included
            if not path.endswith('.tif'):
                path += '.tif'

            full_image_path = os.path.join(self.paths + "Images/", path)
            full_label_path = os.path.join(self.paths + "Labels/", path)

            # Check if both the image and label files exist
            if os.path.exists(full_image_path) and os.path.exists(full_label_path):
                valid_batch_paths.append(path)
            else:
                print(f"Warning: Invalid path detected and skipped: {path}")

        # Generate data for the valid batch paths
        images, masks = self.__data_generation(valid_batch_paths)

        #images = np.expand_dims(images, axis=0)  # Add a batch dimension
        #masks = np.expand_dims(masks, axis=0)  # Add a batch dimension
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        masks = tf.convert_to_tensor(masks, dtype=tf.float32)

        return images, masks




