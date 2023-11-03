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

import ray


def normalize_image(input_block):
    block = input_block.copy()
    vol_max, vol_min = block.max(), block.min()
    if not vol_max == vol_min:  # run a check. otherwise error when divide by 0
        for i in range(block.shape[-1]):
            block[:, :, i] = (block[:, :, i] - vol_min) / (vol_max - vol_min)
    return block

@ray.remote(num_returns=2)
def thread_batch_loader(batch_paths, image_path, label_path, patching, dim, step, augment):
    # Ensure batch_paths is a list before selecting a path
    if not isinstance(batch_paths, list):
        batch_paths = [batch_paths]
    path = random.choice(batch_paths)
    full_image_path = os.path.join(image_path, path)
    full_label_path = os.path.join(label_path, path)

    # Check if the image file exists
    if not os.path.isfile(full_image_path):
        print(f"error: {full_image_path} does not exist")
        return None, None  # Return None to indicate the error

    # Check if the label file exists
    if not os.path.isfile(full_label_path):
        print(f"error: {full_label_path} does not exist")
        return None, None  # Return None to indicate the error
    # Read the image and label files
    img = cv2.imread(full_image_path, -1)
    if img is None:
        raise ValueError(f"Failed to read the image file: {full_image_path}")
    img = img.astype("float32")

    lab = cv2.imread(full_label_path, -1)
    if lab is None:
        raise ValueError(f"Failed to read the label file: {full_label_path}")

    ret, lab = cv2.threshold(lab, 100, 255, cv2.THRESH_BINARY)

    if patching:
        patcher_img = Random_patcher(img, lab, batch_size=1, input_shape=dim, step=step, augment=augment)
        images, masks = patcher_img.patch_image()
    else:
        img = cv2.resize(img, (dim[0], dim[1]))
        img = np.array(img, dtype="float32")
        lab = cv2.resize(lab, ((dim[0], dim[1])))
        img = img.reshape(*dim)
        images = normalize_image(img)
        masks = lab.reshape((dim[0], dim[1]))
    return images, masks
