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
    path = random.choice(batch_paths)
    if not os.path.isfile(image_path + path):
        print("error: " + image_path + path + " does not exist")
    if not os.path.isfile(label_path + path):
        print("error: " + label_path + path + " does not exist")
    img = cv2.imread(image_path + path, -1).astype("float32")
    lab = cv2.imread(label_path + path, -1)
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
