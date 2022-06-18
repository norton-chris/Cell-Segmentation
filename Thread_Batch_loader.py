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
    #print("loop:", image_path + path)
    img = cv2.imread(image_path + path, -1).astype("float32")
    lab = cv2.imread(label_path + path, -1)
    #p = path.split(".")
    #lab = cv2.imread(label_path + p[0] + ".png", 0)
    ret, lab = cv2.threshold(lab, 100, 255, cv2.THRESH_BINARY)

    #img = normalize_image(img.reshape(self.dim))

    #  img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    # lab = cv2.GaussianBlur(lab, (5, 5), cv2.BORDER_DEFAULT)
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
    if patching:
    #if bool(random.getrandbits(1)):
        patcher_img = Random_patcher(img, lab, batch_size=1, input_shape=dim, step=step, augment=augment)
        #patcher_lab = Random_patcher(lab, batch_size= self.batch_size, step=self.step, image = False)
        images, masks = patcher_img.patch_image()
        #images[i] = image
        # plt.imshow(image)
        # plt.title("patched")
        # plt.show()
        #masks[i] = mask
        # plt.imshow(masks[i])
        # plt.title("pathced masks")
        # plt.show()
        # plt.imshow(images[i])
        # plt.title("patched1")
        # plt.show()
        #masks = patcher_lab.patch_image()
    else:
        img = cv2.resize(img, (dim[0], dim[1]))
        img = np.array(img, dtype="float32")
        lab = cv2.resize(lab, ((dim[0], dim[1])))
        img = img.reshape(*dim)
        # plt.imshow(img)
        # plt.title("reshape")
        # plt.show()
        images = normalize_image(img)
        # plt.imshow(img)
        # plt.title("normalized")
        # plt.show()
        #images = img
        # plt.imshow(images[i])
        # plt.title("normalized images")
        # plt.show()
        masks = lab.reshape((dim[0], dim[1]))
    return images, masks
