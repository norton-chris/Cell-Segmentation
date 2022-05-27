import os
import sys
import warnings
from datetime import datetime
import cv2

import numpy as np
import random
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import Models
from tqdm.keras import TqdmCallback
import matplotlib.pyplot as plt
#from Batch_load_test import BatchLoadTest

from Patcher import Patcher
import Batch_loader
from Random_patcher import Random_patcher
from Unpatcher import Unpatcher


from PIL import Image
from patchify import patchify, unpatchify
from typing import Tuple, Union, cast


def dice_plus_bce_loss(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
    inputs_dice = K.flatten(tf.cast(inputs, tf.float32))
    inputs_bce = K.flatten(tf.cast(inputs, tf.float32))
    targets = K.flatten(tf.cast(targets, tf.float32))

    intersection = K.sum(targets * inputs_dice)

    dice = (2 * intersection) / (K.sum(targets) + K.sum(inputs_dice) + smooth)

    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    bce = bce(targets, inputs_bce)

    return -K.log(dice + smooth)

def dice_scoring(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(tf.cast(inputs, tf.float32))
    targets = K.flatten(tf.cast(targets, tf.float32))

    intersection = K.sum(targets * inputs)
    dice = (2 * intersection) / (K.sum(targets) + K.sum(inputs) + smooth)
    return dice

def normalize_image(input_block):
    block = input_block.copy()
    vol_max, vol_min = block.max(), block.min()
    if not vol_max == vol_min:  # run a check. otherwise error when divide by 0
        for i in range(block.shape[-1]):
            block[:, :, i] = (block[:, :, i] - vol_min) / (vol_max - vol_min)
    return block

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
#root = "E:/Han Project/TrainingDataset/TrainingDataset/output/train/"
root = "TrainingDataset/correct_labels_subset/output/test/"
test = root + "Images/"
#test_mask = "E:/Han Project/TrainingDataset/TrainingDataset/output/train/Labels/"
dims = 512
step = 512
# Predict on patches
model = load_model('h5_files/UNET++512TIF150E-20220525-15.22.h5',
                  custom_objects = { 'dice_plus_bce_loss': dice_plus_bce_loss,
                                    'dice_scoring': dice_scoring})

# load test patches
images = np.zeros((len(os.listdir(test)), dims, dims, 1), dtype=int)  # define the numpy array for the batch
masks = np.zeros((len(os.listdir(test)), dims, dims, 1), dtype=bool)
resize =  np.zeros((1, dims, dims, 1), dtype=int)
i = 0
num_of_images = 0
image_name = ""
max_x_pixel = 512
max_y_pixel = 512
print("total image shape:", images.shape)
for path in os.listdir(test):
    print("loop", test + path)
    img = cv2.imread(test + path, -1)
    lab = cv2.imread(root + "Labels/" + path, -1)
    # img = Image.open(test + path)
    # lab = Image.open(root + "Labels/" + path)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
    # try:
    #     img = np.expand_dims(img, axis=2)
    #     lab = np.expand_dims(lab, axis=2)
    # except Exception as e:
    #     print(e)
    #     print("img shape:", img.shape)
    #     continue

    # plt.imshow(img)
    # plt.show()

    # patcher_img = Random_patcher(img, batch_size=4, step=step)
    # patcher_lab = Random_patcher(lab, batch_size=4, step=step, image=False)
    # images = patcher_img.patch_image()
    # masks = patcher_lab.patch_image()
    batch_size = int(img.shape[0]/step) * int(img.shape[1]/step)
    patcher_img = Patcher(img, lab, batch_size=batch_size, input_shape=(dims, dims, 1), step=step)
    # patcher_lab = Random_patcher(lab, batch_size= self.batch_size, step=self.step, image = False)
    images, masks, row, col = patcher_img.patch_image()
    # images = Batch_loader.BatchLoad(test, batch_size = 1, dim = dims, step=step)
    print("1 image shape:", images.shape)
    preds_test = model.predict(images, verbose=1)

    resized = cv2.resize(img, (dims, dims))
    resize = resized.reshape(1, step, step, 1)
    preds_full_image = model.predict(resize)
    #pred_imgs = np.empty((i, dims, dims, 1), dtype=int)
    preds_test = (preds_test > 0.4).astype(np.uint8)
    preds_full_image = (preds_full_image > 0.4).astype(np.uint8)
    for i in range(0, len(preds_test)):
        # create figure
        fig = plt.figure(figsize=(10, 7))

        for j in range(1, batch_size+1):
            # Adds a subplot at the 1st position
            fig.add_subplot(int(row/step) + 1, int(col/step), j)

            # showing image
            plt.imshow(preds_test[j-1])
            plt.axis('off')
            # plt.title("Image")

            # # Adds a subplot at the 2nd position
            # fig.add_subplot(row, col,r)
            # r += 1
            #
            # # showing image
            # plt.imshow(masks[i])
            # plt.axis('off')
            # plt.title("Label")
            #
            # # Adds a subplot at the 3rd position
            # fig.add_subplot(1, 3, r)
            # r += 1
            #
            # # showing image
            # plt.imshow(preds_test[i])
            # plt.axis('off')
            # plt.title("Prediction")
            # plt.show()
            # plt.imshow(images[i])
            # plt.show()
            # plt.imshow(masks[i])
            # plt.show()
            #cv2.imwrite("inference/predictions/images/Image[" + str(i) + "].tif", images[i])

            # plt.imshow(preds_test[i])
            #plt.show()
            #pred_imgs[i] = preds_test[i]
            #cv2.imwrite("inference/predictions/predict/Prediction[" + str(i) + "].tif", preds_test[i])
        fig.add_subplot(int(row/step)+ 1, int(col/step), j+1)

        # showing image
        plt.imshow(img)
        plt.axis('off')
        plt.title("image")

        fig.add_subplot(int(row/step)+ 1, int(col/step), j + 2)

        # showing image
        plt.imshow(lab)
        plt.axis('off')
        plt.title("label")

        # TODO implement full size prediction
        # patches = patchify(preds_test, (len(preds_test), int(preds_test.shape[1]/step), int(preds_test.shape[2]/step), 1), step=1)  # patch shape [2,2,3]
        #print(patches.shape)  # (511, 511, 1, 2, 2, 3). Total patches created: 511x511x1
        #preds_test = np.array(preds_test, dtype=np.uint8)
        #full_pred_image = unpatchify(patches, (batch_size,img.shape[0], img.shape[1],1))
        unpatcher = Unpatcher(img, preds_test, img_name=test+path)
        full_pred_image = unpatcher.unpatch_image()

    #
        fig.add_subplot(int(row/step)+ 1, int(col/step), j + 3)
    #
    #     # showing image
    #     plt.imshow(full_pred_image)
    #     plt.axis('off')
    #     plt.title("full size prediction")
        plt.imshow(full_pred_image)
        #plt.imshow(preds_full_image)
        plt.axis('off')
        plt.title("label")
        plt.show()
        break

#full_pred_image = patch_togetherv3(pred_imgs, images, batch_size=1, step=step)
# batch_size = int((float(max_x_pixel[0]) / float(self.step)) * (float(max_y_pixel[0]) / float(self.step)))
curx = 0
cury = 0

i = 0
# if self.image:
images = np.empty((4, step, step, 1))
# full_pred_image = Image.open(root + "Images/1.tif")
# plt.imshow(full_pred_image)
# plt.show()
#full_pred_image = np.empty((max_x_pixel[0], max_x_pixel[0], 1))
#full_pred_image = Image.new('RGB', (max_x_pixel[0], max_x_pixel[0]))
# else:
#     images = np.empty((batch_size, step, step, 1), dtype=bool)
# print("max x:", max_x_pixel[0])
# print("max y:", max_y_pixel[0])
# while cury < max_y_pixel[0]:
#     x_cur = curx
#     print("xcur:",x_cur, "cury:",cury)
#     while curx < max_x_pixel[0]:
#         print("curx:", curx, "cury:", cury)
#         # if self.image:
#         # cropped = img[curx:curx + step, cury:cury + step]
#
#         # else:
#         #     cropped = self.img[curx:curx + step, cury:cury + step]
#
#
#
#         try:
#             full_pred_image = cv2.hconcat()
#
#             plt.imshow(preds_test[i])
#             plt.show()
#             i += 1
#         except:
#             pass
#         curx = curx + step
#     curx = x_cur
#     cury = cury + step

# x = max_x_pixel[0] / step
# i=0
# k=0
# x_offset = 0
# for im in preds_test:
#     if i > x:
#         k += step
#         x_offset =0
#     print("x_offset:", x_offset, "k:", k)
#     full_pred_image.paste(im, (x_offset,k))
#     x_offset += step
#     i += 1
print(preds_test.shape)
full_pred_image = unpatchify(preds_test, (1536, 1536))
print("patching images together")
#patch_togetherv2(preds_test, image_name)
plt.imshow(full_pred_image)
plt.show()
