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
from PIL import Image
from patchify import patchify, unpatchify


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

def patch_togetherv3(img, full_image, batch_size=1, step=512):
    max_x_pixel = full_image.shape
    max_y_pixel = full_image.shape
    # batch_size = int((float(max_x_pixel[0]) / float(self.step)) * (float(max_y_pixel[0]) / float(self.step)))
    curx = 0
    cury = 0

    i = 0
    # if self.image:
    images = np.empty((batch_size, step, step, 1))
    pred_imgs = np.empty((step, step, 1))
    # else:
    #     images = np.empty((batch_size, step, step, 1), dtype=bool)
    print("max x:", max_x_pixel[1])
    while cury < max_y_pixel[1]:
        x_cur = curx
        while curx < max_x_pixel[1]:

            # if self.image:
            cropped = img[curx:curx + step, cury:cury + step]

            # else:
            #     cropped = self.img[curx:curx + step, cury:cury + step]

            curx = curx + step

            try:
                images[i] = cropped.reshape(step, step, 1)
                i += 1
            except:
                pass
        curx = x_cur
        cury = cury + step

def patch_togetherv2(prediction, image_name):
    img = cv2.imread(image_name)
    print(img.shape)
    print(prediction.shape)
    reconstructed_image = unpatchify(prediction, img.shape)
    plt.imshow(reconstructed_image)
    plt.show()
    return reconstructed_image


def patch_together(prediction, org_shape):
    full_image = np.empty((1, (*org_shape)), dtype=int)
    j = 0
    k = 0
    print(prediction.shape)
    for x in range(0, dims):
        k -= dims
        for y in range(0, dims):
            print(prediction)
            #full_image[j][k] = prediction
            k += 1
        j += 1
    print("writing full image")
    plt.imshow(full_image)
    plt.show()
    cv2.imwrite("patched_full/patched_image" + datetime.now().strftime("-%Y%m%d-%H.%M") + ".jpg")

def visualEvaluation(label, prediction):
    eval_image = np.empty((len(os.listdir(test)), dims, dims, 1), dtype=int)
    for big_image in range(0,len(os.listdir(test))):
        j = 0
        k = 0
        for i in range(0, len(prediction)):
            for x in range(0,len(prediction[i])):
                for y in range(0, len(prediction[i])):
                    if(label[x][y] != prediction[x][y]):
                        eval_image[j][k] == (0,0,1)
                    k += 1
                j += 1


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
root = "E:/Han Project/TrainingDataset/TrainSubset/"
test = root + "Images/"
#test_mask = "E:/Han Project/TrainingDataset/TrainingDataset/output/train/Labels/"
dims = 512
step = 512
# Predict on patches
model = load_model('h5_files/train_subset_UNet512JPG10E-20220410-12.50.h5',
                  custom_objects = { 'dice_plus_bce_loss': dice_plus_bce_loss,
                                    'dice_scoring': dice_scoring})

# load test patches
images = np.empty((len(os.listdir(test)), dims, dims, 1), dtype=int)  # define the numpy array for the batch
masks = np.empty((len(os.listdir(test)), dims, dims, 1), dtype=bool)
i = 0
num_of_images = 0
image_name = ""
max_x_pixel = 512
max_y_pixel = 512
for path in os.listdir(test):
    print("loop", test + path)
    img = cv2.imread(test + path, -1)
    lab = cv2.imread(root + "Labels/" + path, -1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
    try:
        img = np.expand_dims(img, axis=2)
        lab = np.expand_dims(lab, axis=2)
    except Exception as e:
        print(e)
        print("img shape:", img.shape)
        continue

    # plt.imshow(img)
    # plt.show()

    patcher_img = Random_patcher(img, batch_size=4, step=step)
    patcher_lab = Random_patcher(lab, batch_size=4, step=step, image=False)
    images = patcher_img.patch_image()
    masks = patcher_lab.patch_image()
# images = Batch_loader.BatchLoad(test, batch_size = 1, dim = dims, step=step)
print(images.shape)
preds_test = model.predict(images, verbose=1)
#pred_imgs = np.empty((i, dims, dims, 1), dtype=int)
preds_test = (preds_test > 0.12).astype(np.uint8)
for i in range(0, len(preds_test)):
    plt.imshow(images[i])
    plt.show()
    cv2.imwrite("inference/predictions/images/Image[" + str(i) + "].tif", images[i])

    plt.imshow(preds_test[i])
    plt.show()
    #pred_imgs[i] = preds_test[i]
    cv2.imwrite("inference/predictions/predict/Prediction[" + str(i) + "].tif", preds_test[i])

    # plt.imshow(full_pred_image)
    # plt.show()

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




