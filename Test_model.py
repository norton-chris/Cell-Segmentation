import os
import sys
import warnings
from datetime import datetime
import cv2

import numpy as np
import random
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import Models
from tqdm.keras import TqdmCallback
import matplotlib.pyplot as plt
from Batch_load_test import BatchLoadTest
from patchify import patchify, unpatchify
from Patcher import Patcher

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

test = "E:/Han Project/TrainingDataset/test/Images/"
dims = 128

# Predict on patches
model = load_model('train_subset_UNet128JPG25E-20220322-17.07.h5',
                  custom_objects = { 'dice_plus_bce_loss': dice_plus_bce_loss,
                                    'dice_scoring': dice_scoring})

# load test patches
images = np.empty((len(os.listdir(test)), dims, dims, 1), dtype=int)  # define the numpy array for the batch
i = 0
num_of_images = 0
image_name = ""
for path in os.listdir(test):
    print("loop", test + path)
    img = cv2.imread(test + path)
    image_name = test + path

    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        print("error in cvtcolor")
        continue
    # img = np.array(img)
    # lab = np.array(lab)

    max_x_pixel = img[0].shape
    max_y_pixel = img[1].shape
    batch_size = int((float(max_x_pixel[0]) / float(dims)) * int(float(dims) / float(dims)))
    num_of_images += batch_size
    images = np.resize(images, [num_of_images, dims, dims, 1])

    input_size = dims
    # image_patches = patchify(img, (input_size, input_size), step=512)
    # label_patches = patchify(lab, (input_size, input_size), step=512)
    patcher_img = Patcher(img, batch_size= batch_size, step=dims)
    image_patches, actual_batch_size = patcher_img.patch_image()

    k = 0

    while k < actual_batch_size:
        images[i] = image_patches[k]
        if images[i].max() == 0.00:
            print("image array shape:", images.shape)
            print("all pixels 0")
            k += 1
            continue # don't include black images
        i += 1
        k += 1


print(images.shape)
preds_test = model.predict(images, verbose=1)
for i in range(0, i):
    # plt.imshow(images[i])
    # plt.show()
    cv2.imwrite("Image[" + str(i) + "].jpg", images[i])
    # plt.imshow(preds_test[i])
    # plt.show()
    cv2.imwrite("Prediction[" + str(i) + "].jpg", preds_test[i])

print("patching images together")
patch_togetherv2(preds_test, image_name)



