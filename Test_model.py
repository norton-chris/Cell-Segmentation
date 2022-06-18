#----------------------------------------------------------------------------
# Created By  : Chris Norton
# ---------------------------------------------------------------------------
"""
This program will load model weights inputted and show
the output with a test image.
"""
# ---------------------------------------------------------------------------

# Built-in
import os
import sys

# 3rd Party Libs
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
from PIL import Image

# Owned
from Patcher import Patcher
import Batch_loader
from Random_patcher import Random_patcher
import Scoring
__author__ = "Chris Norton"
__maintainer__ = "Chris Norton"
__email__ = "cnorton@mtu.edu"
__status__ = "Dev"

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
root = "TrainingDataset/data_subset/output/train/"
test = root + "Images/"
dims = 512
step = 512
# Predict on patches
model = load_model('h5_files/model-best.h5', # train_UNet512TIF50E-20220519-23.41.h5
                  custom_objects = { 'dice_plus_bce_loss': Scoring.dice_plus_bce_loss,
                                    'dice_scoring': Scoring.dice_scoring})

# load test patches
images = np.empty((len(os.listdir(test)), dims, dims, 1), dtype="float32")  # define the numpy array for the batch
masks = np.empty((len(os.listdir(test)), dims, dims), dtype=bool)
i = 0
num_of_images = 0
image_name = ""
max_x_pixel = 512
max_y_pixel = 512
for path in os.listdir(test):
    print("loop", test + path)
    img = cv2.imread(test + path, -1).astype("float32")
    lab = cv2.imread(root + "Labels/" + path, -1)

    patcher_img = Random_patcher(img, lab, batch_size=1, input_shape=(dims, dims, 1), step=step)
    images, masks = patcher_img.patch_image()
    print(images.shape)
    preds_test = model.predict(images.reshape((1,512,512,1)), verbose=1)
    #pred_imgs = np.empty((i, dims, dims, 1), dtype=int)
    #preds_test = (preds_test > 0.3).astype(np.uint8)
    for i in range(0, len(preds_test)):
        # create figure
        fig = plt.figure(figsize=(10, 7))

        # Adds a subplot at the 1st position
        fig.add_subplot(1, 3, 1)

        # showing image
        plt.imshow(images.reshape((1,512,512))[i])
        plt.axis('off')
        plt.title("Image")

        # Adds a subplot at the 2nd position
        fig.add_subplot(1, 3, 2)

        # showing image
        plt.imshow(masks.reshape((1,512,512))[i])
        plt.axis('off')
        plt.title("Label")

        # Adds a subplot at the 3rd position
        fig.add_subplot(1, 3, 3)

        # showing image
        plt.imshow(preds_test[i])
        plt.axis('off')
        plt.title("Prediction")
        plt.show()

        #cv2.imwrite("inference/predictions/images/Image[" + str(i) + "].tif", images[i])
        #cv2.imwrite("inference/predictions/predict/Prediction[" + str(i) + "].tif", preds_test[i])
