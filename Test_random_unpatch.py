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
from typing import Tuple, Union, cast
import wandb
import image_similarity_measures
from image_similarity_measures.quality_metrics import rmse, psnr, fsim

# Owned
from Patcher import Patcher
import Batch_loader
from Random_patcher import Random_patcher
from Unpatcher import Unpatcher
from Random_unpatcher import Random_unpatcher
import Scoring
__author__ = "Chris Norton"
__maintainer__ = "Chris Norton"
__email__ = "cnorton@mtu.edu"
__status__ = "Dev"

# {code}
################################# EDIT THE LINE BELOW ###############################
test = "TrainingDataset/data_subset/323_subset/output/test/" ## EDIT THIS LINE
useLabels = True # set to true if you have a folder called Labels inside test (the above variable)
# useLabels can be useful for seeing the accuracy.
################################# EDIT THE LINE ABOVE ###############################

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

#################### MAIN ************************
test = test + "Images/" # Uncomment if you have a folder inside called Images

dims = 512
step = 512
# Predict on patches
model_file = 'h5_files/model-best-UNet++512.h5'
model = load_model(model_file,
                  custom_objects = { 'dice_plus_bce_loss': Scoring.dice_plus_bce_loss,
                                    'dice_scoring': Scoring.dice_scoring})

# load test patches
images = np.zeros((len(os.listdir(test)), dims, dims, 1), dtype="float32")  # define the numpy array for the batch
masks = np.zeros((len(os.listdir(test)), dims, dims, 1), dtype=bool)
resize = np.zeros((1, dims, dims, 1), dtype=int)
average_fsim = 0
i = 0
print("total image shape:", images.shape)
#run = wandb.init(project='Cell-Segmentation', entity="nort")
#vis_table = wandb.Table(columns=["image"])
print(len(os.listdir(test)))
for path in os.listdir(test): # Loop over Images in Directory
    print("loop", test + path)
    img = cv2.imread(test + path, -1).astype("float32")
    if useLabels:
        lab = cv2.imread(test + "../Labels/" + path, -1) # HERE'S THE LINE THE READS THE LABELS
    #
    # batch_size = int(img.shape[0]/step) * int(img.shape[1]/step)
    # if not useLabels:
    #     patcher_img = Patcher(img, batch_size=batch_size, input_shape=(dims, dims, 1), step=step)
    # else:
    #     patcher_img = Patcher(img, lab, batch_size=batch_size, input_shape=(dims, dims, 1), step=step)
    # images, masks, row, col = patcher_img.patch_image()
    # print("1 image shape:", images.shape)
    # preds_test = model.predict(images, verbose=1)

    # Predicting resized images
    #resized = cv2.resize(img, (dims, dims))
    #resize = resized.reshape(1, step, step, 1)

    # Predicting full sized images
    #preds_full_image = model.predict(resize)
    #preds_test = (preds_test > 0.7) #.astype(np.uint8) # showing predictions with
    #preds_full_image = (preds_full_image > 0.4).astype(np.uint8)


    lab = lab.reshape(lab.shape[0], lab.shape[1], 1)
    #lab = normalize_image(lab)
    #lab = lab.astype(np.uint8)
    print("lab:", lab.dtype)


    # create figure
    fig = plt.figure(figsize=(10, 4))

    fig.add_subplot(1, 3, 1)

    # showing image
    plt.imshow(img)
    plt.axis('off')
    plt.title("image")

    fig.add_subplot(1, 3, 2)

    # showing image
    if useLabels:
        plt.imshow(lab)
        #cv2.imwrite("./inference/predictions/" + "label" + datetime.now().strftime("-%Y%m%d-%H:%M") + ".tif", lab)
    plt.axis('off')
    plt.title("label")

    unpatcher = Random_unpatcher(img, img_name=test+path, model=model, input_shape=(dims, dims, 1), num_crop=500)
    full_pred_image = unpatcher.efficient_random_unpatch()

    #cv2.imwrite("./inference/predictions/random_unpatch/" + "random_unpatch_prediction" + datetime.now().strftime("-%Y%m%d-%H:%M") + ".tif", full_pred_image)

    int_img = np.array(full_pred_image, dtype="uint8")
    #grey = int_img[:,:,0]
    # ret, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # kernel = np.ones((3,3), np.uint8)
    # remove_noise = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    #full_pred_image = full_pred_image.astype(np.uint8)
    print("pred:", full_pred_image.dtype)


    fig.add_subplot(1, 3, 3)
    plt.imshow(full_pred_image)
    #plt.imshow(preds_full_image)
    plt.axis('off')
    plt.title("prediction")

    plt.subplots_adjust(wspace=.05, hspace=.05, left=.01, right=.99, top=.99, bottom=.01)
    full_pred_image = full_pred_image.reshape(full_pred_image.shape[0], full_pred_image.shape[1], 1)
    fsim_score = fsim(lab, full_pred_image)
    average_fsim += fsim_score
    print(fsim_score)
    text = "fsim_score: " + str(fsim_score)
    fig.text(.5, .05, text, ha='center')

    plt.savefig("./inference/predictions/random_unpatch/" + "random_unpatch_prediction" + datetime.now().strftime("-%Y%m%d-%H:%M") + ".png")
    #plt.show()
    plt.close()

    #out = cv2.imread('data.png')
    #img = wandb.Image(out)
    #img = wandb.Image(PIL.Image.fromarray(out.get_image()[:, :, ::-1]))
    #vis_table.add_data(img)
    #vis_table.add_data(fsim_score)

#run.log({"infer_table": vis_table})
average_fsim /= len(os.listdir(test))
print("average_fsim:", average_fsim)
#wandb.log({"average_fsim": average_fsim})
