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
import Scoring
from pyTiler.tiler import tiler
__author__ = "Chris Norton"
__maintainer__ = "Chris Norton"
__email__ = "cnorton@mtu.edu"
__status__ = "Dev"

# {code}
################################# EDIT THE LINE BELOW ###############################
test = "TrainingDataset/data_subset/323_subset/output/train/" ## EDIT THIS LINE
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
for path in random.sample(os.listdir(test), 5):  # Loop over Images in Directory
    print("loop", test + path)
    img = cv2.imread(test + path, -1).astype("float32")
    if useLabels:
        lab = cv2.imread(test + "../Labels/" + path, -1)  # HERE'S THE LINE THE READS THE LABELS

    batch_size = int(img.shape[0] / step) * int(img.shape[1] / step)
    patcher = Patcher(img=img, lab=lab, batch_size=1, input_shape=(dims, dims, 1), step=dims, num_classes=1)

    # get the patched images
    images, masks, _, _ = patcher.patch_overlap(visualize=True)

    # predict on patches
    preds_test = model.predict(images, verbose=1)

    print("Dimensions of preds_test: ", preds_test.shape)

    preds_test = (preds_test > 0.8)

    # if useLabels:
    #     labels = labels.reshape(labels.shape[0], labels.shape[1], 1)

    fig = plt.figure(figsize=(10, 4))
    fig.add_subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("image")

    fig.add_subplot(1, 3, 2)
    if useLabels:
        plt.imshow(lab)
        #cv2.imwrite("./inference/predictions/" + "label" + datetime.now().strftime("-%Y%m%d-%H:%M") + ".tif", lab)
    plt.axis('off')
    plt.title("label")

    unpatcher = Unpatcher(img, preds_test, test + path)
    full_pred_image = unpatcher.unpatch_image2()

    cv2.imwrite("./inference/predictions/" + "prediction" + datetime.now().strftime("-%Y%m%d-%H:%M") + ".tif", full_pred_image)

    int_img = np.array(full_pred_image, dtype="uint8")
    # grey = int_img[:,:,0]
    # ret, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # kernel = np.ones((3,3), np.uint8)
    # remove_noise = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    full_pred_image = full_pred_image.astype(np.uint8)
    print("pred:", full_pred_image.dtype)


    fig.add_subplot(1, 3, 3)
    plt.imshow(full_pred_image)
    #plt.imshow(preds_full_image)
    plt.axis('off')
    plt.title("prediction")

    plt.subplots_adjust(wspace=.05, hspace=.05, left=.01, right=.99, top=.99, bottom=.01)

    plt.savefig('data.png')
    plt.show()
    plt.close()

    def single_channel_to_three_channel(img):
        return np.repeat(img[..., np.newaxis], 3, axis=-1)

    # Convert single-channel images to three-channel images if necessary
    if full_pred_image.ndim == 2:
        full_pred_image = single_channel_to_three_channel(full_pred_image)

    if lab.ndim == 2:
        lab = single_channel_to_three_channel(lab)

    fsim_score = fsim(lab, full_pred_image)
    average_fsim += fsim_score
    print(fsim_score)

    out = cv2.imread('data.png')
    img = wandb.Image(out)
    #img = wandb.Image(PIL.Image.fromarray(out.get_image()[:, :, ::-1]))
    #vis_table.add_data(img)
    #vis_table.add_data(fsim_score)

#run.log({"infer_table": vis_table})
average_fsim /= 5
#wandb.log({"average_fsim": average_fsim})