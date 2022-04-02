import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images, show
from skimage.transform import resize
from skimage.morphology import label
from PIL import ImageFile

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
# TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = 'test_images/'

test_ids = next(os.walk(TEST_PATH))[1]

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
# for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    # path = TEST_PATH + id_
    
    # #Read images iteratively
    # img = imread(dir_path + path + id_ + '.tif')[:,:,:IMG_CHANNELS]
    
    # imshow(img)
    # show()
    
    # #Get test size
    # sizes_test.append([img.shape[0], img.shape[1]])
    
    # #Resize image to match training data
    # img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    
    # #Append image to numpy array for test dataset
    # X_test[n] = img
    
img = imread("3T3_vinculin001.png")[:,:,:IMG_CHANNELS]

#imshow(img)
#show()

#Get test size
print("image shape",img.shape)
sizes_test.append([img.shape[0], img.shape[1], IMG_CHANNELS])

#Resize image to match training data
img = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS) )#, mode='constant', preserve_range=True)
img = img.reshape((-1,128,128,3)).astype('float')
print("image shape",img.shape)

X_test = img
	

print('Done!')

# Predict on train, val and test
model = load_model('model_unet_checkpoint.h5')
#preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
#preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
#preds_train_t = (preds_train > 0.5).astype(np.uint8)
#preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test_t)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test_t[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))
									   
# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_test_t))
#imshow(X_train[ix])
#plt.show()
#imshow(np.squeeze(Y_train[ix]))
#plt.show()
#imshow(np.squeeze(preds_train_t[ix]))
#plt.show()
imshow(np.squeeze(preds_test_t))
plt.show()