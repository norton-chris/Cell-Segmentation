import os
import warnings
from datetime import datetime

import numpy as np
import random
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
import tensorflow as tf
import Models
from tqdm.keras import TqdmCallback
import Batch_loader

print(tf.config.list_physical_devices('GPU'))

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1

train = "E:/Han Project/TrainingDataset/clean_dataset/TrainingDataset/" # change this to your local training dataset
val = "E:/Han Project/TrainingDataset/clean_dataset/TrainingDataset/" # change this to your local validation set
test = "E:/Han Project/TrainingDataset/TrainingDataset/output/test/" # change this to your local testing set

TEST_PATH = 'test_images/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#  train, test, test_size=0.33, random_state=42)

# count = 0
# i = 0
# for path in os.listdir(train + "Images/"):
#     if not os.path.isfile(train + "Images/"):
#         continue
#     if not (train + "Images/" + path == train + "Label"):
#         print("not same path")
#         print("bad image:", path)
#         count += 1
#     i+=1
# print("number of images:", i)
# print("number of bad image:", count)

dims = (128, 128, 1)
step = 128

unet = Models.UNET(n_filter=16,
                    input_dim=dims,
                    learning_rate=0.0001,
                    num_classes=1)
model = unet.create_model()
print("model summary:", model.summary())

# Fit model
tf.config.experimental_run_functions_eagerly(True)

earlystopper = EarlyStopping(patience=15, verbose=1)
checkpointer = ModelCheckpoint('h5_files/train_UNet512TIF50E' + datetime.now().strftime("-%Y%m%d-%H.%M") + '.h5',
                               verbose=0, save_best_only=False)

log_dir = "logs/fit/UNet_" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
training_generator = Batch_loader.BatchLoad(train, batch_size = 8, dim=dims, step=step)
validation_generator = Batch_loader.BatchLoad(train, batch_size = 8, dim=dims, step=step)
results = model.fit(training_generator, validation_data=validation_generator,
                    epochs=50, # multiprocessing =True, workers = 8,
                    callbacks=[earlystopper, checkpointer,tensorboard_callback]) #  TqdmCallback(verbose=2)

print("Evaluate")
result = model.evaluate(training_generator)
print(result)