#----------------------------------------------------------------------------
# Created By  : Chris Norton
# ---------------------------------------------------------------------------
"""
This program will fit a model with the inputted dataset.
"""
# ---------------------------------------------------------------------------

# Built-in
import os

# 3rd Party Libs
import warnings
from datetime import datetime
import numpy as np
import random
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
import tensorflow as tf
import Models
from tqdm.keras import TqdmCallback
import ray

# Owned
import Batcher_loader_nothread
__author__ = "Chris Norton"
__maintainer__ = "Chris Norton"
__email__ = "cnorton@mtu.edu"
__status__ = "Dev"

# {code}
ray.init(num_cpus=8, num_gpus=1)
print(tf.config.list_physical_devices('GPU'))

def train_model():
    train = "TrainingDataset/correct_labels_subset/output/train/"  # change this to your local training dataset
    val = "TrainingDataset/correct_labels_subset/output/val/"  # change this to your local validation set

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    seed = 42
    random.seed = seed
    np.random.seed = seed

    dims = (512, 512, 1)
    step = 512
    unet = Models.UNET(n_filter=8,
                        input_dim=dims,
                        learning_rate=0.0002,
                        num_classes=1)

    model = unet.create_model()
    print("model summary:", model.summary())

    #tf.config.experimental_run_functions_eagerly(True)
    tf.config.run_functions_eagerly(True)

    #earlystopper = EarlyStopping(patience=15, verbose=1)
    file_name = "UNET512TIF32Flt1000E_300imgs_batchnorm_"
    checkpointer = ModelCheckpoint('h5_files/' + file_name + datetime.now().strftime("-%Y%m%d-%H.%M") + '.h5',
                                   verbose=0, save_best_only=False)

    log_dir = "logs/fit/" + file_name + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    input_shape = (512, 512, 1)
    training_generator = Batcher_loader_nothread.BatchLoad(train, batch_size=8, dim=input_shape, step=step, patching=False, augment=False)
    validation_generator = Batcher_loader_nothread.BatchLoad(val, batch_size=8, dim=input_shape, step=step, augment=False, validate=True)
    results = model.fit(training_generator, validation_data=validation_generator,
                        epochs=1000,  use_multiprocessing=False, workers=8,
                        callbacks=[checkpointer, tensorboard_callback]) #  TqdmCallback(verbose=2), earlystopper

    print("Evaluate")
    result = model.evaluate(training_generator)
    print(result)

if __name__ == "__main__":
    gpu = tf.config.list_physical_devices('GPU')

    if gpu:
        try:
            for g in gpu:
                tf.config.experimental.set_memory_growth(g, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpu), "Physical GPUs", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    train_model()
