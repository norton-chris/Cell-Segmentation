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

import argparse
import wandb
from wandb.keras import WandbCallback

def train_model(args):
    wandb.init(project='Cell-Segmentation', entity="nort")

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    print(tf.config.list_physical_devices('GPU'))

    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    IMG_CHANNELS = 1

    train = "TrainingDataset/correct_labels_subset/output/train/" # change this to your local training dataset
    #val = "TrainingDataset/output/val/" # change this to your local validation set
    val = "TrainingDataset/correct_labels_subset/output/val/"
    test = "TrainingDataset/TrainingDataset/output/test/" # change this to your local testing set

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    seed = 42
    random.seed = seed
    np.random.seed = seed

    dims = (512, 512, 1)
    step = 512
    unet = Models.UNET(n_filter=args.n_filter,
                        input_dim=dims,
                        learning_rate=args.learning_rate,
                        num_classes=1)

    model = unet.create_model()
    print("model summary:", model.summary())

    # Fit model
    #tf.config.experimental_run_functions_eagerly(True)
    tf.config.run_functions_eagerly(True)

    earlystopper = EarlyStopping(patience=15, verbose=1)
    file_name = "UNET++512TIF32Flt1000E_200imgs_batchnorm_-20220526-21.07.h5"
    checkpointer = ModelCheckpoint('h5_files/' + file_name + datetime.now().strftime("-%Y%m%d-%H.%M") + '.h5',
                                   verbose=0, save_best_only=False)

    log_dir = "logs/fit/" + file_name + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    input_shape = (512, 512, 1)
    training_generator = Batch_loader.BatchLoad(train, batch_size = args.batch_size, dim=input_shape, step=step, patching=False, validation=False)
    validation_generator = Batch_loader.BatchLoad(train, batch_size = args.batch_size, dim=input_shape, step=step, validation=False)
    results = model.fit(training_generator, validation_data=validation_generator,
                        epochs=args.epochs,  use_multiprocessing=True, workers=8,
                        callbacks=[earlystopper, checkpointer, tensorboard_callback, WandbCallback()]) #  TqdmCallback(verbose=2), earlystopper

    print("Evaluate")
    result = model.evaluate(training_generator)
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=20,
        help="number of training epochs (passes through full training data)")
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.0001,
        help="learning rate")
    parser.add_argument(
        "--n_filter",
        type=int,
        default=32,
        help="number of filters")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Size of batch"
    )

    args = parser.parse_args()

    train_model(args)
