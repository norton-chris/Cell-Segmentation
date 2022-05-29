import os
import warnings
from datetime import datetime

import keras
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

import multiprocessing as mp


def train_model(args):
    id = wandb.util.generate_id()
    wandb.init(id=id, project='Cell-Segmentation', entity="nort", resume="allow", group="GRP")
    wandb.config.update(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    print(tf.config.list_physical_devices('GPU'))

    train = "TrainingDataset/correct_labels_subset/output/train/" # change this to your local training dataset
    #val = "TrainingDataset/output/val/" # change this to your local validation set
    val = "TrainingDataset/correct_labels_subset/output/val/"
    test = "TrainingDataset/TrainingDataset/output/test/" # change this to your local testing set

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    seed = 42
    random.seed = seed
    np.random.seed = seed

    if args.input_shape == 512:
        dims = (512, 512, 1)
        step = 512
    elif args.input_shape == 256:
        dims = (256, 256, 1)
        step = 256
    elif args.input_shape == 128:
        dims = (128, 128, 1)
        step = 128
    else:
        print("No input shape given.. defaulting to 512x512..")
        dims = (512, 512, 1)
        step = 512

    if args.model == "unet":
        unet = Models.UNET(n_filter=args.n_filter,
                            input_dim=dims,
                            learning_rate=args.learning_rate,
                            num_classes=1)
    elif args.model == "unet++":
        unet = Models.UNetPlusPlus(n_filter=args.n_filter,
                           input_dim=dims,
                           learning_rate=args.learning_rate,
                           num_classes=1)
    else:
        print("Error: No model set.. exiting..")
        exit(-1)

    if args.augment:
        file_name = args.model + str(args.input_shape) + str(args.n_filter) + "Flt" + str(
            args.learning_rate) + "lr" + str(
            args.epochs) + "E" + "augment_200imgs_batchnorm_" + datetime.now().strftime("-%Y%m%d-%H.%M")
    else:
        file_name = args.model + str(args.input_shape) + str(args.n_filter) + "Flt" + str(
            args.learning_rate) + "lr" + str(
            args.epochs) + "E" + "_200imgs_batchnorm" + datetime.now().strftime("-%Y%m%d-%H.%M")

    if wandb.run.resumed:
        try:
            model = keras.models.load_model(wandb.restore('model.h5').name)
        except:
            model = unet.create_model()
    else:
        model = unet.create_model()
    #print("model summary:", model.summary())

    # Fit model
    #tf.config.experimental_run_functions_eagerly(True)
    tf.config.run_functions_eagerly(True)

    #earlystopper = EarlyStopping(patience=15, verbose=1)


    checkpointer = ModelCheckpoint('h5_files/' + file_name + '.h5',
                                   verbose=0, save_best_only=False)

    log_dir = "logs/fit/" + file_name
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    training_generator = Batch_loader.BatchLoad(train, batch_size = args.batch_size, dim=dims, step=step,
                                                patching=args.patching, augment=args.augment)
    validation_generator = Batch_loader.BatchLoad(train, batch_size = args.batch_size, dim=dims, step=step, augment=False)
    results = model.fit(training_generator, validation_data=validation_generator,
                        epochs=args.epochs,  use_multiprocessing=True, workers=8,
                        callbacks=[wandb.save("model.h5"), checkpointer, tensorboard_callback, WandbCallback(
                        monitor="val_dice_scoring", verbose=0, mode="max", save_weights_only=(False),
                        log_weights=(False), log_gradients=(False), save_model=(True),
                        training_data=training_generator, validation_data=validation_generator, labels=[], predictions=5,
                        generator=None, input_type="image", output_type="segmentation_mask", log_evaluation=(True),
                        validation_steps=None, class_colors=([0,0,0], [255,255,255]), log_batch_frequency=None,
                        log_best_prefix="best_", save_graph=(True), validation_indexes=None,
                        validation_row_processor=None, prediction_row_processor=None,
                        infer_missing_processors=(True), log_evaluation_frequency=0
                                                                                    )
                                    ]
                        ) #  TqdmCallback(verbose=2), earlystopper

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
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        help="Network to use"
    )
    parser.add_argument(
        "--input_shape",
        type=int,
        default=512,
        help="Model input shape"
    )
    parser.add_argument(
        "--augment",
        type=bool,
        default=False,
        help="Use image augmentation"
    )
    parser.add_argument(
        "--patching",
        type=bool,
        default=True,
        help="Use image patching (True) or image resizing (False)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="Cell-Segmentation",
        help="Name of project"
    )

    args = parser.parse_args()

    wandb.require("service")
    p = mp.Process(target=train_model, kwargs=dict(args=args))
    p.start()
    p.join()

