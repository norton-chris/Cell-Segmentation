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

from tensorflow.keras import mixed_precision
import ray
from ray.train import Trainer

import json

mixed_precision.set_global_policy('mixed_float16')
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

id = wandb.util.generate_id()
wandb.init(id=id, project='Cell-Segmentation', entity="nort", resume="allow")
#print(args)


# This environment variable will be set by Ray Train.
#tf_config = json.loads(os.environ['TF_CONFIG'])
#num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distribute.MultiWorkerMirroredStrategy()


#@ray.remote(num_gpus=.5)
def train_model():
    wandb.config.update(args)
    #global_batch_size = args.batch_size * num_workers
    #multi_worker_dataset = mnist_dataset(global_batch_size)



    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    gpu = tf.config.list_physical_devices('GPU')

    if gpu:
        try:
            for g in gpu:
                tf.config.experimental.set_memory_growth(g, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpu), "Physical GPUs", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    train = "TrainingDataset/output/train/" # change this to your local training dataset
    #val = "TrainingDataset/output/val/" # change this to your local validation set
    val = "TrainingDataset/output/val/"
    test = "TrainingDataset/output/test/" # change this to your local testing set

    #num_train_images = len([name for name in os.listdir(train) if os.path.isfile(os.path.join(train, name))])
    #num_val_images = len([name for name in os.listdir(val) if os.path.isfile(os.path.join(val, name))])

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
                            num_classes=1,
                            dropout_rate=args.dropout_rate,
                            activation=args.activation,
                            kernel_size=args.kernel_size)
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
            args.epochs) + "E" + "augment_300imgs_" + datetime.now().strftime("-%Y%m%d-%H.%M")
    else:
        file_name = args.model + str(args.input_shape) + str(args.n_filter) + "Flt" + str(
            args.learning_rate) + "lr" + str(
            args.epochs) + "E" + "_300imgs_" + datetime.now().strftime("-%Y%m%d-%H.%M")

    if wandb.run.resumed:
        try:
            model = keras.models.load_model(wandb.restore('model.h5').name)
        except:

            with strategy.scope():
                # Model building/compiling need to be within `strategy.scope()`.
                #multi_worker_model = build_and_compile_cnn_model()
                model = unet.create_model()
    else:
        with strategy.scope():
            model = unet.create_model()
    #print("model summary:", model.summary())

    # Fit model
    #tf.config.experimental_run_functions_eagerly(True)
    tf.config.run_functions_eagerly(True)

    #earlystopper = EarlyStopping(patience=15, verbose=1)


    checkpointer = ModelCheckpoint('h5_files/' + file_name + '.h5',
                                   verbose=0, monitor="val_dice_scoring", mode="max", save_best_only=True)

    log_dir = "logs/fit/" + file_name
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    training_generator = Batch_loader.BatchLoad(train, batch_size = args.batch_size, dim=dims, step=step,
                                                patching=args.patching, augment=args.augment)
    validation_generator = Batch_loader.BatchLoad(val, batch_size = args.batch_size, dim=dims, step=step, augment=True, validate=False)
    print("starting training")

    results = model.fit(training_generator, validation_data=validation_generator,
                                     epochs=args.epochs, use_multiprocessing=False, workers=8,
                                     callbacks=[wandb.save("model.h5"), checkpointer, tensorboard_callback,
                                                WandbCallback()])  # TqdmCallback(verbose=2), earlystopper

    #multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)

    print("results:", results)
    print("Evaluate:")
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
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.25,
        help="Layer dropout rate"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="selu",
        help="Activation function to use"
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
        help="Size of kernel"
    )

    args = parser.parse_args()

    # for i in range(0, 1):
    #     wandb.require("service")
    #     p = mp.Process(target=train_model, kwargs=dict(args=args))
    #     p.start()
    #     p.join()

    ray.init(num_cpus=args.batch_size, num_gpus=3)
    #ray.get([train_model.remote(args) for _ in range(2)]) # run 2 jobs on one gpu

    #train_model(args)

    trainer = Trainer(backend="tensorflow", num_workers=3)

    # For GPU Training, set `use_gpu` to True.
    # trainer = Trainer(backend="tensorflow", num_workers=4, use_gpu=True)

    trainer.start()
    trainer_results = trainer.run(train_model)
    trainer.shutdown()
    print(trainer_results)

