
#----------------------------------------------------------------------------
# Created By  : Chris Norton
# ---------------------------------------------------------------------------
"""
This program will fit a model with the inputted dataset.
It uses wandb to use hyper-parametrization and there is a single
GPU that is specified. This is the main program used currently.
"""
# ---------------------------------------------------------------------------

# Built-in
import os

# 3rd Party Libs
import time
import warnings
from datetime import datetime
import keras
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
import Models
from tqdm.keras import TqdmCallback
import argparse
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model, load_model
import ray
from ray.train import Trainer
import cv2
import matplotlib.pyplot as plt
import image_similarity_measures
from image_similarity_measures.quality_metrics import rmse, psnr, fsim
import multiprocessing

# Owned
import Batch_loader
from Patcher import Patcher
import Batch_loader
from Random_patcher import Random_patcher
from Unpatcher import Unpatcher
from Random_unpatcher import Random_unpatcher
import Scoring
from WandB_Training_Visualization import WandbValidationVisualization
__author__ = "Chris Norton"
__maintainer__ = "Chris Norton"
__email__ = "cnorton@mtu.edu"
__status__ = "Dev"

# {code}
mixed_precision.set_global_policy('mixed_float16')
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
num_cores = multiprocessing.cpu_count()
print("num cores:", num_cores)


def normalize_image(input_block):
    block = input_block.copy()
    vol_max, vol_min = block.max(), block.min()
    if vol_max != vol_min:
        block = (block - vol_min) / (vol_max - vol_min)
    return block

def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.tif'):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Load as 16-bit image
            img = img.astype('float32')  # Convert to float to prepare for normalization
            img = normalize_image(img)  # Apply your normalization function
            img = np.expand_dims(img, axis=-1)  # Add channel dimension if needed
            images.append(img)
    return np.array(images)

def train_model(args):
    id = wandb.util.generate_id()
    run = wandb.init(id=id, project='Cell-Segmentation', entity="nort", resume="allow")
    wandb.config.update(args)

    gpu = tf.config.list_physical_devices('GPU')
    if gpu:
        try:
            for g in gpu:
                tf.config.experimental.set_memory_growth(g, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpu), "Physical GPUs", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    train = "TrainingDataset/data_subset/323_subset/output/train/" # change this to your local training dataset
    val = "TrainingDataset/data_subset/323_subset/output/val/"

    num_train_imgs = len(os.listdir(train))
    num_val_imgs = len(os.listdir(val))

    # artifact = wandb.Artifact('Cells', type='dataset')
    # artifact.add_dir("TrainingDataset/data_subset/323_subset/output/")  # Adds multiple files to artifact
    # run.log_artifact(artifact)

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
    elif args.model == "unet_multi":
        unet = Models.UNET_multiple_image_size(
            n_filter=args.n_filter,
            input_dim=dims,
            learning_rate=args.learning_rate,
            num_classes=1,
            dropout_rate=args.dropout_rate,
            activation=args.activation,
            depth=args.depth,
            use_batchnorm=args.use_batchnorm,
            dilation_rate=args.dilation_rate,
            kernel_size=args.kernel_size
        )
    elif args.model == "unet++":
        unet = Models.UNetPlusPlus(n_filter=args.n_filter,
                           input_dim=dims,
                           learning_rate=args.learning_rate,
                           num_classes=1,
                           dropout_rate=args.dropout_rate,
                           activation=args.activation,
                           kernel_size=args.kernel_size)
    elif args.model == "cbam":
        unet = Models.UNetPlusPlus_CBAM(n_filter=args.n_filter,
                           input_dim=dims,
                           learning_rate=args.learning_rate,
                           num_classes=1,
                           dropout_rate=args.dropout_rate,
                           activation=args.activation,
                           kernel_size=args.kernel_size)
    else:
        print("Error: No model set.. exiting..")
        exit(-1)

    if args.augment:
        file_name = args.model + str(args.input_shape) + "shp" + str(args.n_filter) + "Flt" + str(
            round(args.learning_rate, 2)) + "lr" + str(
            args.epochs) + "E" + "augment_" + str(num_train_imgs) + "imgs" + datetime.now().strftime("-%Y%m%d-%H:%M")
    else:
        file_name = args.model + str(args.input_shape) + "shp" + str(args.n_filter) + "Flt" + str(
            round(args.learning_rate, 2)) + "lr" + str(
            args.epochs) + "E_" + str(num_train_imgs) + "imgs" + datetime.now().strftime("-%Y%m%d-%H:%M")
    if wandb.run.resumed:
        try:
            model = keras.models.load_model(wandb.restore('model.h5').name)
        except:
            model = unet.create_model()
    else:
        model = unet.create_model()

    tf.config.run_functions_eagerly(True)

    earlystopper = EarlyStopping(patience=15, verbose=1)
    checkpointer = ModelCheckpoint('h5_files/' + file_name + '.h5',
                                   verbose=0, monitor="val_dice_scoring", mode="max", save_best_only=True)

    log_dir = "logs/fit/" + file_name
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    training_generator = Batch_loader.BatchLoad(train, batch_size=args.batch_size, dim=dims, step=step,
                                                patching=args.patching, augment=args.augment, multiple_sizes=args.multiple_sizes)
    validation_generator = Batch_loader.BatchLoad(val, batch_size=args.batch_size, dim=dims, step=step, augment=False,
                                                  validate=True, multiple_sizes=args.multiple_sizes)
    print("starting training")

    val_images = load_images_from_directory(val + "Images")
    val_labels = load_images_from_directory(val + "Labels")

    val_viz_callback = WandbValidationVisualization(val_images, val_labels, frequency=1)

    results = model.fit(training_generator, validation_data=validation_generator,
                                     epochs=args.epochs, use_multiprocessing=False, workers=num_cores,
                                     callbacks=[wandb.save("model.h5"), checkpointer, tensorboard_callback,
                                                WandbCallback(), val_viz_callback])  # TqdmCallback(verbose=2), earlystopper
    print("results:", results)
    print("Evaluate:")
    result = model.evaluate(training_generator)
    if args.multiple_sizes:
        file_name += "_multi"
    model_file = "h5_files/" + str(file_name) + ".h5"
    model.save(model_file)
    print(result)

    # model_artifact = wandb.Artifact(
    #     "trained_model", type="model",
    #     description="Model trained on" + file_name
    # )
    #
    # model_artifact.add_file("h5_files/" + str(file_name) + ".h5")
    wandb.save("h5_files" + str(file_name) + ".h5")

    #run.log_artifact(model_artifact)


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


    # After training is complete, evaluate on the test set
    print("Evaluating on test set...")
    test_dir = "TrainingDataset/data_subset/323_subset/output/test/Images/"  # Path to test images
    useLabels = True  # set to true if you have a folder called Labels inside test (the above variable)
    model_file = "h5_files/" + str(file_name) + ".h5"  # Path to the saved model

    # Load the trained model
    model = load_model(model_file,
                       custom_objects={'dice_plus_bce_loss': Scoring.dice_plus_bce_loss,
                                       'dice_scoring': Scoring.dice_scoring})

    total_fsim_score = 0
    num_images = 0

    # Iterate over the test images
    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path, -1).astype("float32")
        img = normalize_image(img)  # Normalize the image
        img = img.reshape(1, *img.shape, 1)  # Add batch dimension

        # Predict on the full image
        preds = model.predict(img)

        # Apply a confidence threshold to the predictions
        preds_thresholded = (preds > 0.5).astype(np.uint8)

        # Remove the batch dimension from the predictions
        preds_thresholded_reshape = np.squeeze(preds_thresholded, axis=0)

        # Optionally, load the corresponding label for accuracy assessment
        if useLabels:
            label_path = os.path.join(test_dir, "../Labels/", img_name)
            label = cv2.imread(label_path, -1)
            label = label.reshape(*label.shape, 1)  # Add channel dimension

        # Calculate FSIM score
        if useLabels:
            fsim_score = fsim(label, preds_thresholded_reshape)
            total_fsim_score += fsim_score
            num_images += 1

        # Visualization
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img[0, ..., 0], cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        if useLabels:
            plt.subplot(1, 3, 2)
            plt.imshow(label[..., 0], cmap='gray')
            plt.title("Ground Truth")
            plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(preds_thresholded[0, ..., 0], cmap='gray')
        plt.title(f"Prediction - FSIM: {fsim_score:.4f}")
        plt.axis('off')

        plt.show()
        plt.close(fig)

    # Calculate and print the average FSIM score
    average_fsim = total_fsim_score / num_images if num_images > 0 else 0
    print(f"Average FSIM Score: {average_fsim}")
    wandb.log({"average_fsim": average_fsim})




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
        default=8,
        help="number of filters")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Size of batch"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet_multi",
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
        "--use_batchnorm",
        type=str,
        default="True",
        help="Batch Normalization"
    )
    parser.add_argument(
        "--depth",
        type=str,
        default="5",
        help="Depth on UNET (only supported on unet-multi"
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
        help="Size of kernel"
    )
    parser.add_argument(
        "--dilation_rate",
        type=int,
        default=1,
        help="Size of kernel"
    )
    parser.add_argument(
        "--multiple_sizes",
        type=bool,
        default=False,
        help="Use multiple image sizes"
    )

    args = parser.parse_args()

    # for i in range(0, 1):
    #     wandb.require("service")
    #     p = mp.Process(target=train_model, kwargs=dict(args=args))
    #     p.start()
    #     p.join()

    ray.init(num_cpus=args.batch_size, num_gpus=1)
    with tf.device('/device:GPU:0'): # change to the gpu you specify
        train_model(args)

