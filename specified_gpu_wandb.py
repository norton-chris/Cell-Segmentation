
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


# Owned
import Batch_loader
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
mixed_precision.set_global_policy('mixed_float16')
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

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
    elif args.model == "unet++":
        unet = Models.UNetPlusPlus(n_filter=args.n_filter,
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
                                     epochs=args.epochs, use_multiprocessing=False, workers=4,
                                     callbacks=[wandb.save("model.h5"), checkpointer, tensorboard_callback,
                                                WandbCallback()])  # TqdmCallback(verbose=2), earlystopper
    print("results:", results)
    print("Evaluate:")
    result = model.evaluate(training_generator)
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

    ################################# EDIT THE LINE BELOW ###############################
    test = "TrainingDataset/data_subset/323_subset/output/test/"  ## EDIT THIS LINE
    useLabels = True  # set to true if you have a folder called Labels inside test (the above variable)

    # useLabels can be useful for seeing the accuracy.
    ################################# EDIT THE LINE ABOVE ###############################

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
    test = test + "Images/"  # Uncomment if you have a folder inside called Images
    dims = args.input_shape
    step = args.input_shape
    # Predict on patches
    while True: # try loop
        try:
            model = load_model(model_file,
                       custom_objects={'dice_plus_bce_loss': Scoring.dice_plus_bce_loss,
                                       'dice_scoring': Scoring.dice_scoring})
            break
        except Exception as e:
            time.sleep(10)
            print(e)
            print("model_file:", model_file)
            continue
    # load test patches
    images = np.zeros((len(os.listdir(test)), dims, dims, 1), dtype="float32")  # define the numpy array for the batch
    masks = np.zeros((len(os.listdir(test)), dims, dims, 1), dtype=bool)
    resize = np.zeros((1, dims, dims, 1), dtype=int)

    average_fsim = 0
    i = 0
    print("total image shape:", images.shape)
    vis_table = wandb.Table(columns=["image"])
    for path in os.listdir(test):  # Loop over Images in Directory
        print("loop", test + path)
        img = cv2.imread(test + path, -1).astype("float32")
        if useLabels:
            lab = cv2.imread(test + "../Labels/" + path, -1)  # HERE'S THE LINE THE READS THE LABELS

        # batch_size = int(img.shape[0] / step) * int(img.shape[1] / step)
        # if not useLabels:
        #     patcher_img = Patcher(img, batch_size=batch_size, input_shape=(dims, dims, 1), step=step)
        # else:
        #     patcher_img = Patcher(img, lab, batch_size=batch_size, input_shape=(dims, dims, 1), step=step)
        # images, masks, row, col = patcher_img.patch_image()
        # print("1 image shape:", images.shape)
        # preds_test = model.predict(images, verbose=1)
        #
        # # Predicting resized images
        # # resized = cv2.resize(img, (dims, dims))
        # # resize = resized.reshape(1, step, step, 1)
        #
        # # Predicting full sized images
        # # preds_full_image = model.predict(resize)
        # preds_test = (preds_test > 0.2)  # .astype(np.uint8) # showing predictions with
        # # preds_full_image = (preds_full_image > 0.4).astype(np.uint8)

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
        lab = lab.reshape(lab.shape[0], lab.shape[1], 1)
        plt.axis('off')
        plt.title("label")

        unpatcher = Random_unpatcher(img, img_name=test + path, model=model, input_shape=(dims, dims, 1), step=dims,
                                     num_crop=500)
        full_pred_image = unpatcher.efficient_random_unpatch()

        # int_img = np.array(full_pred_image, dtype="uint8")
        # grey = int_img[:, :, 0]
        # ret, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # kernel = np.ones((3, 3), np.uint8)
        # remove_noise = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        fig.add_subplot(1, 3, 3)
        plt.imshow(full_pred_image)
        # plt.imshow(preds_full_image)
        plt.axis('off')
        plt.title("prediction")

        plt.subplots_adjust(wspace=.05, hspace=.05, left=.01, right=.99, top=.99, bottom=.01)
        full_pred_image = full_pred_image.reshape(full_pred_image.shape[0], full_pred_image.shape[1], 1)
        fsim_score = fsim(lab, full_pred_image)
        average_fsim += fsim_score
        print(fsim_score)
        text = "fsim_score: " + str(fsim_score)
        fig.text(.5, .05, text, ha='center')

        plt.savefig('data.png')
        #plt.show()
        plt.close()

        fsim_score = fsim(lab, full_pred_image)
        average_fsim += fsim_score

        #plt.show()
        out = cv2.imread('data.png')
        img = wandb.Image(out)
        # img = wandb.Image(PIL.Image.fromarray(out.get_image()[:, :, ::-1]))
        vis_table.add_data(img)
        #vis_table.add_data(fsim_score)
        i += 1
    run.log({"infer_table": vis_table})
    average_fsim /= i
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

    ray.init(num_cpus=args.batch_size, num_gpus=1)
    with tf.device('/device:GPU:0'): # change to the gpu you specify
        train_model(args)

