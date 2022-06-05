import os
import warnings
from datetime import datetime

import numpy as np
import random
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
import tensorflow as tf
import Models
from tqdm.keras import TqdmCallback
import Batcher_loader_nothread
import ray

ray.init(num_cpus=64, num_gpus=3)

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
print(tf.config.list_physical_devices('GPU'))

@ray.remote(num_gpus=.5)
def train_model():
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    IMG_CHANNELS = 1

    train = "TrainingDataset/correct_labels_subset/output/train/" # change this to your local training dataset
    #val = "TrainingDataset/output/val/" # change this to your local validation set
    val = "TrainingDataset/correct_labels_subset/output/val/"
    test = "TrainingDataset/TrainingDataset/output/test/" # change this to your local testing set

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

    dims = (512, 512, 1)
    step = 512
    unet = Models.UNET(n_filter=8,
                        input_dim=dims,
                        learning_rate=0.0002,
                        num_classes=1)

    model = unet.create_model()
    print("model summary:", model.summary())

    # Fit model
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
    ray.get([train_model.remote() for _ in range(6)])
