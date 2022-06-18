#----------------------------------------------------------------------------
# Created By  : Chris Norton
# ---------------------------------------------------------------------------
"""
Models class file. Defines neural network models.
"""
# ---------------------------------------------------------------------------

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.convolutional import UpSampling2D
from keras.layers.merge import concatenate
from keras.engine.input_layer import Input
from keras.layers.convolutional import Conv2D
from keras.layers import BatchNormalization, Activation
from keras.engine.training import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow as tf

import Scoring

def convolution(filters, kernel_size, use_batchnorm=True, use_dropout=True, drop_rate=0.05,
                conv_name='conv', bn_name='bn', activation='selu', stride=1):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding='same', use_bias=not use_batchnorm, strides=stride)(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        if use_dropout:
            x = Dropout(drop_rate)(x)
        x = Activation(activation)(x)
        return x

    return layer


class UNetPlusPlus(object):

    def __init__(self, n_filter=16, input_dim=(512, 512, 1), learning_rate=3e-5, num_classes=1):
        self.n_filter = n_filter
        self.input = Input(input_dim)
        self.lr = learning_rate
        self.num_classes = num_classes

    def create_model(self):
        # ######### Frame 1 ######### #
        # X00 (Top layer, no up-sample)
        conv_x00 = convolution(self.n_filter, 3)(self.input)
        skip_x00 = convolution(self.n_filter, 3)(conv_x00)
        down_x00 = MaxPooling2D(pool_size=(2, 2))(skip_x00)

        # X10
        conv_x10 = convolution(self.n_filter * 2, 3)(down_x00)
        skip_x10 = convolution(self.n_filter * 2, 3)(conv_x10)
        up_x10 = convolution(self.n_filter, 3)(UpSampling2D(size=(2, 2))(skip_x10))
        down_x10 = MaxPooling2D(pool_size=(2, 2))(skip_x10)

        # X20
        conv_x20 = convolution(self.n_filter * 4, 3)(down_x10)
        skip_x20 = convolution(self.n_filter * 4, 3)(conv_x20)
        up_x20 = convolution(self.n_filter * 2, 3)(UpSampling2D(size=(2, 2))(skip_x20))
        down_x20 = MaxPooling2D(pool_size=(2, 2))(skip_x20)

        # X30
        conv_x30 = convolution(self.n_filter * 8, 3)(down_x20)
        skip_x30 = convolution(self.n_filter * 8, 3)(conv_x30)
        up_x30 = convolution(self.n_filter * 4, 3)(UpSampling2D(size=(2, 2))(skip_x30))
        down_x30 = MaxPooling2D(pool_size=(2, 2))(skip_x30)

        # X40 (Deepest point)
        conv_x40 = convolution(self.n_filter * 16, 3)(down_x30)
        skip_x40 = convolution(self.n_filter * 16, 3)(conv_x40)  # This skip connection will not be used later
        up_x40 = convolution(self.n_filter * 8, 3)(UpSampling2D(size=(2, 2))(skip_x40))

        # ######### Frame 2 ######### #

        # X01
        merge_x01 = concatenate([skip_x00, up_x10], axis=3)
        conv_x01 = convolution(self.n_filter, 3)(merge_x01)
        skip_x01 = convolution(self.n_filter, 3)(conv_x01)

        # X11
        merge_x11 = concatenate([skip_x10, up_x20], axis=3)
        conv_x11 = convolution(self.n_filter * 2, 3)(merge_x11)
        skip_x11 = convolution(self.n_filter * 2, 3)(conv_x11)
        up_x11 = convolution(self.n_filter, 3)(UpSampling2D(size=(2, 2))(skip_x11))

        # X21
        merge_x21 = concatenate([skip_x20, up_x30], axis=3)
        conv_x21 = convolution(self.n_filter * 4, 3)(merge_x21)
        skip_x21 = convolution(self.n_filter * 4, 3)(conv_x21)
        up_x21 = convolution(self.n_filter * 2, 3)(UpSampling2D(size=(2, 2))(skip_x21))

        # X31
        merge_x31 = concatenate([skip_x30, up_x40], axis=3)
        conv_x31 = convolution(self.n_filter * 8, 3)(merge_x31)
        skip_x31 = convolution(self.n_filter * 8, 3)(conv_x31)  # will not be used later
        up_x31 = convolution(self.n_filter * 4, 3)(UpSampling2D(size=(2, 2))(skip_x31))

        # ######### Frame 3 ######### #

        # X02
        merge_x02 = concatenate([skip_x00, skip_x01, up_x11], axis=3)
        conv_x02 = convolution(self.n_filter, 3)(merge_x02)
        skip_x02 = convolution(self.n_filter, 3)(conv_x02)

        # X12
        merge_x12 = concatenate([skip_x10, skip_x11, up_x21], axis=3)
        conv_x12 = convolution(self.n_filter * 2, 3)(merge_x12)
        skip_x12 = convolution(self.n_filter * 2, 3)(conv_x12)
        up_x12 = convolution(self.n_filter, 3)(UpSampling2D(size=(2, 2))(skip_x12))

        # X22
        merge_x22 = concatenate([skip_x20, skip_x21, up_x31], axis=3)
        conv_x22 = convolution(self.n_filter * 4, 3)(merge_x22)
        skip_x22 = convolution(self.n_filter * 4, 3)(conv_x22)  # will not be used later
        up_x22 = convolution(self.n_filter * 2, 3)(UpSampling2D(size=(2, 2))(skip_x22))

        # ######### Frame 4 ######### #

        # X03
        merge_x03 = concatenate([skip_x00, skip_x01, skip_x02, up_x12], axis=3)
        conv_x03 = convolution(self.n_filter, 3)(merge_x03)
        conv_x03 = BatchNormalization()(conv_x03)
        skip_x03 = convolution(self.n_filter, 3)(conv_x03)

        # X13
        merge_x13 = concatenate([skip_x10, skip_x11, skip_x12, up_x22], axis=3)
        conv_x13 = convolution(self.n_filter * 2, 3)(merge_x13)
        conv_x13 = BatchNormalization()(conv_x13)
        skip_x13 = convolution(self.n_filter * 2, 3)(conv_x13)

        up_x13 = convolution(self.n_filter, 3)(UpSampling2D(size=(2, 2))(skip_x13))

        # ######### Frame 5 ######### #

        # X04
        merge_x04 = concatenate([skip_x00, skip_x01, skip_x02, skip_x03, up_x13], axis=3)
        conv_x04 = convolution(self.n_filter, 3, use_batchnorm=True, use_dropout=False)(merge_x04)
        out = convolution(2, 3)(conv_x04)

        # ######### output ######### #
        output = Conv2D(self.num_classes, 1, activation='sigmoid')(out)

        model = Model(inputs=self.input, outputs=output)
        opt = Adam(learning_rate=self.lr)
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        model.compile(optimizer=opt, loss=Scoring.dice_plus_bce_loss, metrics=Scoring.dice_scoring)
        # model.summary()
        return model

class UNET(object):

    def __init__(self, n_filter=16,
                 input_dim=(512, 512, 1),
                 learning_rate=3e-5, num_classes=1,
                 dropout_rate=0.25,
                 activation="selu",
                 kernel_size = 3):
        self.n_filter = n_filter
        self.input = Input(input_dim)
        self.lr = learning_rate
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.kernel_size = kernel_size

    def create_model(self):
        # Level 1
        skip1 = convolution(self.n_filter, kernel_size=self.kernel_size,
                                      drop_rate=self.dropout_rate, activation=self.activation)(self.input)
        #Conv2D(self.n_filter, self.kernel_size, padding='same', use_bias=not use_batchnorm)(x)
        down1 = MaxPooling2D(pool_size=[2,2])(skip1)

        # level 2
        skip2 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                                      drop_rate=self.dropout_rate, activation=self.activation)(down1)
        down2 = MaxPooling2D(pool_size=[2,2])(skip2)

        # level 3
        skip3 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                                      drop_rate=self.dropout_rate, activation=self.activation)(down2)
        down3 = MaxPooling2D(pool_size=[2,2])(skip3)

        # level 4
        skip4 = convolution(self.n_filter * 8, kernel_size=self.kernel_size,
                                      drop_rate=self.dropout_rate, activation=self.activation)(down3)
        down4 = MaxPooling2D(pool_size=[2, 2])(skip4)

        # level 5. Deepest
        l5 = convolution(self.n_filter * 16, kernel_size=self.kernel_size,
                                   drop_rate=self.dropout_rate, activation=self.activation)(down4)

        # level 4
        concat4 = concatenate([UpSampling2D(size=[2, 2])(l5), skip4])
        l4 = convolution(self.n_filter * 8,  kernel_size=self.kernel_size, drop_rate=self.dropout_rate,
                                   activation=self.activation)(concat4)

        # level 3
        concat3 = concatenate([UpSampling2D(size=[2, 2])(l4), skip3])
        l3 = convolution(self.n_filter * 4,  kernel_size=self.kernel_size, drop_rate=self.dropout_rate,
                                   activation=self.activation)(concat3)

        # level 2
        concat2 = concatenate([UpSampling2D(size=[2, 2])(l3), skip2])
        l2 = convolution(self.n_filter * 2,  kernel_size=self.kernel_size, drop_rate=self.dropout_rate,
                                   activation=self.activation)(concat2)

        # level 1
        concat1 = concatenate([UpSampling2D(size=[2, 2])(l2), skip1])
        l1 = convolution(self.n_filter, kernel_size=self.kernel_size, drop_rate=self.dropout_rate,
                                   activation=self.activation)(concat1)

        output = Conv2D(1, [1, 1], activation='sigmoid')(l1)

        model = Model(inputs=self.input, outputs=output)

        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.lr,
            decay_steps=10000,
            decay_rate=0.9)
        opt = Adam(learning_rate=lr_schedule)
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        model.compile(optimizer=opt, loss=Scoring.dice_plus_bce_loss, metrics=Scoring.dice_scoring)
        # model.summary()
        return model
