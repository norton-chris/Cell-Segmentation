# ----------------------------------------------------------------------------
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
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import backend as K
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
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Add, Permute, \
        Reshape
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

class CBAM(tf.keras.layers.Layer):
    def __init__(self, ratio=16, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.global_avg_pool = GlobalAveragePooling2D()
        self.global_max_pool = GlobalMaxPooling2D()

        self.dense1 = Dense(input_shape[-1] // self.ratio, activation='relu')
        self.dense2 = Dense(input_shape[-1], activation='sigmoid')

        self.conv1 = Conv2D(1, self.kernel_size, padding='same', activation='sigmoid', use_bias=False)

    def call(self, inputs, **kwargs):
        # Channel attention
        avg_pool = self.global_avg_pool(inputs)
        max_pool = self.global_max_pool(inputs)

        avg_pool = self.dense2(self.dense1(avg_pool))
        max_pool = self.dense2(self.dense1(max_pool))

        channel_attention = add([avg_pool, max_pool])
        channel_attention = tf.expand_dims(tf.expand_dims(channel_attention, 1), 1)

        # Spatial attention
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

        spatial_attention = self.conv1(tf.concat([avg_pool, max_pool], axis=-1))

        # Combine channel and spatial attention
        attention = multiply([channel_attention, inputs])
        attention = multiply([spatial_attention, attention])

        return attention


class UNetPlusPlus(object):

    def __init__(self, n_filter=16,
                 input_dim=(512, 512, 1),
                 learning_rate=3e-5,
                 num_classes=1,
                 dropout_rate=0.05,
                 kernel_size=3,
                 activation="selu",
                 epochs=50000,
                 batch_size=2):
        self.n_filter = n_filter
        self.input = Input(input_dim)
        self.lr = learning_rate
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size

    def create_model(self):
        # ######### Frame 1 ######### #
        # X00 (Top layer, no up-sample)
        conv_x00 = convolution(self.n_filter, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(self.input)
        skip_x00 = convolution(self.n_filter, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x00)
        down_x00 = MaxPooling2D(pool_size=(2, 2))(skip_x00)

        # X10
        conv_x10 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(down_x00)
        skip_x10 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x10)
        up_x10 = convolution(self.n_filter, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x10))
        down_x10 = MaxPooling2D(pool_size=(2, 2))(skip_x10)

        # X20
        conv_x20 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(down_x10)
        skip_x20 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x20)
        up_x20 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x20))
        down_x20 = MaxPooling2D(pool_size=(2, 2))(skip_x20)

        # X30
        conv_x30 = convolution(self.n_filter * 8, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(down_x20)
        skip_x30 = convolution(self.n_filter * 8, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x30)
        up_x30 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x30))
        down_x30 = MaxPooling2D(pool_size=(2, 2))(skip_x30)

        # X40 (Deepest point)
        conv_x40 = convolution(self.n_filter * 16, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(down_x30)
        skip_x40 = convolution(self.n_filter * 16, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(
            conv_x40)  # This skip connection will not be used later
        up_x40 = convolution(self.n_filter * 8, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x40))

        # ######### Frame 2 ######### #

        # X01
        merge_x01 = concatenate([skip_x00, up_x10], axis=3)
        conv_x01 = convolution(self.n_filter, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x01)
        skip_x01 = convolution(self.n_filter, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x01)

        # X11
        merge_x11 = concatenate([skip_x10, up_x20], axis=3)
        conv_x11 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x11)
        skip_x11 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x11)
        up_x11 = convolution(self.n_filter, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x11))

        # X21
        merge_x21 = concatenate([skip_x20, up_x30], axis=3)
        conv_x21 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x21)
        skip_x21 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x21)
        up_x21 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x21))

        # X31
        merge_x31 = concatenate([skip_x30, up_x40], axis=3)
        conv_x31 = convolution(self.n_filter * 8, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x31)
        skip_x31 = convolution(self.n_filter * 8, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(
            conv_x31)  # will not be used later
        up_x31 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x31))

        # ######### Frame 3 ######### #

        # X02
        merge_x02 = concatenate([skip_x00, skip_x01, up_x11], axis=3)
        conv_x02 = convolution(self.n_filter, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x02)
        skip_x02 = convolution(self.n_filter, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x02)

        # X12
        merge_x12 = concatenate([skip_x10, skip_x11, up_x21], axis=3)
        conv_x12 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x12)
        skip_x12 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x12)
        up_x12 = convolution(self.n_filter, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x12))

        # X22
        merge_x22 = concatenate([skip_x20, skip_x21, up_x31], axis=3)
        conv_x22 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x22)
        skip_x22 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(
            conv_x22)  # will not be used later
        up_x22 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x22))

        # ######### Frame 4 ######### #

        # X03
        merge_x03 = concatenate([skip_x00, skip_x01, skip_x02, up_x12], axis=3)
        conv_x03 = convolution(self.n_filter, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x03)
        conv_x03 = BatchNormalization()(conv_x03)
        skip_x03 = convolution(self.n_filter, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x03)

        # X13
        merge_x13 = concatenate([skip_x10, skip_x11, skip_x12, up_x22], axis=3)
        conv_x13 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x13)
        conv_x13 = BatchNormalization()(conv_x13)
        skip_x13 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x13)

        up_x13 = convolution(self.n_filter, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x13))

        # ######### Frame 5 ######### #

        # X04
        merge_x04 = concatenate([skip_x00, skip_x01, skip_x02, skip_x03, up_x13], axis=3)
        conv_x04 = convolution(self.n_filter, kernel_size=self.kernel_size, drop_rate=self.dropout_rate,
                               activation=self.activation, use_batchnorm=True, use_dropout=False)(merge_x04)
        out = convolution(2, kernel_size=self.kernel_size, drop_rate=self.dropout_rate,
                          activation=self.activation)(conv_x04)

        # ######### output ######### #
        if self.num_classes == 1:
            output = Conv2D(self.num_classes, 1, activation='sigmoid')(out)
        else:
            output = Conv2D(self.num_classes, 1, activation='softmax')(out)

        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.lr,
            decay_steps=10000,
            decay_rate=0.9)

        model = Model(inputs=self.input, outputs=output)
        opt = Adam(learning_rate=lr_schedule)
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
                 kernel_size=3,
                 epochs=50000,
                 batch_size=2):
        self.n_filter = n_filter
        self.input = Input(input_dim)
        self.lr = learning_rate
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.batch_size = batch_size

    def create_model(self):
        # Level 1
        skip1 = convolution(self.n_filter, kernel_size=self.kernel_size,
                            drop_rate=self.dropout_rate, activation=self.activation)(self.input)
        # Conv2D(self.n_filter, self.kernel_size, padding='same', use_bias=not use_batchnorm)(x)
        down1 = MaxPooling2D(pool_size=[2, 2])(skip1)

        # level 2
        skip2 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                            drop_rate=self.dropout_rate, activation=self.activation)(down1)
        down2 = MaxPooling2D(pool_size=[2, 2])(skip2)

        # level 3
        skip3 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                            drop_rate=self.dropout_rate, activation=self.activation)(down2)
        down3 = MaxPooling2D(pool_size=[2, 2])(skip3)

        # level 4
        skip4 = convolution(self.n_filter * 8, kernel_size=self.kernel_size,
                            drop_rate=self.dropout_rate, activation=self.activation)(down3)
        down4 = MaxPooling2D(pool_size=[2, 2])(skip4)

        # level 5. Deepest
        l5 = convolution(self.n_filter * 16, kernel_size=self.kernel_size,
                         drop_rate=self.dropout_rate, activation=self.activation)(down4)

        # level 4
        concat4 = concatenate([UpSampling2D(size=[2, 2])(l5), skip4])
        l4 = convolution(self.n_filter * 8, kernel_size=self.kernel_size, drop_rate=self.dropout_rate,
                         activation=self.activation)(concat4)

        # level 3
        concat3 = concatenate([UpSampling2D(size=[2, 2])(l4), skip3])
        l3 = convolution(self.n_filter * 4, kernel_size=self.kernel_size, drop_rate=self.dropout_rate,
                         activation=self.activation)(concat3)

        # level 2
        concat2 = concatenate([UpSampling2D(size=[2, 2])(l3), skip2])
        l2 = convolution(self.n_filter * 2, kernel_size=self.kernel_size, drop_rate=self.dropout_rate,
                         activation=self.activation)(concat2)

        # level 1
        concat1 = concatenate([UpSampling2D(size=[2, 2])(l2), skip1])
        l1 = convolution(self.n_filter, kernel_size=self.kernel_size, drop_rate=self.dropout_rate,
                         activation=self.activation)(concat1)

        if self.num_classes == 1:
            output = Conv2D(1, [1, 1], activation='sigmoid')(l1)
        else:
            output = Conv2D(num_classes, [1, 1], activation='softmax')(l1)

        model = Model(inputs=self.input, outputs=output)

        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.lr,
            decay_steps=10000,
            decay_rate=0.9)
        opt = Adam(learning_rate=lr_schedule)
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        if self.num_classes == 1:
            model.compile(optimizer=opt, loss=Scoring.dice_plus_bce_loss, metrics=Scoring.dice_scoring)
        else:
            model.compile(optimizer=opt, loss=Scoring.dice_plus_cce_loss, metrics=Scoring.dice_scoring)
        # model.summary()
        return model

class UNetPlusPlus_CBAM(object):

    def __init__(self, n_filter=16,
                 input_dim=(512, 512, 1),
                 learning_rate=3e-5,
                 num_classes=1,
                 dropout_rate=0.05,
                 kernel_size=3,
                 activation="selu",
                 epochs=50000,
                 batch_size=2):
        self.n_filter = n_filter
        self.input = Input(input_dim)
        self.lr = learning_rate
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size

    def ChannelAttention(self, input_feature, ratio=8):
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = input_feature.shape[channel_axis]

        shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,
                                 bias_initializer='zeros')
        shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

        avg_pool = GlobalAveragePooling2D()(input_feature)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        assert avg_pool.shape[1:] == (1, 1, channel)
        avg_pool = shared_layer_one(avg_pool)
        avg_pool = shared_layer_two(avg_pool)

        max_pool = GlobalMaxPooling2D()(input_feature)
        max_pool = Reshape((1, 1, channel))(max_pool)
        assert max_pool.shape[1:] == (1, 1, channel)
        max_pool = shared_layer_one(max_pool)
        max_pool = shared_layer_two(max_pool)

        cbam_feature = Add()([avg_pool, max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)

        return Multiply()([input_feature, cbam_feature])

    def SpatialAttention(self, input_feature, kernel_size=7):
        if K.image_data_format() == "channels_first":
            channel = input_feature.shape[1]
            cbam_feature = Permute((2, 3, 1))(input_feature)
        else:
            channel = input_feature.shape[-1]
            cbam_feature = input_feature

        avg_pool = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(cbam_feature)
        max_pool = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(cbam_feature)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, padding='same', activation='sigmoid',
                              kernel_initializer='he_normal', use_bias=False)(concat)

        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)

        return Multiply()([input_feature, cbam_feature])

    def cbam_block(self, input_feature, ratio=8, kernel_size=7):
        x = self.ChannelAttention(input_feature, ratio)
        x = self.SpatialAttention(x, kernel_size)
        return x

    def create_model(self):
        # ######### Frame 1 ######### #
        # X00 (Top layer, no up-sample)
        conv_x00 = convolution(self.n_filter, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(self.input)
        conv_x00 = self.cbam_block(conv_x00)
        skip_x00 = convolution(self.n_filter, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x00)
        down_x00 = MaxPooling2D(pool_size=(2, 2))(skip_x00)

        # X10
        conv_x10 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(down_x00)
        conv_x10 = self.cbam_block(conv_x10)
        skip_x10 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x10)
        up_x10 = convolution(self.n_filter, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x10))
        down_x10 = MaxPooling2D(pool_size=(2, 2))(skip_x10)

        # X20
        conv_x20 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(down_x10)
        conv_x20 = self.cbam_block(conv_x20)
        skip_x20 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x20)
        up_x20 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x20))
        down_x20 = MaxPooling2D(pool_size=(2, 2))(skip_x20)

        # X30
        conv_x30 = convolution(self.n_filter * 8, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(down_x20)
        conv_x30 = self.cbam_block(conv_x30)
        skip_x30 = convolution(self.n_filter * 8, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x30)
        up_x30 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x30))
        down_x30 = MaxPooling2D(pool_size=(2, 2))(skip_x30)

        # X40 (Deepest point)
        conv_x40 = convolution(self.n_filter * 16, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(down_x30)
        conv_x40 = self.cbam_block(conv_x40)
        skip_x40 = convolution(self.n_filter * 16, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(
            conv_x40)  # This skip connection will not be used later
        up_x40 = convolution(self.n_filter * 8, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x40))

        # ######### Frame 2 ######### #

        # X01
        merge_x01 = concatenate([skip_x00, up_x10], axis=3)
        conv_x01 = convolution(self.n_filter, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x01)
        conv_x01 = self.cbam_block(conv_x01)
        skip_x01 = convolution(self.n_filter, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x01)

        # X11
        merge_x11 = concatenate([skip_x10, up_x20], axis=3)
        conv_x11 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x11)
        conv_x11 = self.cbam_block(conv_x11)
        skip_x11 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x11)
        up_x11 = convolution(self.n_filter, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x11))

        # X21
        merge_x21 = concatenate([skip_x20, up_x30], axis=3)
        conv_x21 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x21)
        conv_x21 = self.cbam_block(conv_x21)
        skip_x21 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x21)
        up_x21 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x21))

        # X31
        merge_x31 = concatenate([skip_x30, up_x40], axis=3)
        conv_x31 = convolution(self.n_filter * 8, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x31)
        conv_x31 = self.cbam_block(conv_x31)
        skip_x31 = convolution(self.n_filter * 8, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(
            conv_x31)  # will not be used later
        up_x31 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x31))

        # ######### Frame 3 ######### #

        # X02
        merge_x02 = concatenate([skip_x00, skip_x01, up_x11], axis=3)
        conv_x02 = convolution(self.n_filter, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x02)
        conv_x02 = self.cbam_block(conv_x02)
        skip_x02 = convolution(self.n_filter, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x02)

        # X12
        merge_x12 = concatenate([skip_x10, skip_x11, up_x21], axis=3)
        conv_x12 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x12)
        conv_x12 = self.cbam_block(conv_x12)
        skip_x12 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x12)
        up_x12 = convolution(self.n_filter, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x12))

        # X22
        merge_x22 = concatenate([skip_x20, skip_x21, up_x31], axis=3)
        conv_x22 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x22)
        conv_x22 = self.cbam_block(conv_x22)
        skip_x22 = convolution(self.n_filter * 4, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(
            conv_x22)  # will not be used later
        up_x22 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x22))

        # ######### Frame 4 ######### #

        # X03
        merge_x03 = concatenate([skip_x00, skip_x01, skip_x02, up_x12], axis=3)
        conv_x03 = convolution(self.n_filter, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x03)
        conv_x03 = self.cbam_block(conv_x03)
        conv_x03 = BatchNormalization()(conv_x03)
        skip_x03 = convolution(self.n_filter, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x03)

        # X13
        merge_x13 = concatenate([skip_x10, skip_x11, skip_x12, up_x22], axis=3)
        conv_x13 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(merge_x13)
        conv_x13 = self.cbam_block(conv_x13)
        conv_x13 = BatchNormalization()(conv_x13)
        skip_x13 = convolution(self.n_filter * 2, kernel_size=self.kernel_size,
                               drop_rate=self.dropout_rate, activation=self.activation)(conv_x13)

        up_x13 = convolution(self.n_filter, kernel_size=self.kernel_size,
                             drop_rate=self.dropout_rate, activation=self.activation)(
            UpSampling2D(size=(2, 2))(skip_x13))

        # ######### Frame 5 ######### #

        # X04
        merge_x04 = concatenate([skip_x00, skip_x01, skip_x02, skip_x03, up_x13], axis=3)
        conv_x04 = convolution(self.n_filter, kernel_size=self.kernel_size, drop_rate=self.dropout_rate,
                               activation=self.activation, use_batchnorm=True, use_dropout=False)(merge_x04)
        conv_x04 = self.cbam_block(conv_x04)
        out = convolution(2, kernel_size=self.kernel_size, drop_rate=self.dropout_rate,
                          activation=self.activation)(conv_x04)

        # ######### output ######### #
        if self.num_classes == 1:
            output = Conv2D(self.num_classes, 1, activation='sigmoid')(out)
        else:
            output = Conv2D(self.num_classes, 1, activation='softmax')(out)

        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.lr,
            decay_steps=10000,
            decay_rate=0.9)

        model = Model(inputs=self.input, outputs=output)
        opt = Adam(learning_rate=lr_schedule)
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        model.compile(optimizer=opt, loss=Scoring.dice_plus_bce_loss, metrics=Scoring.dice_scoring)
        # model.summary()
        return model