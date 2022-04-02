from tensorflow.keras import backend as K
import tensorflow as tf

def dice_plus_bce_loss(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
    inputs_dice = K.flatten(tf.cast(inputs, tf.float32))
    inputs_bce = K.flatten(tf.cast(inputs, tf.float32))
    targets = K.flatten(tf.cast(targets, tf.float32))

    intersection = K.sum(targets * inputs_dice)

    dice = (2 * intersection) / (K.sum(targets) + K.sum(inputs_dice) + smooth)

    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    bce = bce(targets, inputs_bce)

    return -K.log(dice + smooth)

def dice_scoring(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(tf.cast(inputs, tf.float32))
    targets = K.flatten(tf.cast(targets, tf.float32))

    intersection = K.sum(targets * inputs)
    dice = (2 * intersection) / (K.sum(targets) + K.sum(inputs) + smooth)
    return dice
