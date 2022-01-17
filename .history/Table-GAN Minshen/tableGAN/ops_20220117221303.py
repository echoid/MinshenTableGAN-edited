import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.compat.v1.summary.image
    scalar_summary = tf.compat.v1.summary.scalar
    histogram_summary = tf.compat.v1.summary.histogram
    merge_summary = tf.compat.v1.summary.merge
    SummaryWriter = tf.compat.v1.summary.FileWriter

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.compat.v1.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return BatchNormalization(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))

        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.compat.v1.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):

        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    # print( "Linear shape = " + str(shape) )

    with tf.compat.v1.variable_scope(scope or "Linear"):

        matrix = tf.compat.v1.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))

        bias = tf.compat.v1.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))

        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def masking(input, label_col, attrib_num):
    i_shape = input.get_shape().as_list()

    print("i_shape = " + str(i_shape))  # 64 * 8* 8 * 1 , 64 * 16 * 16 *1

    # input data  is flatten version of G
    temp = tf.reshape(input, [i_shape[0], -1])
    t_shape = temp.get_shape().as_list()
    print("t_shape = " + str(t_shape))  # 64 * 64 , 64*256

    # Masking Label Columns
    mask = np.zeros(t_shape)

    # A tensor with shape of GC having True in all elements
    mask = np.equal(mask, mask)

    mask_col = label_col

    # Masking all label columns in cases that the inital data has been duplicated
    for i in range(t_shape[1] // attrib_num):
        mask[:, mask_col] = False
        mask_col += attrib_num

    inp_mask = tf.constant(mask)

    temp = tf.where(inp_mask, temp, tf.zeros_like(temp))

    return tf.reshape(temp, i_shape)
