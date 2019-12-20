import tensorflow as tf
import netdef_slim as nd
from lmbspecialops import leaky_relu

k_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_IN', uniform=False)


ENCODER_OPS = {
  'none'         : lambda C, stride, affine, trainable: ZeroE(stride),
  'max_pool_3x3' : lambda C, stride, affine, trainable: tf.keras.layers.MaxPooling2D(3, strides=stride, padding='same', data_format='channels_first'),
  'avg_pool_3x3' : lambda C, stride, affine, trainable: tf.keras.layers.AveragePooling2D(3, strides=stride, padding='same', data_format='channels_first'),
  'skip_connect' : lambda C, stride, affine, trainable: Identity() if stride == 1  else SpatialReduce(C, affine=affine, trainable=trainable),
  'sep_conv_3x3' : lambda C, stride, affine, trainable: DepthWiseSeparableConv(C, 3, stride, affine=affine, trainable=trainable),
  'sep_conv_5x5' : lambda C, stride, affine, trainable: DepthWiseSeparableConv(C, 5, stride, affine=affine, trainable=trainable),
  'conv_3x3'     : lambda C, stride, affine, trainable: ConvReLU(C, 3, stride, affine=affine, trainable=trainable),
  'dil_conv_3x3' : lambda C, stride, affine, trainable: DilatedSeparableConv(C, 3, stride, dilation_rate=2, affine=affine, trainable=trainable),
  'dil_conv_5x5' : lambda C, stride, affine, trainable: DilatedSeparableConv(C, 5, stride, dilation_rate=2, affine=affine, trainable=trainable),
  'conv_1x1'     : lambda C, stride, affine, trainable: ConvReLU(C, 1, stride, affine=affine, trainable=trainable),
  }


# Removing DilatedSeparableConv because tf doesn't support stride!=1 with dilation rate !=1
ENCODER_REDUCTION_OPS = {
  'none'         : lambda C, stride, affine, trainable: ZeroE(stride),
  'max_pool_3x3' : lambda C, stride, affine, trainable: tf.keras.layers.MaxPooling2D(3, strides=stride, padding='same', data_format='channels_first'),
  'avg_pool_3x3' : lambda C, stride, affine, trainable: tf.keras.layers.AveragePooling2D(3, strides=stride, padding='same', data_format='channels_first'),
  'skip_connect' : lambda C, stride, affine, trainable: Identity() if stride == 1  else SpatialReduce(C, affine=affine, trainable=trainable),
  'sep_conv_3x3' : lambda C, stride, affine, trainable: DepthWiseSeparableConv(C, 3, stride, affine=affine, trainable=trainable),
  'sep_conv_5x5' : lambda C, stride, affine, trainable: DepthWiseSeparableConv(C, 5, stride, affine=affine, trainable=trainable),
  'conv_3x3'     : lambda C, stride, affine, trainable: ConvReLU(C, 3, stride, affine=affine, trainable=trainable),
  'conv_1x1'     : lambda C, stride, affine, trainable: ConvReLU(C, 1, stride, affine=affine, trainable=trainable),
  }


DECODER_OPS = {
  'none'         : lambda C, stride, affine, trainable: ZeroD(stride),
  'max_pool_3x3' : lambda C, stride, affine, trainable: tf.keras.layers.MaxPooling2D(3, strides=stride, padding='same', data_format='channels_first'),
  'avg_pool_3x3' : lambda C, stride, affine, trainable: tf.keras.layers.AveragePooling2D(3, strides=stride, padding='same', data_format='channels_first'),
  'skip_connect' : lambda C, stride, affine, trainable: Identity(), #if stride == 1  else raise('Skip connection with stride >1 not supported'),
  'sep_conv_3x3' : lambda C, stride, affine, trainable: DepthWiseSeparableConv(C, 3, stride, affine=affine, trainable=trainable),
  'sep_conv_5x5' : lambda C, stride, affine, trainable: DepthWiseSeparableConv(C, 5, stride, affine=affine, trainable=trainable),
  'dil_conv_3x3' : lambda C, stride, affine, trainable: DilatedSeparableConv(C, 3, stride, dilation_rate=2, affine=affine, trainable=trainable),
  'dil_conv_5x5' : lambda C, stride, affine, trainable: DilatedSeparableConv(C, 5, stride, dilation_rate=2, affine=affine, trainable=trainable),
  'conv_3x3'     : lambda C, stride, affine, trainable: ConvReLU(C, 3, stride, affine=affine, trainable=trainable),
  'conv_1x1'     : lambda C, stride, affine, trainable: ConvReLU(C, 1, stride, affine=affine, trainable=trainable),
  }



OPS = {
        'encoder' : ENCODER_OPS,
        'encoder_reduction': ENCODER_REDUCTION_OPS,
	'decoder' : DECODER_OPS
      }

class Sequential(tf.keras.layers.Layer):
    def __init__(self, sequence):
        super().__init__()
        self._sequence = sequence

    def call(self, x):
        for op in self._sequence:
            x = op(x)
        return x



class ZeroE(tf.keras.layers.Layer):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def call(self, x):
        if self.stride == 1:
            return x*0.0
        return x[:,:,::self.stride,::self.stride]*0

class ZeroD(tf.keras.layers.Layer):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def call(self, x):
        "double the spatial resolution if stride is not 1"
        x = x*0.0
        if self.stride == 1:
            return x

        input_shape= x.get_shape()
        zeros = tf.zeros(shape=input_shape, dtype=tf.float32)
        x = tf.concat([x, zeros], axis=2)

        input_shape= x.get_shape()
        zeros = tf.zeros(shape=input_shape, dtype=tf.float32)
        x = tf.concat([x, zeros], axis=3)
        return x

class Identity(tf.keras.layers.Layer):
    def call(self, x):
        return x


class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, affine, axis=1, trainable=True):
        super().__init__()
        self.trainable = trainable
        self.affine = affine
        self.no_bn = nd.config.get('no_bn', True)
        if not self.no_bn:
            if affine:
                self.bn =  tf.keras.layers.BatchNormalization(axis=1, scale=True, center=True,
                                                              trainable=self.trainable,
                                                              momentum=0.9,
                                                              epsilon=1e-3
                                                              )
            else:
                self.bn =  tf.keras.layers.BatchNormalization(axis=1, scale=False, center=False,
                                                              trainable=self.trainable,
                                                              momentum=0.9,
                                                              epsilon=1e-3
                                                              )
    def call(self, x):
        if self.no_bn:
            return x
        is_training = nd.phase == 'train'
        bn_out = self.bn(x, training=is_training)
        for update in self.bn.updates:
            tf.add_to_collection('update_ops', update)
        return bn_out


class ConvReLU(tf.keras.layers.Layer):
    def __init__(self, C_out, kernel_size, stride, padding='SAME', activation=leaky_relu, affine=True, trainable=True):
        super().__init__()
        ops = []
        conv = tf.keras.layers.Conv2D(filters=C_out, kernel_size=kernel_size,
                                      strides=stride, padding=padding,
                                      data_format='channels_first', activation=None,
                                      kernel_initializer=k_initializer,
                                      use_bias = True,
                                      trainable=trainable)
        ops.append(conv)
        ops.append(activation)
        bn = BatchNorm(affine=affine, trainable=trainable)
        ops.append(bn)
        self.op = Sequential(ops)

    def call(self, x):
        return self.op(x)


class Upsample(tf.keras.layers.Layer):
    def __init__(self, C_out, factor=2, affine=True, activation=leaky_relu, trainable=True):
        super().__init__()
        ops = []
        conv = tf.keras.layers.Conv2DTranspose(filters=C_out, kernel_size=4,
                                               strides=2, padding='same',
                                               data_format='channels_first', activation=None,
                                               kernel_initializer=k_initializer,
                                               use_bias = True, trainable=trainable)
        ops.append(conv)
        ops.append(activation)
        bn = BatchNorm(affine=affine, trainable=trainable)
        ops.append(bn)
        self.op = Sequential(ops)

    def call(self, x):
        return self.op(x)


class DepthWiseSeparableConv(tf.keras.layers.Layer):
    def __init__(self, C_out, kernel_size, stride, padding='SAME', activation=leaky_relu, affine=True, trainable=True):
        super().__init__()
        # DARTS uses de sep convs twice
        conv1 = tf.keras.layers.SeparableConv2D(filters=C_out, kernel_size=kernel_size,
                                              strides=stride, padding=padding,
                                              data_format='channels_first',
                                              use_bias = True,
                                              activation=None,
                                              kernel_initializer=k_initializer,
                                              depthwise_initializer=k_initializer,
                                              pointwise_initializer=k_initializer,
                                              trainable=trainable
                                              )

        bn1 = BatchNorm(affine=affine, trainable=trainable)
        conv2 = tf.keras.layers.SeparableConv2D(filters=C_out, kernel_size=kernel_size,
                                              strides=1, padding=padding,
                                              data_format='channels_first',
                                              use_bias = True,
                                              activation=None,
                                              kernel_initializer=k_initializer,
                                              depthwise_initializer=k_initializer,
                                              pointwise_initializer=k_initializer,
                                              trainable=trainable
                                              )

        bn2 = BatchNorm(affine=affine, trainable=trainable)
        self.op = Sequential([conv1, activation, bn1,
                              conv2, activation, bn2
                              ])

    def call(self, x):
        return self.op(x)

class DilatedSeparableConv(tf.keras.layers.Layer):
    def __init__(self, C_out, kernel_size, stride, dilation_rate, padding='SAME', activation=leaky_relu, affine=True, trainable=True):
        super().__init__()
        assert(stride==1)
        ops = []
        conv = tf.keras.layers.SeparableConv2D(filters=C_out, kernel_size=kernel_size,
                                              strides=stride, padding=padding,
                                              data_format='channels_first',
                                              dilation_rate=dilation_rate,
                                              activation=None,
                                              use_bias=True,
                                              kernel_initializer=k_initializer,
                                              depthwise_initializer=k_initializer,
                                              pointwise_initializer=k_initializer,
                                              trainable=trainable
                                              )
        ops.append(conv)
        ops.append(activation)
        bn = BatchNorm(affine=affine, trainable=trainable)
        ops.append(bn)
        self.op = Sequential(ops)

    def call(self, x):
        return self.op(x)



class SpatialReduce(tf.keras.layers.Layer):
    def __init__(self, C_out, affine=True, trainable=True):
        super().__init__()
        self.op = ConvReLU(C_out, 3, 2)

    def call(self, x):
        return self.op(x)

