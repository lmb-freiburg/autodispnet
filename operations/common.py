import tensorflow as tf
import netdef_slim as nd
import math
import itertools
import six
import numpy as np
from math import ceil



class DispNetGenotypeBase:
    def __init__(self, genotype, C, encoder_depth=2, decoder_depth=1, feature_depth=1, stem_multiplier=3, max_displacement=40, trainable=True):
        self._C = C
        self._feature_depth = feature_depth
        self._encoder_depth = encoder_depth
        self._decoder_depth = decoder_depth
        self._stem_multiplier = stem_multiplier
        self._max_displacement = max_displacement
        self.trainable = trainable

        self.encoder_cells = []
        self.encoder_feature_cells = []
        self.decoder_cells = []

        self.spatial_level = 0
        self._feature_store = {}
        self.genotype = genotype
        self.init_model()


    def init_model(self):
        raise NotImplementedError('To be implemented in child class')


    def __call__(self, inputs):
        raise NotImplementedError('To be implemented in child class')

    def weight_parameters(self):
        return tf.trainable_variables()

class Correlation1D:
    def __init__(self, max_displacement):
        self.max_displacement = max_displacement

    def __call__(self, feat1, feat2):
        return nd.ops.correlation_1d(feat1, feat2, kernel_size=1,
                                                   max_displacement=self.max_displacement,
                                                   pad=self.max_displacement,
                                                   stride1 =1,
                                                   stride2 =1)

class DispPredBlock(tf.keras.layers.Layer):

    def __init__(self, trainable):
        k_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_IN', uniform=False)
        self.conv = tf.keras.layers.Conv2D(filters=1, kernel_size=3,
                                                      strides=1,
                                                      data_format='channels_first',
                                                      use_bias = True,
                                                      trainable=trainable,
                                                      kernel_initializer=k_initializer,
                                                      padding='same')
    def __call__(self, x):
        return nd.ops.neg_relu(self.conv(x))



def get_param_count(params):
    count = 0
    for p in params:
        count+=np.prod(p.shape)
    return count

def pred_eval_spec(model, images, labels, mode, params):
    with tf.name_scope('tower_%d' % 0) as name_scope:
        divisor = nd.config.get('divisor', 64.0)
        height = int(images[0].get_shape()[2])
        width = int(images[0].get_shape()[3])
        scale = nd.config.get('eval_scale', 1.0)
        print("#### Eval scale is {}:".format(scale))
        temp_width = ceil(width*scale/divisor) * divisor
        temp_height = ceil(height*scale/divisor) * divisor
        rescale_coeff_x = width/temp_width
        resample_input = lambda x: nd.ops.resample(x, width=temp_width, height=temp_height,
                                                                        type='LINEAR',
                                                                        antialias=True)
        resampled_images = map(resample_input, images)
        predictions = model(resampled_images, train_mode=False)
        final_pred = predictions['final']
        resample_output = lambda x: nd.ops.scale(nd.ops.resample(x, width=width,
                                                                    height=height,
                                                                    type='LINEAR',
                                                                    antialias=True), rescale_coeff_x)
        final_pred = resample_output(final_pred)
        metrics = None
        if labels:
            epe = nd.ops.disp_accuracy(final_pred, labels, 'disp_epe')
            aepe = tf.metrics.mean(epe)
            metrics = {'AEPE': aepe}
    return final_pred, metrics
