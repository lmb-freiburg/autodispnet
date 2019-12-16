import tensorflow as tf
import netdef as nd
import math
import itertools
import six
from netdef.losses.disp import disp_epe_loss, disp_epe_loss_nan
import numpy as np
from math import ceil


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



def epe_loss(predictions, labels, suffix, loss_weights):
    losses = []
    for level, pred in enumerate(predictions):
        print("Adding loss level {} ({}) with weight {}".format(level, pred.shape, loss_weights[level]))

        loss = disp_epe_loss(name='{}_level_{}.disp.L'.format(suffix, level),
                             gt = labels, pred = pred,
                             weight = loss_weights[level]
                             )
        losses.append(loss)

    total_loss = tf.add_n(losses)
    tf.add_to_collection('epe_loss_'+suffix, total_loss)
    return total_loss

def average_gradients(tower_gradvars, name):
    # Now compute global loss and gradients.
    gradvars = []
    with tf.name_scope('gradient_averaging_' + name):
        all_grads = {}
        for grad, var in itertools.chain(*tower_gradvars):
            if grad is not None:
                all_grads.setdefault(var, []).append(grad)

        for var, grads in six.iteritems(all_grads):
          # Average gradients on the same device as the variables # to which they apply.
            with tf.device(var.device):
                if len(grads) == 1:
                    avg_grad = grads[0]
                else:
                    avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
                gradvars.append((avg_grad, var))
    return gradvars


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
        #rescale_coeff_y = height/temp_height
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
        epe = nd.ops.disp_accuracy(final_pred, labels, 'disp_epe')
        aepe = tf.metrics.mean(epe)
        metrics = {'AEPE': aepe}
    return final_pred, metrics
