import tensorflow as tf
import netdef_slim  as nd
import math

from autodispnet.operations.common import pred_eval_spec
from autodispnet.operations.corr_net import DispNetCGenotype
from autodispnet.operations.refine_net import DispNetSGenotype

class DispNetCSS:
    def __init__(self, net1_args, net2_args, net3_args):
        self.net1_args = net1_args
        self.net2_args = net2_args
        self.net3_args = net3_args
        self.init_model()

    def init_model(self):
        self.model_c = DispNetCGenotype(**self.net1_args)
        self.model_s1 = DispNetSGenotype(full_res=False, **self.net2_args)
        self.model_s2 = DispNetSGenotype(full_res=False, **self.net3_args)

    def __call__(self, inputs, train_mode=False):
        print("Graph is in train mode: {}".format(train_mode))
        print("===== Building net1 =====")
        imageL, imageR = inputs
        model_c_out = self.model_c([imageL, imageR], train_mode=False)
        prev_pred = model_c_out['final']
        prev_pred_upsampled = nd.ops.resample(prev_pred, reference=imageL, type='LINEAR')
        print("===== Building net2 =====")
        model_s_pred = self.model_s1([imageL, imageR, prev_pred_upsampled], train_mode=False, scope='refine')
        prev_pred = model_s_pred['final']
        prev_pred_upsampled = nd.ops.resample(prev_pred, reference=imageL, type='LINEAR')
        print("===== Building net3 =====")
        model_s_pred = self.model_s2([imageL, imageR, prev_pred_upsampled], train_mode=False, scope='refine1')
        return model_s_pred


def model_func(features, labels, mode, params):
    # Initialize the model
    smult = 2
    net1_args = { 'C':18, 'feature_depth':1 , 'encoder_depth':6, 'decoder_depth':4, 'stem_multiplier': smult, 'genotype': nd.config['hyperparams']['genotype']}
    net2_args = { 'C':18, 'encoder_depth':7,  'decoder_depth':4, 'stem_multiplier': smult, 'genotype': nd.config['hyperparams']['genotype']}
    net3_args = { 'C':18, 'encoder_depth':7,  'decoder_depth':4, 'stem_multiplier': smult, 'genotype': nd.config['hyperparams']['genotype']}
    model = DispNetCSS(net1_args=net1_args, net2_args=net2_args, net3_args=net3_args)
    if mode == 'NO_DATA':
        images = [ tf.ones(shape=(1, 3, 448, 1024)) , tf.ones(shape=(1, 3, 448, 1024))]
        return model(images)
    else:
        return pred_eval_spec(model, features, labels, mode, params)
