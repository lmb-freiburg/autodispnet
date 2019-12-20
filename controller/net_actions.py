import tensorflow as tf
import os, sys, time
import netdef_slim as nd
import numpy as np
import timeit
import signal
from netdef_slim.tensorflow.tools.trainer.simpletrainer import SimpleTrainer
from tensorflow.contrib import slim
tf.logging.set_verbosity(tf.logging.INFO)
from tensorflow.python.framework import graph_util
from scipy import misc
import os
import re
os.environ['TF_SYNC_ON_FINISH'] = '0'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

def comma_me(amount):
        orig = amount
        new = re.sub("^(-?\d+)(\d{3})", '\g<1>,\g<2>', amount)
        if orig == new:
            return new
        else:
            return comma_me(new)

class NetActions:

    def __init__(self, net_dir, save_snapshots=True, save_summaries=True):
        self._check_evo_manager_init()
        self.save_snapshots = save_snapshots
        self.save_summaries = save_summaries
        self.net_dir = net_dir
        self.eval_session = None

    def _check_evo_manager_init(self):
        if (len(nd.evo_manager.evolutions()) == 0):
            raise ValueError('Evolutions are empty. Make sure evo manager has correctly loaded config.py in your network directory')

    def _create_session(self):
        config = tf.ConfigProto(log_device_placement=False,
                                allow_soft_placement=True
                               )
        session = tf.Session(config = config)
        return session


    def params(self):
        nd.phase = 'test'
        last_evo, current_evo = nd.evo_manager.get_status()
        model_fn = nd.config['model_fn']
        hyperparams = nd.config['hyperparams']
        model_fn(features=None, labels=None, mode='NO_DATA', params=hyperparams)
        total_parameters = 0
        for variable in tf.global_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print('{} : {}'.format(variable.name, variable_parameters))
            total_parameters += variable_parameters
        print('Total params: {:,}'.format(total_parameters))
        return total_parameters



    def flops(self):
        def load_pb(pb):
            with tf.gfile.GFile(pb, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name='')
                return graph

        nd.phase = 'test'
        model_fn = nd.config['model_fn']
        hyperparams = nd.config['hyperparams']
        g = tf.Graph()
        sess = tf.Session(graph=g)
        with g.as_default():
            output = model_fn(features=None, labels=None, mode='NO_DATA', params=hyperparams)
            tf.identity(output['final'], name='output')
            sess.run(tf.global_variables_initializer())

        output_graph_def = graph_util.convert_variables_to_constants(sess, g.as_graph_def(), ['output'])

        with tf.gfile.GFile('/tmp/graph.pb', "wb") as f:
            f.write(output_graph_def.SerializeToString())

        g2 = load_pb('/tmp/graph.pb')
        with g2.as_default():
            flops = tf.profiler.profile(g2, options = tf.profiler.ProfileOptionBuilder.float_operation())
            print('FLOP after freezing', comma_me(str(flops.total_float_ops)))

    def eval(self, image_0, image_1, state=None):
        nd.phase = 'test'
        if isinstance(image_0, str): image_0=misc.imread(image_0).transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
        if isinstance(image_1, str): image_1=misc.imread(image_1).transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
        tf.reset_default_graph()
        height = image_0.shape[2]
        width = image_0.shape[3]
        last_evo, current_evo = nd.evo_manager.get_status()
        print('Evolution: ' + last_evo.path())
        model_fn = nd.config['model_fn']
        hyperparams = nd.config['hyperparams']

        pl_image0 = tf.placeholder( 'float32',
                        shape=image_0.shape,
                        name='image0'
                        )
        pl_image1 = tf.placeholder( 'float32',
                        shape=image_0.shape,
                        name='image1'
                        )

        features = [nd.ops.scale_and_subtract_mean(pl_image0),
                    nd.ops.scale_and_subtract_mean(pl_image1)]

        pred, _ = model_fn(features, None, mode=tf.estimator.ModeKeys.EVAL, params=hyperparams)

        session = self._create_session()
        trainer = SimpleTrainer(session=session, train_dir=last_evo.path())
        session.run(tf.global_variables_initializer())
        ignore_vars = []
        if state is None:
            state = last_evo.last_state()
            trainer.load_checkpoint(state.path(), ignore_vars=ignore_vars)
        else:
            state = nd.evo_manager.get_state(state)
            trainer.load_checkpoint(state.path(), ignore_vars=ignore_vars)
        out = session.run(pred, feed_dict={ pl_image0: image_0,
                                            pl_image1: image_1})
        return out

    def _signal_handler(self, signum, frame):
        print("received signal {0}".format(signum), flush=True)
        sys.exit(0)
