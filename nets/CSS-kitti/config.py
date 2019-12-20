import netdef_slim as nd
import os
import json
from collections import namedtuple
from model_genotype import model_func
from netdef.schedules.named_schedule import NamedSchedule

nd.evo_manager.set_training_dir(os.path.join(os.path.dirname(__file__), 'training'))

# add evolution
max_steps = 90000
evo = nd.Evolution('kitti', [], NamedSchedule('genotype', max_steps))
nd.add_evo(evo)

nd.config['model_fn'] = model_func
nd.config['num_gpus'] = 1
nd.config['test_batch_size'] = 1
nd.config['no_bn'] = True

Genotype = namedtuple('Genotype', 'normal reduce upsample')

genotype = Genotype(normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0)],
                    reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_5x5', 3)],
                    upsample=[('sep_conv_3x3', 3), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 4), ('sep_conv_3x3', 2), ('dil_conv_5x5', 4)]
                    )

hyperparams={'genotype': genotype}

nd.config['hyperparams'] = hyperparams
