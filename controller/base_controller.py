import netdef as nd
nd.choose_framework('tensorflow')

import argparse, re, datetime, sys, tb, os
import tensorflow as tf
from net_actions import BaseNetActions
import signal

class BaseController:
    base_path=None

    def __init__(self, path=None, net_actions=BaseNetActions):
        if path is not None:
            self.base_path = path
        nd.load_module(os.path.join(self.base_path, 'config.py'))
        self.net_actions = net_actions

    def run(self):
        self._command_hooks = {}
        self._parser = argparse.ArgumentParser(description='process network')
        self._configure_parser()
        self._configure_subparsers()
        self._args = self._parser.parse_args()

        command = self._args.command
        if command is None:
            self._parser.print_help()
            return

        if command not in self._command_hooks:
            raise BaseException('Unknown command: ' + command)

        self._command_hooks[command]()

    def _configure_parser(self):
        self._parser.add_argument('--gpu-id',    help='outside cluster: gpu ID to use (default=0)', default=None, type=int)
        self._subparsers = self._parser.add_subparsers(dest='command', prog='controller')

    def _configure_subparsers(self):
        # test
        subparser = self._subparsers.add_parser('test', help='test a network')
        subparser.add_argument('--dataset',             help='test dataset', default=None)
        subparser.add_argument('--state',               help='state pointing to the snapshot to test, in the form EvoName:iteration', default=None)
        subparser.add_argument('--output',      help='output images to folder output_...', action='store_true')

        def run_test():
            args = self._args
            states = []
            states.append(args.state)
            for s in states:
                print("Testing at state: {} ".format(s))
                self.test(dataset=args.dataset, state=nd.evo_manager.get_state(s), output=args.output, caffe_weights=args.caffe_weights)
            sys.exit(nd.status.SUCCESS)
        self._command_hooks['test'] = run_test

        # eval
        subparser = self._subparsers.add_parser('eval', help='run network on images')
        subparser.add_argument('img0',        help='path to input img0')
        subparser.add_argument('img1',        help='path to input img1')
        subparser.add_argument('out_dir',        help='path to output dir')
        subparser.add_argument('--state',     help='state of the snapshot', default=None)

        def eval():
            self.eval(image_0=self._args.img0,
                      image_1=self._args.img1,
                      out_dir=self._args.out_dir,
                      state=self._args.state)
            sys.exit(nd.status.SUCCESS)
        self._command_hooks['eval'] = eval

        # perf_test
        subparser = self._subparsers.add_parser('perf-test', help='measure of runtime of core net with no data I/O')
        subparser.add_argument('--burn_in',     help='number of iters to burn-in before measureing runtime', default=50)
        subparser.add_argument('--iters',       help='number of iters to average runtime', default=100)
        subparser.add_argument('--resolution',  help='the resolution used to measure runtime (width height), default is the Sintel resolution', nargs=2, default=(1024, 436))

        def perf_test():
            self.perf_test(burn_in = self._args.burn_in,
                           iters=self._args.iters,
                           resolution=self._args.resolution)
            sys.exit(nd.status.SUCCESS)
        self._command_hooks['perf-test'] = perf_test


        # eval-list
        subparser = self._subparsers.add_parser('eval-list', help='run network on a list of images')
        subparser.add_argument('list',        help='file containing "img0 img1 flow"')
        subparser.add_argument('--state',     help='state from which to get .caffemodel', default=None)

        # list-evos
        subparser = self._subparsers.add_parser('list-evos', help='list evolution definitions')
        def list_evos():
            for evo in nd.evo_manager.evolutions():
                print(evo)
            sys.exit(nd.status.SUCCESS)
        self._command_hooks['list-evos'] = list_evos

        # list-states
        subparser = self._subparsers.add_parser('list-states', help='list present states')
        def list_states():
            for evo in nd.evo_manager.evolutions():
                for state in evo.states():
                    print(state)
            sys.exit(nd.status.SUCCESS)
        self._command_hooks['list-states'] = list_states

        # state
        subparser = self._subparsers.add_parser('state', help='list present states')
        def state():
            raise NotImplementedError
            sys.exit(nd.status.SUCCESS)
        self._command_hooks['state'] = state

        # train
        subparser = self._subparsers.add_parser('train', help='train a network from scratch')
        subparser.add_argument('--weights',     help='path to snapshot to initialize from', default=None)
        subparser.add_argument('--walltime',    help='process walltime', nargs=1, default=['24:00:00'], required=False, type=str)
        subparser.add_argument('--check',       help='check only', default=False, required=False, action='store_true')
        def train():
            args = self._args
            if 'finetune_from' in nd.config.keys():
                weights = nd.config['finetune_from']
            else:
                weights = None
            wt_match = re.compile('([0-9]{2}):([0-9]{2}):([0-9]{2})').match(args.walltime[0])
            if not wt_match:
                raise ValueError('Invalid walltime: use the format HH:MM:SS')
            wt_timedelta = datetime.timedelta(hours=int(wt_match.group(1)), minutes=int(wt_match.group(2)),
                                              seconds=int(wt_match.group(3)))
            # always finetune when using train
            print("Finetune weights: {}".format(weights))
            self.train(walltime=wt_timedelta.total_seconds(), weights=weights, finetune=True, check_only=args.check)
            sys.exit(nd.status.SUCCESS)
        self._command_hooks['train'] = train

        # continue
        subparser = self._subparsers.add_parser('continue', help='continue training from last saved (or specified) state')
        subparser.add_argument('--state',       help='state from which to continue (default=last)', default=None)
        subparser.add_argument('--check',       help='check only if command can be executed, but do not run', action="store_true")
        subparser.add_argument('--walltime', help='process walltime', nargs=1, default=['24:00:00'], required=False, type=str)
        def continue_training():
            args = self._args
            wt_match = re.compile('([0-9]{2}):([0-9]{2}):([0-9]{2})').match(args.walltime[0])
            if not wt_match:
                raise ValueError('Invalid walltime: use the format HH:MM:SS')
            wt_timedelta = datetime.timedelta(hours=int(wt_match.group(1)), minutes=int(wt_match.group(2)),
                                              seconds=int(wt_match.group(3)))
            if args.state is not None: nd.evo_manager.clean_after(args.state)
            self.train(walltime=wt_timedelta.total_seconds(), weights=None, check_only=args.check, finetune=False)
            sys.exit(nd.status.SUCCESS)
        self._command_hooks['continue'] = continue_training

        # copy
        subparser = self._subparsers.add_parser('copy', help='copy current network')
        subparser.add_argument('tgt', help='target location')
        subparser.add_argument('-w','--with-snapshot', help='copy with snapshot', default=False, action='store_true')
        subparser.add_argument('-s','--state', help='state to copy', default=None)
        def copy():
            self.copy(
                self._args.tgt,
                copy_snapshot=self._args.with_snapshot,
                selected_state=self._args.state,
                verbose=True
            )
            sys.exit(nd.status.SUCCESS)
        self._command_hooks['copy'] = copy

        # num params
        subparser = self._subparsers.add_parser('params', help='calculate the number of parameters')
        self._command_hooks['params'] = self.params

        # num Flops
        subparser = self._subparsers.add_parser('flops', help='calculate the number of FLOPs')
        self._command_hooks['flops'] = self.flops


    def train(self, **kwargs):
        return self.net_actions(net_dir=self.base_path).train(**kwargs)

    def eval(self, **kwargs):
        out_dir = kwargs.pop('out_dir')
        output = self.net_actions(net_dir=self.base_path).eval(**kwargs)
        return output

    def test(self, **kwargs):
        return self.net_actions(net_dir=self.base_path).test(**kwargs)

    def perf_test(self, **kwargs):
        return self.net_actions(net_dir=self.base_path).perf_test(**kwargs)

    def params(self, **kwargs):
        return self.net_actions(net_dir=self.base_path).params(**kwargs)

    def flops(self, **kwargs):
        return self.net_actions(net_dir=self.base_path).flops(**kwargs)

    def copy(self, target, copy_snapshot, selected_state, verbose=False):
        source = self.base_path

        if selected_state is None:
            selected_state = nd.evo_manager.last_trained_evolution().last_state()
        else:
            selected_state = nd.evo_manager.get_state(selected_state)

        print('target', target)
        os.system('mkdir -p %s' % target)
        non_copy_list = ['.','..','scratch','jobs','__pycache__']

        if copy_snapshot:
            target_snapshot_folder = os.path.join(target, 'training', selected_state.evo().name(), 'checkpoints')
            os.system('mkdir -p %s' % target_snapshot_folder)
            state_files = selected_state.files()
            os.system('cp %s %s %s' % ('-v' if verbose else '', state_files['meta'], target_snapshot_folder))
            os.system('cp %s %s %s' % ('-v' if verbose else '', state_files['index'], target_snapshot_folder))
            for f in state_files['data']:
                os.system('cp %s %s %s' % ('-v' if verbose else '', f, target_snapshot_folder))

        for f in os.listdir(source):
            if f.endswith('.py'):
                os.system('cp %s %s/%s %s' % ('-v' if verbose else '', source, f, target))
