import netdef_slim as nd
nd.choose_framework('tensorflow')

from netdef_slim.utils import io
import argparse, re, datetime, sys, tb, os
import tensorflow as tf
from autodispnet.controller.net_actions import NetActions
import signal

class BaseController:
    base_path=None

    def __init__(self, path=None, net_actions=NetActions):
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



    def eval(self, **kwargs):
        out_dir = kwargs.pop('out_dir')
        output = self.net_actions(net_dir=self.base_path).eval(**kwargs)
        out_path = os.path.join(out_dir, 'disp.float3')
        print("Writing output to {}".format(out_path))
        io.write(out_path, output[0,:,:,:].transpose(1,2,0))
        return output

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
