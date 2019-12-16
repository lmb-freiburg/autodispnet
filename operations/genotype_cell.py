import tensorflow as tf
from operations import *
import numpy as np

class EncoderCell:
    def __init__(self, genotype, C, reduction, reduction_prev, trainable=True):
        """
        genotype: extracted genotype for the encoder cell
        C: number of output channels for the cell
        reduction: Whether current cell is a reduction cell
        reduction_prev: Whether previous cell was a reduction cell
        trainable: whether the cell is trainable
        """
        affine = True
        self.affine = affine
        super().__init__()
        self.reduction = reduction
        if reduction_prev:
            self.preprocess0 = SpatialReduce(C_out=C, trainable=trainable, affine=affine)
        else:
            self.preprocess0 = ConvReLU(C_out=C, kernel_size=1, stride=1, trainable=trainable, affine=affine)

        self.preprocess1 = ConvReLU(C_out=C, kernel_size=1, stride=1, trainable=trainable, affine=affine)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
        else:
            op_names, indices = zip(*genotype.normal)
        self._init_cell(C, op_names, indices, reduction, trainable)

    def _init_cell(self, C, op_names, indices, reduction, trainable):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._ops = []
        ops = OPS['encoder']
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = ops[name](C, stride, affine=self.affine, trainable=trainable)
            self._ops += [op]
        self._indices = indices

    def __call__(self, input_list, drop_prob=0, train_mode=True):
        s0, s1 = input_list
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        intermediate_states = []
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states.append(s)
            intermediate_states.append(s)
        return tf.concat(intermediate_states, axis=1)


class DecoderCell:
    def __init__(self, genotype, C, upsample, upsample_prev, trainable=True):
        """
        genotype: extracted genotype for the encoder cell
        C: number of output channels for the cell
        upsample: Whether current cell is a upsampling
        upsample_prev: Whether previous cell was a upsampling cell
        trainable: whether the cell is trainable
        """
        super().__init__()
        affine = True
        self.affine = affine
        self.upsample = upsample
        if upsample_prev:
            self.preprocess0 = Upsample(C_out=C, factor=2, trainable=trainable, affine=affine)
        else:
            self.preprocess0 = ConvReLU(C_out=C, kernel_size=1, stride=1, trainable=trainable, affine=affine)
        self.preprocess1 = ConvReLU(C_out=C, kernel_size=1, stride=1, trainable=trainable, affine=affine)
        self.preprocess_pred = ConvReLU(C_out=C, kernel_size=1, stride=1, trainable=trainable, affine=affine)
        self.preprocess_support = ConvReLU(C_out=C, kernel_size=1, stride=1, trainable=trainable, affine=affine)

        if upsample:
            self.upsample0 = Upsample(C_out=C, factor=2, trainable=trainable, affine=affine)
            self.upsample1 = Upsample(C_out=C, factor=2, trainable=trainable, affine=affine)
            op_names, indices = zip(*genotype.upsample)
        else:
            op_names, indices = zip(*genotype.normal)
        self._init_cell(C, op_names, indices, upsample, trainable)


    def _init_cell(self, C, op_names, indices, upsample, trainable):
        assert len(op_names) == len(indices)
        self._steps =  int(1 + (len(op_names) - 4)* 0.5)
        self._ops = []
        ops = OPS['decoder']
        for name, index in zip(op_names, indices):
            op = ops[name](C, stride=1, affine=self.affine, trainable=trainable)
            self._ops += [op]
        self._indices = indices


    def __call__(self, input_list, drop_prob=0, train_mode=True):
        s0, s1, prev_pred, s_support = input_list
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        prev_pred = self.preprocess_pred(prev_pred)
        support = self.preprocess_support(s_support)
        # the input nodes
        states = []
        if self.upsample:
            states += [self.upsample0(s0), self.upsample1(s1)]
        else:
            states += [s0, s1]
        states += [prev_pred, support]

        offset = 0
        intermediate_states = []
        for i in range(self._steps):
            if i==0:
                h1 = states[self._indices[0]]
                h2 = states[self._indices[1]]
                h3 = states[self._indices[2]]
                h4 = states[self._indices[3]]
                op1 = self._ops[0]
                op2 = self._ops[1]
                op3 = self._ops[2]
                op4 = self._ops[3]
                h1 = op1(h1)
                h2 = op2(h2)
                h3 = op3(h3)
                h4 = op4(h4)
                s = h1 + h2 + h3 + h4
            else:
                h1 = states[self._indices[2*(i-1)+4]]
                h2 = states[self._indices[2*(i-1)+5]]
                op1 = self._ops[2*(i-1)+4]
                op2 = self._ops[2*(i-1)+5]
                h1 = op1(h1)
                h2 = op2(h2)
                s = h1 + h2
            states.append(s)
            intermediate_states.append(s)
        return tf.concat(intermediate_states, axis=1)
