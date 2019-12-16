from operations import *
from hyparchsearch.cnn.architecture.disparity.genotype_base import *
from hyparchsearch.cnn.architecture.disparity.common import Correlation1D, DispPredBlock
from genotype_cell import *


class DispNetCGenotype(DispNetGenotypeBase):

    def init_model(self):
        print("Initializing model...")
        C_curr = int(self._stem_multiplier*self._C)
        print("Stem channels: {}".format(C_curr))
        self.stems = [ConvReLU(C_out=C_curr, kernel_size=7,
                                                        stride=2,
                                                        trainable=self.trainable),
                      ConvReLU(C_out=C_curr*2, kernel_size=5,
                                                        stride=2,
                                                        trainable=self.trainable),
                      ]
        C_curr = self._C
        reduction_prev = False
        for i in range(0, self._feature_depth):
            C_curr *= 2
            print("Feature block {} channels: {}".format( i, C_curr))
            reduction = True
            cell = EncoderCell(self.genotype, C_curr, reduction, reduction_prev, trainable=self.trainable)
            reduction_prev = reduction
            self.encoder_feature_cells += [cell]

        self.corr_block = Correlation1D(max_displacement=self._max_displacement)
        reduction_prev = False # corr op does not reduce spatial resolution
        for i in range(0, self._encoder_depth):
            if (i+1)%2==0:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            print("Encoder block {} channels: {}".format( i, C_curr))
            cell = EncoderCell(self.genotype, C_curr, reduction, reduction_prev, trainable=self.trainable)
            reduction_prev = reduction
            self.encoder_cells += [cell]

        self.pred_blocks = [DispPredBlock(self.trainable)]

        upsample_prev = False
        for i in range(0, self._decoder_depth):
            # Always upsample
            upsample=True
            C_curr=int(C_curr/2)
            print("Decoder block {} channels: {}".format( i, C_curr))
            cell = DecoderCell(self.genotype, C_curr, upsample=upsample, upsample_prev=upsample_prev, trainable=self.trainable)
            self.decoder_cells.append(cell)
            self.pred_blocks.append(DispPredBlock(self.trainable))
            upsample_prev = upsample


    def __call__(self, inputs, drop_prob=tf.zeros(shape=()) , train_mode=True):
        print("Graph is in train mode: {}".format(train_mode))
        if train_mode:
            nd.phase='train'
        else:
            nd.phase = 'test'

        image_L, image_R = inputs
        spatial_level = 0
        f0 = image_L
        f1 = image_R
        self._feature_store[spatial_level] = f0

        print("Tensor shapes are:")
        print("image_L : {}".format(image_L.shape))
        print("image_L : {}".format(image_R.shape))
        for i, stem in enumerate(self.stems):
            f0 = stem(f0)
            f1 = stem(f1)
            print("stem {} : {}".format( i, f0.shape))
            spatial_level+=1
            self._feature_store[spatial_level] = f0

        f0_s0 = f0_s1 = f0
        f1_s0 = f1_s1 = f1
        for i, cell in enumerate(self.encoder_feature_cells):
            f0_s0, f0_s1 = f0_s1, cell([f0_s0, f0_s1], drop_prob, train_mode)
            f1_s0, f1_s1 = f1_s1, cell([f1_s0, f1_s1], drop_prob, train_mode)
            print("feature {} : {}".format( i, f0_s1.shape))
            spatial_level+=1
            self._feature_store[spatial_level] = f0_s1

        s0 = tf.concat([f0_s1, f1_s1], axis=1)
        s1 = self.corr_block(f0_s1, f1_s1)

        for i, cell in enumerate(self.encoder_cells):
            if cell.reduction:
                s0, s1 = s1, cell([s0, s1], drop_prob, train_mode)
                spatial_level+=1
                self._feature_store[spatial_level]=s1
            else:
                s0, s1 = s1, cell([s0, s1], drop_prob, train_mode)
            print("encoder {} : {}".format( i, s1.shape))

        prev_pred = self.pred_blocks[0](s1)
        predictions = {'all':[], 'final': None}
        predictions['all'].append(prev_pred)
        s0 = s1
        for i, cell in enumerate(self.decoder_cells):
            s_support = self._feature_store[spatial_level-1] # get a spatial feature from a higher spatial level
            prev_pred_upsampled = nd.ops.resample(prev_pred, reference=s_support, type='LINEAR')
            s0, s1  = s1, cell([s0, s1, prev_pred_upsampled, s_support], drop_prob, train_mode)
            prev_pred = self.pred_blocks[i+1](s1)
            predictions['all'].append(prev_pred)
            spatial_level=spatial_level-1
            print("decoder {} : {}".format( i, s1.shape))
        predictions['final'] = prev_pred
        return predictions
