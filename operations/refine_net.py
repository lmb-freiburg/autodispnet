from autodispnet.operations.cell_ops import *
from autodispnet.operations.genotype_cell import *
from autodispnet.operations.common import *


class DispPredRefineBlock(tf.keras.layers.Layer):
    def __init__(self, trainable):
        k_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_IN', uniform=False)
        self.conv = tf.keras.layers.Conv2D(filters=1, kernel_size=3,
                                                      strides=1,
                                                      data_format='channels_first',
                                                      use_bias = True,
                                                      trainable=trainable,
                                                      kernel_initializer=k_initializer,
                                                      padding='same')
    def __call__(self, x, prev_pred):
        resampled_pred = nd.ops.resample(prev_pred, reference=x, type='LINEAR')
        pred = nd.ops.neg_relu(self.conv(x) + resampled_pred)
        return pred


class Refine:
    def __init__(self, C, pred_block):
        self.upsample = Upsample(C)
        self.pred_block = pred_block

    def __call__(self, inputs):
        x, skip_conn, prev_pred, prev_net_pred = inputs
        upsampled = self.upsample(x)
        prev_pred = nd.ops.resample(prev_pred, reference=skip_conn, type='LINEAR')
        concat_feats = nd.ops.concat([skip_conn, upsampled, prev_pred], axis = 1)
        return self.pred_block(concat_feats, prev_net_pred), concat_feats

class DispNetSGenotype(DispNetGenotypeBase):
    def __init__(self, full_res, *args, **kwargs):
        self.full_res = full_res
        super().__init__(*args, **kwargs)

    def init_model(self):
        C_curr = int(self._stem_multiplier*self._C)
        self.stems = [ConvReLU(C_out=C_curr, kernel_size=7,
                                                        stride=2,
                                                        trainable=self.trainable),
                      ConvReLU(C_out=C_curr*2, kernel_size=5,
                                                        stride=2,
                                                        trainable=self.trainable),
                      ]
        C_curr = self._C
        reduction_prev = False
        for i in range(0, self._encoder_depth):
            if i%2==0:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = EncoderCell(self.genotype, C_curr, reduction, reduction_prev, trainable=self.trainable)
            reduction_prev = reduction
            self.encoder_cells += [cell]

        self.pred_blocks = [DispPredRefineBlock(self.trainable)] # Change to residual block
        upsample_prev = False
        for i in range(0, self._decoder_depth):
            # Always upsample
            upsample=True
            C_curr=int(C_curr/2)
            cell = DecoderCell(self.genotype, C_curr, upsample=upsample, upsample_prev=upsample_prev, trainable=self.trainable)
            self.decoder_cells.append(cell)
            self.pred_blocks.append(DispPredRefineBlock(self.trainable))
            upsample_prev = upsample

        if self.full_res:
            # Refinement aux ops
            C_curr=int(C_curr/2)
            ref_pred_1 = DispPredRefineBlock(self.trainable)
            self.pred_blocks.append(ref_pred_1)
            self.refine1  = Refine(C_curr, ref_pred_1)

            C_curr=int(C_curr/2)
            self.image_pp = ConvReLU(C_out=C_curr, kernel_size=3, stride=1, trainable=self.trainable)
            ref_pred_2 = DispPredRefineBlock(self.trainable)
            self.pred_blocks.append(ref_pred_2)
            self.refine2  = Refine(C_curr, ref_pred_2)



    def __call__(self, inputs, train_mode=True, scope=None):
        print("Graph is in train mode: {}".format(train_mode))
        if train_mode:
            nd.phase='train'
        else:
            nd.phase='test'
        spatial_level = 0
        with tf.name_scope(scope) as name_scope:
            image_L, image_R, prev_net_pred = inputs
            tf.summary.image('prev_dispL.pred', nd.ops.to_nhwc(prev_net_pred), max_outputs=1)

            warped_image = nd.ops.warp(image_R, nd.ops.disp_to_flow(prev_net_pred))

            s = tf.concat([image_L, image_R, warped_image, nd.ops.scale(prev_net_pred, 0.05)], axis=1)

            self._feature_store[spatial_level] = image_L

            print("Tensor shapes are:")
            print("image_L : {}".format(image_L.shape))
            print("image_L : {}".format(image_R.shape))

            for i, stem in enumerate(self.stems):
                s = stem(s)
                print("stem {} : {}".format( i, s.shape))
                spatial_level+=1
                self._feature_store[spatial_level] = s

            s0 = s1 = s

            for i, cell in enumerate(self.encoder_cells):
                if cell.reduction:
                    s0, s1 = s1, cell([s0, s1], train_mode=train_mode)
                    spatial_level+=1
                    self._feature_store[spatial_level]=s1
                else:
                    s0, s1 = s1, cell([s0, s1], train_mode=train_mode)
                print("encoder {} : {}".format( i, s1.shape))

            prev_pred = self.pred_blocks[0](s1, prev_net_pred)
            predictions = {'all':[], 'final': None}
            predictions['all'].append(prev_pred)

            s0 = s1

            #print("Feature store: ")
            #for (k,v) in self._feature_store.items():
            #    print("Level {} : {}, {}".format(k, v.name, v.shape))

            for i, cell in enumerate(self.decoder_cells):
                s_support = self._feature_store[spatial_level-1] # get a spatial feature from a higher spatial level
                prev_pred_upsampled = nd.ops.resample(prev_pred, reference=s_support, type='LINEAR')
                s0, s1  = s1, cell([s0, s1, prev_pred_upsampled, s_support], train_mode=train_mode)
                prev_pred = self.pred_blocks[i+1](s1, prev_net_pred)
                predictions['all'].append(prev_pred)
                spatial_level-=1
                print("decoder {} : {}".format( i, s1.shape))

            # run aux ops
            if self.full_res:
                s_support = self._feature_store[spatial_level-1] # get a spatial feature from a higher spatial level
                prev_pred, feats_ref_1 = self.refine1([s1, s_support, prev_pred, prev_net_pred])
                print("decoder {} : {}".format( i+1, feats_ref_1.shape))
                spatial_level-=1
                predictions['all'].append(prev_pred)

                s_support = self._feature_store[spatial_level-1] # get a spatial feature from a higher spatial level
                prev_pred, feats_ref_2 = self.refine2([feats_ref_1, self.image_pp(s_support), prev_pred, prev_net_pred])
                print("decoder {} : {}".format( i+2, feats_ref_2.shape))
                predictions['all'].append(prev_pred)

            final = nd.ops.resample(prev_pred, reference=prev_net_pred, type='LINEAR')
            tf.summary.image('diff_disp.pred', nd.ops.to_nhwc(final - prev_net_pred), max_outputs=1)
            tf.summary.image('disp.pred', nd.ops.to_nhwc(final) , max_outputs=1)
            predictions['final'] = final
            return predictions
