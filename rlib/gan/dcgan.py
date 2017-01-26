import sys
import logging
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage import transform

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import TransposedConv2DLayer as DeconvLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ReshapeLayer
from lasagne.layers import batch_norm
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import rectify, tanh, LeakyRectify, sigmoid
from lasagne.regularization import regularize_network_params, l2

from .helper import sample_normal_noise as sample_noise
from ..graphics import vgg2img, tanh2img
from .resnet import add_residual_block

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


# def augment_do_bn(layer, **kwargs):
#     nonlinearity = getattr(layer, 'nonlinearity', None)
#     if nonlinearity is not None:
#         layer.nonlinearity = lasagne.nonlinearities.identity
#     if hasattr(layer, 'b') and layer.b is not None:
#         del layer.params[layer.b]
#         layer.b = None
#     bn_name = (kwargs.pop('name', None) or
#                (getattr(layer, 'name', None) and layer.name + '_bn'))
#     layer = BatchNormLayer(layer, name=bn_name, **kwargs)
#     if nonlinearity is not None:
#         from .special import NonlinearityLayer
#         nonlin_name = bn_name and bn_name + '_nonlin'
#         layer = NonlinearityLayer(layer, nonlinearity, name=nonlin_name)
#     return layer


def create_johnson_generator(x_in_sym, n_res_blocks=5, n_noise_dim=100, leaky_relu_min=None):
    nonlin = rectify if leaky_relu_min is None else LeakyRectify(leaky_relu_min)
    net = {}
    net['input'] = InputLayer(shape=(None, n_noise_dim), input_var=x_in_sym)
    net['project'] = batch_norm(DenseLayer(net['input'],
                                num_units=3 * 64 * 64, nonlinearity=nonlin))
    net['reshape'] = ReshapeLayer(net['project'], shape=([0], 3, 64, 64))
    net['conv_head_1'] = batch_norm(
        ConvLayer(net['reshape'],
                  num_filters=32,
                  filter_size=(9, 9),
                  stride=(1, 1),
                  pad='same',
                  nonlinearity=nonlin))
    net['conv_head_2'] = batch_norm(
        ConvLayer(net['conv_head_1'],
                  num_filters=64,
                  filter_size=(3, 3),
                  stride=(2, 2),
                  pad='same',
                  nonlinearity=nonlin))
    net['conv_head_3'] = batch_norm(
        ConvLayer(net['conv_head_2'],
                  num_filters=128,
                  filter_size=(3, 3),
                  stride=(2, 2),
                  pad='same',
                  nonlinearity=nonlin))

    prev_layer_name = 'conv_head_3'
    for ix in range(n_res_blocks):
        net['resb_%i_conv1' % ix] = batch_norm(
            DeconvLayer(net[prev_layer_name],
                        num_filters=128,
                        filter_size=(3, 3),
                        stride=(1, 1),
                        crop='same',
                        nonlinearity=nonlin))
        net['resb_%i_conv2' % ix] = batch_norm(
            DeconvLayer(net['resb_%i_conv1' % ix],
                        num_filters=128,
                        filter_size=(3, 3),
                        stride=(1, 1),
                        crop='same',
                        nonlinearity=None))
        net['resb_%i_elsum' % ix] = ElemwiseSumLayer([net['resb_%i_conv2' % ix],
                                                     net[prev_layer_name]])
        prev_layer_name = 'resb_%i_elsum' % ix

    net['tconv_tail_1'] = batch_norm(
        DeconvLayer(net[prev_layer_name],
                    num_filters=64,
                    filter_size=(3, 3),
                    stride=(2, 2),
                    crop='same',
                    output_size=(32, 32),
                    nonlinearity=nonlin))
    net['tconv_tail_2'] = batch_norm(
        DeconvLayer(net['tconv_tail_1'],
                    num_filters=32,
                    filter_size=(3, 3),
                    stride=(2, 2),
                    crop='same',
                    output_size=(64, 64),
                    nonlinearity=nonlin))
    net['tconv_tail_3'] = batch_norm(
        DeconvLayer(net['tconv_tail_2'],
                    num_filters=3,
                    filter_size=(9, 9),
                    stride=(1, 1),
                    crop='same',
                    nonlinearity=tanh))
    return net['tconv_tail_3']


# fc = fixed checkers
def create_johnson_generator_fc(x_in_sym, n_res_blocks=5, n_noise_dim=100, leaky_relu_min=None):
    nonlin = rectify if leaky_relu_min is None else LeakyRectify(leaky_relu_min)
    net = {}
    net['input'] = InputLayer(shape=(None, n_noise_dim), input_var=x_in_sym)
    net['project'] = batch_norm(DenseLayer(net['input'],
                                num_units=3 * 64 * 64, nonlinearity=nonlin))
    net['reshape'] = ReshapeLayer(net['project'], shape=([0], 3, 64, 64))
    net['conv_head_1'] = batch_norm(
        ConvLayer(net['reshape'],
                  num_filters=32,
                  filter_size=(9, 9),
                  stride=(1, 1),
                  pad='same',
                  nonlinearity=nonlin))
    net['conv_head_2'] = batch_norm(
        ConvLayer(net['conv_head_1'],
                  num_filters=64,
                  filter_size=(3, 3),
                  stride=(2, 2),
                  pad='same',
                  nonlinearity=nonlin))
    net['conv_head_3'] = batch_norm(
        ConvLayer(net['conv_head_2'],
                  num_filters=128,
                  filter_size=(3, 3),
                  stride=(2, 2),
                  pad='same',
                  nonlinearity=nonlin))

    prev_layer_name = 'conv_head_3'
    for ix in range(n_res_blocks):
        net['resb_%i_conv1' % ix] = batch_norm(
            DeconvLayer(net[prev_layer_name],
                        num_filters=128,
                        filter_size=(3, 3),
                        stride=(1, 1),
                        crop='same',
                        nonlinearity=nonlin))
        net['resb_%i_conv2' % ix] = batch_norm(
            DeconvLayer(net['resb_%i_conv1' % ix],
                        num_filters=128,
                        filter_size=(3, 3),
                        stride=(1, 1),
                        crop='same',
                        nonlinearity=None))
        net['resb_%i_elsum' % ix] = ElemwiseSumLayer([net['resb_%i_conv2' % ix],
                                                     net[prev_layer_name]])
        prev_layer_name = 'resb_%i_elsum' % ix

    net['tconv_tail_1'] = batch_norm(
        DeconvLayer(net[prev_layer_name],
                    num_filters=64,
                    filter_size=(4, 4),
                    stride=(2, 2),
                    crop=(1, 1),
                    nonlinearity=nonlin))
    net['tconv_tail_2'] = batch_norm(
        DeconvLayer(net['tconv_tail_1'],
                    num_filters=32,
                    filter_size=(4, 4),
                    stride=(2, 2),
                    crop=(1, 1),
                    nonlinearity=nonlin))
    net['conv_tail_3'] = batch_norm(
        ConvLayer(net['tconv_tail_2'],
                  num_filters=3,
                  filter_size=(1, 1),
                  stride=(1, 1),
                  pad='full',
                  nonlinearity=tanh))
    return net['conv_tail_3']


def create_radford_generator(x_in_sym, n_noise_dim=100):
    net = {}
    net['input'] = InputLayer(shape=(None, n_noise_dim), input_var=x_in_sym)
    net['project'] = batch_norm(DenseLayer(net['input'], num_units=1024 * 4 * 4, nonlinearity=rectify))
    net['reshape'] = ReshapeLayer(net['project'], shape=([0], 1024, 4, 4))
    net['tconv_1024x4x4_512x8x8'] = batch_norm(
        DeconvLayer(net['reshape'],
                    num_filters=512,
                    filter_size=(5, 5),
                    stride=(1, 1),
                    nonlinearity=rectify))
    net['tconv_512x8x8_256x16x16'] = batch_norm(
        DeconvLayer(net['tconv_1024x4x4_512x8x8'],
                    num_filters=256,
                    filter_size=(5, 5),
                    stride=(2, 2),
                    crop='same',
                    output_size=(16, 16),
                    nonlinearity=rectify))
    net['tconv_256x16x16_128x32x32'] = batch_norm(
        DeconvLayer(net['tconv_512x8x8_256x16x16'],
                    num_filters=128,
                    filter_size=(5, 5),
                    stride=(2, 2),
                    crop='same',
                    output_size=(32, 32),
                    nonlinearity=rectify))
    net['tconv_128x32x32_3x64x64'] = \
        DeconvLayer(net['tconv_256x16x16_128x32x32'],
                    num_filters=3,
                    stride=(2, 2),
                    crop='same',
                    filter_size=(5, 5),
                    output_size=(64, 64),
                    nonlinearity=tanh)
    return net['tconv_128x32x32_3x64x64']


def create_redford_resnet_generator(x_in_sym, n_res_blocks=5, n_noise_dim=100, leaky_relu_min=None):
    nonlin = rectify if leaky_relu_min is None else LeakyRectify(leaky_relu_min)

    net = {}
    net['input'] = InputLayer(shape=(None, n_noise_dim), input_var=x_in_sym)
    net['head_project'] = batch_norm(DenseLayer(net['input'],
                                     num_units=256*4*4, nonlinearity=nonlin))
    net['head_reshape'] = ReshapeLayer(net['head_project'], shape=([0], 256, 4, 4))
    net['head_tconv_1'] = batch_norm(
        DeconvLayer(net['head_reshape'],
                    num_filters=256,
                    filter_size=(4, 4),
                    stride=(2, 2),
                    crop=(1, 1),
                    nonlinearity=nonlin))
    net['head_tconv_2'] = batch_norm(
        DeconvLayer(net['head_tconv_1'],
                    num_filters=128,
                    filter_size=(4, 4),
                    stride=(2, 2),
                    crop=(1, 1),
                    nonlinearity=nonlin))
    net['head_tconv_3'] = batch_norm(
        DeconvLayer(net['head_tconv_2'],
                    num_filters=64,
                    filter_size=(4, 4),
                    stride=(2, 2),
                    crop=(1, 1),
                    nonlinearity=nonlin))
    net['head_tconv_4'] = batch_norm(
        DeconvLayer(net['head_tconv_3'],
                    num_filters=32,
                    filter_size=(4, 4),
                    stride=(2, 2),
                    crop=(1, 1),
                    nonlinearity=nonlin))

    prev_layer_name = 'head_tconv_4'
    for ix in range(n_res_blocks):
        net['resb_%i_conv1' % ix] = batch_norm(
            ConvLayer(net[prev_layer_name],
                      num_filters=32,
                      filter_size=(3, 3),
                      stride=(1, 1),
                      pad=(1, 1),
                      nonlinearity=nonlin))
        net['resb_%i_conv2' % ix] = batch_norm(
            ConvLayer(net['resb_%i_conv1' % ix],
                      num_filters=32,
                      filter_size=(3, 3),
                      stride=(1, 1),
                      pad=(1, 1),
                      nonlinearity=None))
        net['resb_%i_elsum' % ix] = ElemwiseSumLayer([net['resb_%i_conv2' % ix],
                                                      net[prev_layer_name]])
        prev_layer_name = 'resb_%i_elsum' % ix

    net['tail_conv_1'] = batch_norm(
        ConvLayer(net[prev_layer_name],
                  num_filters=32,
                  filter_size=(3, 3),
                  stride=(1, 1),
                  pad=(1, 1),
                  nonlinearity=nonlin))
    net['tail_conv_2'] = batch_norm(
        ConvLayer(net['tail_conv_1'],
                  num_filters=16,
                  filter_size=(3, 3),
                  stride=(1, 1),
                  pad=(1, 1),
                  nonlinearity=nonlin))
    net['tail_conv_3'] = \
        ConvLayer(net['tail_conv_2'],
                  num_filters=3,
                  filter_size=(3, 3),
                  stride=(1, 1),
                  pad=(1, 1),
                  nonlinearity=tanh)
    return net['tail_conv_3']


def create_radford_generator_fc(x_in_sym, n_noise_dim=100, leaky_relu_min=None):
    nonlin = rectify if leaky_relu_min is None else LeakyRectify(leaky_relu_min)
    net = {}
    net['input'] = InputLayer(shape=(None, n_noise_dim), input_var=x_in_sym)
    net['project'] = batch_norm(DenseLayer(net['input'], num_units=1024 * 4 * 4, nonlinearity=rectify))
    net['reshape'] = ReshapeLayer(net['project'], shape=([0], 1024, 4, 4))
    net['tconv1'] = batch_norm(
        DeconvLayer(net['reshape'],
                    num_filters=512,
                    filter_size=(4, 4),
                    stride=(2, 2),
                    crop=(1, 1),
                    nonlinearity=rectify))
    net['tconv2'] = batch_norm(
        DeconvLayer(net['tconv1'],
                    num_filters=256,
                    filter_size=(4, 4),
                    stride=(2, 2),
                    crop=(1, 1),
                    nonlinearity=rectify))
    net['tconv3'] = batch_norm(
        DeconvLayer(net['tconv2'],
                    num_filters=128,
                    filter_size=(4, 4),
                    stride=(2, 2),
                    crop=(1, 1),
                    nonlinearity=rectify))
    net['tconv4'] = batch_norm(
        DeconvLayer(net['tconv3'],
                    num_filters=64,
                    filter_size=(4, 4),
                    stride=(2, 2),
                    crop=(1, 1),
                    nonlinearity=rectify))
    net['tail_conv1'] = batch_norm(
            ConvLayer(net['tconv4'],
                      num_filters=32,
                      filter_size=(3, 3),
                      stride=(1, 1),
                      pad=(1, 1),
                      nonlinearity=nonlin))
    net['tail_conv2'] = \
        ConvLayer(net['tail_conv1'],
                  num_filters=3,
                  filter_size=(1, 1),
                  stride=(1, 1),
                  pad=(0, 0),
                  nonlinearity=tanh)
    return net['tail_conv2']


def create_discriminator_conv5(x_in_sym, size_img=64, leaky_relu_min=0.2):
    nonlin = rectify if leaky_relu_min is None else LeakyRectify(leaky_relu_min)
    net = {}
    net['input'] = InputLayer(shape=(None, 3, size_img, size_img), input_var=x_in_sym)
    net['conv1'] = ConvLayer(net['input'], num_filters=128,
                             filter_size=(3, 3), stride=(2, 2),
                             pad=(1, 1), nonlinearity=nonlin)
    net['conv2'] = batch_norm(ConvLayer(net['conv1'], num_filters=256,
                                        filter_size=(3, 3), stride=(2, 2),
                                        pad=(1, 1), nonlinearity=nonlin))
    net['conv3'] = batch_norm(ConvLayer(net['conv2'], num_filters=512,
                                        filter_size=(3, 3), stride=(2, 2),
                                        pad=(1, 1), nonlinearity=nonlin))
    net['conv4'] = batch_norm(ConvLayer(net['conv3'], num_filters=1024,
                                        filter_size=(3, 3), stride=(2, 2),
                                        pad=(1, 1), nonlinearity=nonlin))
    net['conv5'] = batch_norm(ConvLayer(net['conv4'], num_filters=1024,
                                        filter_size=(3, 3), stride=(2, 2),
                                        pad=(1, 1), nonlinearity=nonlin))
    net['prob'] = DenseLayer(net['conv5'], num_units=1, nonlinearity=sigmoid)
    return net['prob']


def create_discriminator_conv13(x_in_sym, size_img=64, leaky_relu_min=0.2):
    nonlin = rectify if leaky_relu_min is None else LeakyRectify(leaky_relu_min)
    net = {}
    net['input'] = InputLayer(shape=(None, 3, size_img, size_img), input_var=x_in_sym)
    net['conv1'] = ConvLayer(net['input'], num_filters=64,
                             filter_size=(3, 3), stride=(1, 1),
                             pad=(1, 1), nonlinearity=nonlin)
    net['conv2'] = batch_norm(
        ConvLayer(net['conv1'], num_filters=64,
                  filter_size=(3, 3), stride=(1, 1),
                  pad=(1, 1), nonlinearity=nonlin))
    net['conv3'] = batch_norm(
        ConvLayer(net['conv2'], num_filters=128,
                  filter_size=(3, 3), stride=(2, 2),
                  pad=(1, 1), nonlinearity=nonlin))
    net['conv4'] = batch_norm(
        ConvLayer(net['conv3'], num_filters=128,
                  filter_size=(3, 3), stride=(1, 1),
                  pad=(1, 1), nonlinearity=nonlin))
    net['conv5'] = batch_norm(
        ConvLayer(net['conv4'], num_filters=128,
                  filter_size=(3, 3), stride=(1, 1),
                  pad=(1, 1), nonlinearity=nonlin))
    net['conv6'] = batch_norm(
        ConvLayer(net['conv5'], num_filters=256,
                  filter_size=(3, 3), stride=(2, 2),
                  pad=(1, 1), nonlinearity=nonlin))
    net['conv7'] = batch_norm(
        ConvLayer(net['conv6'], num_filters=256,
                  filter_size=(3, 3), stride=(1, 1),
                  pad=(1, 1), nonlinearity=nonlin))
    net['conv8'] = batch_norm(
        ConvLayer(net['conv7'], num_filters=256,
                  filter_size=(3, 3), stride=(1, 1),
                  pad=(1, 1), nonlinearity=nonlin))
    net['conv9'] = batch_norm(
        ConvLayer(net['conv8'], num_filters=256,
                  filter_size=(3, 3), stride=(1, 1),
                  pad=(1, 1), nonlinearity=nonlin))
    net['conv10'] = batch_norm(
        ConvLayer(net['conv9'], num_filters=512,
                  filter_size=(3, 3), stride=(2, 2),
                  pad=(1, 1), nonlinearity=nonlin))
    net['conv11'] = batch_norm(
        ConvLayer(net['conv10'], num_filters=512,
                  filter_size=(3, 3), stride=(1, 1),
                  pad=(1, 1), nonlinearity=nonlin))
    net['conv12'] = batch_norm(
        ConvLayer(net['conv11'], num_filters=512,
                  filter_size=(3, 3), stride=(1, 1),
                  pad=(1, 1), nonlinearity=nonlin))
    net['conv13'] = batch_norm(
        ConvLayer(net['conv12'], num_filters=512,
                  filter_size=(3, 3), stride=(1, 1),
                  pad=(1, 1), nonlinearity=nonlin))
    net['tail_dense1'] = batch_norm(
        DenseLayer(net['conv13'], num_units=512, nonlinearity=nonlin))
    net['prob'] = DenseLayer(net['tail_dense1'], num_units=1, nonlinearity=sigmoid)
    return net['prob']


def create_discriminator_resnet(x_in_sym, size_img=64, leaky_relu_min=0.2, res_structure=None):
    nonlin = rectify if leaky_relu_min is None else LeakyRectify(leaky_relu_min)
    if res_structure is None:
        res_structure = [3, 4, 6, 3]

    net = {}
    net['input'] = InputLayer(shape=(None, 3, size_img, size_img), input_var=x_in_sym)
    net['head_conv1'] = batch_norm(ConvLayer(
        net['input'], num_filters=64,
        filter_size=(7, 7), stride=(1, 1),
        pad=(3, 3), nonlinearity=nonlin))
    net['head_conv2'] = batch_norm(ConvLayer(
        net['head_conv1'], num_filters=64,
        filter_size=(3, 3), stride=(2, 2),
        pad=(1, 1), nonlinearity=nonlin))

    last_layer_name = 'head_conv2'

    for ix_block, block_size in enumerate(res_structure):
        for ix_layer in range(block_size):
            layer_name = 'res_%i_%i' % (ix_block, ix_layer)
            if ix_layer == 0:
                if last_layer_name == 'head_conv2':

                    net[layer_name] = add_residual_block(
                        net[last_layer_name], nonlin,
                        add_left=True,
                        depth_downscale=1,
                        upscale_Factor=4,
                        first_stride=(1, 1))
                else:
                    net[layer_name] = add_residual_block(
                        net[last_layer_name], nonlin,
                        add_left=True,
                        depth_downscale=2,
                        upscale_Factor=2,
                        first_stride=(2, 2))
            else:
                net[layer_name] = add_residual_block(
                    net[last_layer_name], nonlin,
                    add_left=False,
                    depth_downscale=4,
                    upscale_Factor=1,
                    first_stride=(1, 1))
            last_layer_name = layer_name

    net['tail_conv1'] = batch_norm(ConvLayer(
        net[last_layer_name], num_filters=net[last_layer_name].output_shape[1],
        filter_size=(4, 4), stride=(1, 1),
        pad=(0, 0), nonlinearity=nonlin))
    net['prob'] = DenseLayer(
        net['tail_conv1'],
        num_units=1,
        nonlinearity=sigmoid)

    return net['prob']


def create_discriminator_resnet_small(x_in_sym, size_img=64, leaky_relu_min=0.2):
    nonlin = rectify if leaky_relu_min is None else LeakyRectify(leaky_relu_min)

    net = {}
    net['input'] = InputLayer(shape=(None, 3, size_img, size_img), input_var=x_in_sym)
    net['head_conv1'] = ConvLayer(net['input'], num_filters=32,
                                  filter_size=(1, 1), stride=(1, 1),
                                  pad=(0, 0), nonlinearity=nonlin)
    net['res1'] = add_residual_block(net['head_conv1'], nonlin, add_left=False,
                                     depth_downscale=4,
                                     upscale_Factor=4,
                                     first_stride=(1, 1))
    net['res2'] = add_residual_block(net['res1'], nonlin, add_left=True,
                                     depth_downscale=2,
                                     upscale_Factor=2,
                                     first_stride=(2, 2))
    net['res3'] = add_residual_block(net['res2'], nonlin, add_left=False,
                                     depth_downscale=4,
                                     upscale_Factor=4,
                                     first_stride=(1, 1))
    net['res4'] = add_residual_block(net['res3'], nonlin, add_left=True,
                                     depth_downscale=2,
                                     upscale_Factor=2,
                                     first_stride=(2, 2))
    net['res5'] = add_residual_block(net['res4'], nonlin, add_left=False,
                                     depth_downscale=4,
                                     upscale_Factor=4,
                                     first_stride=(1, 1))
    net['tail_pool1'] = MaxPool2DLayer(net['res5'],
                                       pool_size=(16, 16), stride=(1, 1),
                                       pad=(0, 0))
    # net['res6'] = add_residual_block(net['res5'], nonlin, add_left=True,
    #                                  depth_downscale=2,
    #                                  upscale_Factor=2,
    #                                  first_stride=(2, 2))
    # net['res7'] = add_residual_block(net['res6'], nonlin, add_left=False,
    #                                  depth_downscale=4,
    #                                  upscale_Factor=4,
    #                                  first_stride=(1, 1))
    # net['tail_conv1'] = ConvLayer(net['res7'], num_filters=1024,
    #                               filter_size=(8, 8), stride=(1, 1),
    #                               pad=(0, 0), nonlinearity=nonlin)
    # net['res8'] = add_residual_block(net['res7'], nonlin, add_left=True,
    #                                  depth_downscale=2,
    #                                  upscale_Factor=2,
    #                                  first_stride=(2, 2))
    # net['res9'] = add_residual_block(net['res8'], nonlin, add_left=False,
    #                                  depth_downscale=4,
    #                                  upscale_Factor=4,
    #                                  first_stride=(1, 1))
    # net['res10'] = add_residual_block(net['res9'], nonlin, add_left=True,
    #                                   depth_downscale=2,
    #                                   upscale_Factor=2,
    #                                   first_stride=(2, 2))
    # net['res11'] = add_residual_block(net['res10'], nonlin, add_left=False,
    #                                   depth_downscale=4,
    #                                   upscale_Factor=4,
    #                                   first_stride=(1, 1))
    net['prob'] = DenseLayer(net['tail_pool1'], num_units=1, nonlinearity=sigmoid)

    return net['prob']


class DCGAN:

    def __init__(self, generator_creator, discriminator_creator, load_model=None,
                 lr_gen=0.005, lr_dis=0.005, dis_l2_reg=0.0001, data_label_smoothing=1.0, compile=True,
                 log_level=logging.INFO):
        self.logger = logging.getLogger(str(self))
        self.logger.setLevel(log_level)

        self.logger.info('Creating graph...')
        start_time = time.time()
        self.x_noise_sym = T.matrix('input_noise')
        self.x_img_sym = T.tensor4('input_image')

        self.generator = generator_creator(self.x_noise_sym)
        self.discriminator = discriminator_creator(self.x_img_sym)

        if load_model is not None:
            self.logger.info('Loading weights from %s' % load_model)
            model = pickle.load(open(load_model, 'rb'))
            lasagne.layers.set_all_param_values(self.generator, model['gen'])
            lasagne.layers.set_all_param_values(self.discriminator, model['dis'])

        self.fake_prob = lasagne.layers.get_output(
            self.discriminator,
            inputs=lasagne.layers.get_output(self.generator))

        self.real_prob = lasagne.layers.get_output(self.discriminator)

        self.generator_loss = lasagne.objectives.binary_crossentropy(self.fake_prob, 1).mean()
        # self.discriminator_loss = (
        #     lasagne.objectives.binary_crossentropy(self.real_prob, data_label_smoothing) +
        #     lasagne.objectives.binary_crossentropy(self.fake_prob, 0)).mean()
        self.discriminator_real_loss = \
            lasagne.objectives.binary_crossentropy(self.real_prob, data_label_smoothing).mean()
        self.discriminator_fake_loss = \
            lasagne.objectives.binary_crossentropy(self.fake_prob, 0).mean()
        if dis_l2_reg is not None:
            self.discriminator_l2_reg = dis_l2_reg*regularize_network_params(self.discriminator, l2)
            self.discriminator_real_loss += self.discriminator_l2_reg
            self.discriminator_fake_loss += self.discriminator_l2_reg

        self.generator_params = lasagne.layers.get_all_params(self.generator, trainable=True)
        self.discriminator_params = lasagne.layers.get_all_params(self.discriminator, trainable=True)

        self.generator_updates = lasagne.updates.adam(
            self.generator_loss,
            self.generator_params,
            learning_rate=lr_gen, beta1=0.5, beta2=0.999)
        self.logger.info('len(self.generator_updates) = %i' % len(self.generator_updates))
        # self.discriminator_updates = lasagne.updates.adam(
        #     self.discriminator_loss,
        #     self.discriminator_params,
        #     learning_rate=lr_dis, beta1=0.5, beta2=0.999)
        self.discriminator_real_updates = lasagne.updates.adam(
            self.discriminator_real_loss,
            self.discriminator_params,
            learning_rate=lr_dis, beta1=0.5, beta2=0.999)
        self.logger.info('len(self.discriminator_real_updates) = %i' % len(self.discriminator_real_updates))
        self.discriminator_fake_updates = lasagne.updates.adam(
            self.discriminator_fake_loss,
            self.discriminator_params,
            learning_rate=lr_dis, beta1=0.5, beta2=0.999)
        self.logger.info('len(self.discriminator_fake_updates) = %i' % len(self.discriminator_fake_updates))

        self.sample_image = lasagne.layers.get_output(self.generator, deterministic=True)

        self.logger.info('Graph is created: %0.2f sec' % (time.time() - start_time))

        if compile:
            self._compile()

    def _compile(self):
        self.logger.info('Compiling graph...')
        start_time = time.time()

        self.f_train_generator = theano.function(
            inputs=[self.x_noise_sym],
            outputs=[self.generator_loss, self.fake_prob],
            updates=self.generator_updates)
        # self.f_train_discriminator = theano.function(
        #     inputs=[self.x_noise_sym, self.x_img_sym],
        #     outputs=[self.discriminator_loss, self.fake_prob, self.real_prob],
        #     updates=self.discriminator_updates)
        self.f_train_discriminator_real = theano.function(
            inputs=[self.x_img_sym],
            outputs=[self.discriminator_real_loss, self.real_prob],
            updates=self.discriminator_real_updates)
        self.f_train_discriminator_fake = theano.function(
            inputs=[self.x_noise_sym],
            outputs=[self.discriminator_fake_loss, self.fake_prob],
            updates=self.discriminator_fake_updates)
        self.f_sample_image = theano.function(
            inputs=[self.x_noise_sym],
            outputs=[self.sample_image])

        self.logger.info('Graph is compiled: %0.2f sec' % (time.time() - start_time))

    def _get_default_train_params(self):
        return {
            'n_iters': 5000,
            'n_dis_max_steps': 100,
            'n_dis_batch_size': 500,
            'acc_max_dis': 0.99,
            'alpha_dis': 0.25,
            'n_gen_max_steps': 100,
            'n_gen_batch_size': 500,
            'acc_max_gen': 0.9,
            'alpha_gen': 0.25,
            'n_log_step': 10,
            'n_sample_images': 9,
            'n_noise_dim': 100,
            'real_fake_dis_batch': True,
            'test_samples': None,
            'test_rescale': None,
            'log_dir': None
        }

    def fit(self, data, **kwargs):
        self.logger.info('Start training...')

        params = self._get_default_train_params()
        for k, v in kwargs.items():
            if k in params:
                params[k] = v
            else:
                raise ValueError('Unknown train parameter %s' % k)

        def make_str(k, v):
            if k == 'test_samples':
                return '%s: n=%i' % (k, len(v))
            return '%s: %s' % (k, str(v))

        self.logger.info('\n'.join([make_str(k, v) for (k, v) in params.items()]))

        self.train_log = []
        start_time = time.time()
        for ix_iter in range(params['n_iters']):
            dis_iter_loss = []
            acc_dis = None
            for ix_dis_step in range(params['n_dis_max_steps']):
                x_img_mat = data[np.random.choice(range(data.shape[0]), params['n_dis_batch_size'], replace=False), :]
                x_noise_mat = sample_noise(params['n_dis_batch_size'], params['n_noise_dim'])
                if params['real_fake_dis_batch']:
                    dis_step_loss_1, p_fake = self.f_train_discriminator_fake(x_noise_mat)
                    dis_step_loss_2, p_real = self.f_train_discriminator_real(x_img_mat)
                    dis_step_loss = 0.5*dis_step_loss_1 + 0.5*dis_step_loss_2
                else:
                    dis_step_loss, p_fake, p_real = self.f_train_discriminator(x_noise_mat, x_img_mat)
                acc_dis_step = ((p_fake < 0.5).sum() + (p_real >= 0.5).sum()) / \
                    float(x_img_mat.shape[0] + x_noise_mat.shape[0])
                dis_iter_loss.append((dis_step_loss.tolist(), acc_dis_step))
                if acc_dis is None:
                    acc_dis = acc_dis_step
                    continue
                acc_dis = (1 - params['alpha_dis']) * acc_dis + params['alpha_dis'] * acc_dis_step
                if acc_dis > params['acc_max_dis']:
                    break

            gen_iter_loss = []
            acc_gen = None
            for ix_gen_step in range(params['n_gen_max_steps']):
                x_noise_mat = sample_noise(params['n_gen_batch_size'], params['n_noise_dim'])
                gen_step_loss, p_fake = self.f_train_generator(x_noise_mat)
                acc_gen_step = (p_fake >= 0.5).sum() / float(p_fake.shape[0])
                gen_iter_loss.append((gen_step_loss.tolist(), acc_gen_step))
                if acc_gen is None:
                    acc_gen = acc_gen_step
                    continue
                acc_gen = (1 - params['alpha_gen']) * acc_gen + params['alpha_gen'] * acc_gen_step
                if acc_gen > params['acc_max_gen']:
                    break

            loss_dis, acc_dis_mean = np.array(dis_iter_loss).mean(axis=0)
            loss_gen, acc_gen_mean = np.array(gen_iter_loss).mean(axis=0)
            self.train_log.append({
                'ix_iter': ix_iter,
                'loss_dis': loss_dis,
                'acc_dis_mean': acc_dis_mean,
                'len_dis': len(dis_iter_loss),
                'loss_gen': loss_gen,
                'acc_gen_mean': acc_gen_mean,
                'len_gen': len(gen_iter_loss)
            })

            if ix_iter % params['n_log_step'] == 0:
                log_lines = []
                log_lines.append('Report: %i; time: %i' % (ix_iter, int(time.time() - start_time)))
                log_lines.append('  Dis: %3i | %0.6f | %0.6f ' %
                                 (self.train_log[-1]['len_dis'],
                                  self.train_log[-1]['loss_dis'],
                                  self.train_log[-1]['acc_dis_mean']))
                log_lines.append('  Gen: %3i | %0.6f | %0.6f ' %
                                 (self.train_log[-1]['len_gen'],
                                  self.train_log[-1]['loss_gen'],
                                  self.train_log[-1]['acc_gen_mean']))
                self.logger.info('\n'.join(log_lines))

                if params['log_dir'] is not None:
                    x_noise_mat = sample_noise(params['n_sample_images'], params['n_noise_dim'])
                    sample_images = self.f_sample_image(x_noise_mat)[0]
                    plt.cla()
                    for ix in range(sample_images.shape[0]):
                        plt.subplot(331 + ix * 1)
                        plt.imshow(tanh2img(vgg2img(sample_images[ix, :])).astype(np.uint8))
                        plt.axis('off')
                    plt.savefig(os.path.join(params['log_dir'], 'random_sample_iter_%i.png' % ix_iter), dpi=80)
                    plt.close()

                    if params['test_samples'] is not None:
                        sample_images = self.f_sample_image(params['test_samples'])[0]
                        for ix in range(sample_images.shape[0]):
                            img = tanh2img(vgg2img(sample_images[ix, :])).astype(np.uint8)
                            if params['test_rescale'] is not None:
                                img = transform.rescale(
                                    img,
                                    params['test_rescale'], preserve_range=True).astype(np.uint8)
                            plt.imsave(
                                os.path.join(
                                    params['log_dir'],
                                    'test_sample_%i_iter_%i.png' % (ix, ix_iter)),
                                img)

                    model = {
                        'gen': lasagne.layers.get_all_param_values(self.generator),
                        'dis': lasagne.layers.get_all_param_values(self.discriminator),
                        'log': self.train_log,
                        'params': params
                    }

                    pickle.dump(model,
                                open(os.path.join(params['log_dir'], 'model_%i.pkl' % ix_iter),
                                     'wb'), protocol=-1)

                start_time = time.time()
