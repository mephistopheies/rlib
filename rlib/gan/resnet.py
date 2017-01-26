from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import batch_norm
from lasagne.layers import ElemwiseSumLayer, NonlinearityLayer


def add_residual_block(in_layer, nonlin, add_left=False,
                       depth_downscale=4, upscale_Factor=4, first_stride=(1, 1)):
    layer_right = batch_norm(ConvLayer(
        in_layer, num_filters=in_layer.output_shape[1]/depth_downscale,
        filter_size=(1, 1), stride=first_stride,
        pad=(0, 0), nonlinearity=nonlin))

    layer_right = batch_norm(ConvLayer(
        layer_right, num_filters=layer_right.output_shape[1],
        filter_size=(3, 3), stride=(1, 1),
        pad=(1, 1), nonlinearity=nonlin))

    layer_right = batch_norm(ConvLayer(
        layer_right, num_filters=layer_right.output_shape[1]*4,
        filter_size=(1, 1), stride=(1, 1),
        pad=(0, 0), nonlinearity=nonlin))

    if add_left:
        layer_left = batch_norm(ConvLayer(
            in_layer, num_filters=in_layer.output_shape[1]*upscale_Factor,
            filter_size=(1, 1), stride=first_stride,
            pad=(0, 0), nonlinearity=nonlin))
    else:
        layer_left = in_layer

    layer = ElemwiseSumLayer([layer_left, layer_right])
    layer = NonlinearityLayer(layer, nonlinearity=nonlin)

    return layer
