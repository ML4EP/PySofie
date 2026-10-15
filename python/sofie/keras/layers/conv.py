import math

from .. import get_keras_version


def make_keras_conv(layer):
    """
    Create a Keras-compatible convolutional layer operation using SOFIE.

    This function takes a dictionary representing a convolutional layer and its attributes and
    constructs a Keras-compatible convolutional layer operation using SOFIE.
    A convolutional layer applies a convolution operation between the input tensor and a set
    of learnable filters (kernels).

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  data type (must be float), weight and bias name, kernel size, dilations, padding and strides.
                  When padding is same (keep in the same dimensions), the padding shape is calculated.

    Returns:
    ROperator_Conv: A SOFIE operator representing the convolutional layer operation.
    """
    import sofie

    keras_version = get_keras_version()

    finput = layer["layer_input"]
    foutput = layer["layer_output"]
    layer_dtype = layer["layer_dtype"]
    layer_input_name = finput[0]
    layer_output_name = foutput[0]
    attributes = layer["layer_attributes"]
    weight_names = layer["layer_weight"]
    kernel_name = weight_names[0]
    bias_name = weight_names[1]
    attr_dilations = attributes["dilation_rate"]
    attr_group = int(attributes["groups"])
    attr_kernel_shape = attributes["kernel_size"]
    keras_padding = str(attributes["padding"])
    attr_strides = attributes["strides"]
    attr_pads = []

    if keras_padding == "valid":
        attr_autopad = "VALID"
    elif keras_padding == "same":
        attr_autopad = "NOTSET"
        if keras_version < "2.16":
            input_shape = attributes["_build_input_shape"]
        else:
            input_shape = attributes["_build_shapes_dict"]["input_shape"]
        if layer.get("channels_last", True):
            input_height = input_shape[1]
            input_width = input_shape[2]
        else:
            input_height = input_shape[2]
            input_width = input_shape[3]
        output_height = math.ceil(float(input_height) / float(attr_strides[0]))
        output_width = math.ceil(float(input_width) / float(attr_strides[1]))
        effective_kh = attr_dilations[0] * (attr_kernel_shape[0] - 1) + 1
        effective_kw = attr_dilations[1] * (attr_kernel_shape[1] - 1) + 1
        padding_height = max((output_height - 1) * attr_strides[0] + effective_kh - input_height, 0)
        padding_width = max((output_width - 1) * attr_strides[1] + effective_kw - input_width, 0)
        padding_top = math.floor(padding_height / 2)
        padding_bottom = padding_height - padding_top
        padding_left = math.floor(padding_width / 2)
        padding_right = padding_width - padding_left
        attr_pads = [padding_top, padding_left, padding_bottom, padding_right]
    else:
        raise RuntimeError(
            "SOFIE - RModel Keras Parser doesn't yet supports Convolution layer with padding " + keras_padding
        )
    if sofie.convert_string_to_type(layer_dtype) == sofie.ETensorType.FLOAT:
        op = sofie.ROperator_Conv["float"](
            attr_autopad,
            attr_dilations,
            attr_group,
            attr_kernel_shape,
            attr_pads,
            attr_strides,
            layer_input_name,
            kernel_name,
            bias_name,
            layer_output_name,
        )
        return op
    else:
        raise RuntimeError("SOFIE - Unsupported - Operator Conv does not yet support input type " + layer_dtype)
