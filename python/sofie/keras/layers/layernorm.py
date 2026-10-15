from .. import get_keras_version


def make_keras_layer_norm(layer):
    """
    Create a Keras-compatible layer normalization operation using SOFIE.

    This function takes a dictionary representing a layer normalization layer and its
    attributes and constructs a Keras-compatible layer normalization operation using
    SOFIE. Unlike Batch normalization, Layer normalization used to normalize
    the activations of a layer across the entire layer, independently for each sample in
    the batch.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  gamma, beta, epsilon, data type (assumed to be float), and other
                  relevant information.

    Returns:
    ROperator_BatchNormalization: A SOFIE operator representing the layer normalization operation.
    """
    import sofie

    keras_version = get_keras_version()

    finput = layer["layer_input"]
    foutput = layer["layer_output"]
    attributes = layer["layer_attributes"]
    gamma = attributes["gamma"]
    beta = attributes["beta"]
    axes = attributes["axis"]
    if "_build_input_shape" in attributes.keys():
        num_input_shapes = len(attributes["_build_input_shape"])
    elif "_build_shapes_dict" in attributes.keys():
        num_input_shapes = len(list(attributes["_build_shapes_dict"]["input_shape"]))
    if len(axes) == 1:
        axis = axes[0]
        if axis < 0:
            axis += num_input_shapes
    else:
        raise Exception("TMVA.SOFIE - LayerNormalization layer - parsing different axes at once is not supported")
    layer_dtype = layer["layer_dtype"]
    nx = str(finput[0])
    ny = str(foutput[0])

    if keras_version < "2.16":
        n_scale = gamma.name
        nb = beta.name
    else:
        n_scale = gamma.path
        nb = beta.path

    epsilon = attributes["epsilon"]
    n_inv_std_dev = ""

    if sofie.convert_string_to_type(layer_dtype) == sofie.ETensorType.FLOAT:
        op = sofie.ROperator_LayerNormalization("float")(axis, epsilon, 1, nx, n_scale, nb, ny, "", n_inv_std_dev)
    else:
        raise RuntimeError(
            "SOFIE - Unsupported - Operator BatchNormalization does not yet support input type " + layer_dtype
        )

    return op
