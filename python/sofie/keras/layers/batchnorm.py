from .. import get_keras_version


def make_keras_batch_norm(layer):
    """
    Create a Keras-compatible batch normalization operation using SOFIE.

    This function takes a dictionary representing a batch normalization layer and its
    attributes and constructs a Keras-compatible batch normalization operation using
    SOFIE. Batch normalization is used to normalize the activations of
    a neural network, typically applied after the convolutional or dense layers.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  gamma, beta, moving mean, moving variance, epsilon,
                  momentum, data type (assumed to be float), and other relevant information.

    Returns:
    ROperator_BatchNormalization: A SOFIE operator representing the batch normalization operation.
    """
    import sofie

    keras_version = get_keras_version()

    finput = layer["layer_input"]
    foutput = layer["layer_output"]
    attributes = layer["layer_attributes"]
    gamma = attributes["gamma"]
    beta = attributes["beta"]
    moving_mean = attributes["moving_mean"]
    moving_variance = attributes["moving_variance"]
    layer_dtype = layer["layer_dtype"]
    nx = str(finput[0])
    ny = str(foutput[0])

    if keras_version < "2.16":
        n_scale = gamma.name
        nb = beta.name
        n_mean = moving_mean.name
        n_var = moving_variance.name
    else:
        n_scale = gamma.path
        nb = beta.path
        n_mean = moving_mean.path
        n_var = moving_variance.path

    epsilon = attributes["epsilon"]
    momentum = attributes["momentum"]

    if sofie.convert_string_to_type(layer_dtype) == sofie.ETensorType.FLOAT:
        op = sofie.ROperator_BatchNormalization("float")(epsilon, momentum, 0, nx, n_scale, nb, n_mean, n_var, ny)
    else:
        raise RuntimeError(
            "SOFIE - Unsupported - Operator BatchNormalization does not yet support input type " + layer_dtype
        )
    return op
