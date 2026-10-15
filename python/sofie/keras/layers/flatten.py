from .. import get_keras_version


def make_keras_flatten(layer):
    """
    Create a Keras-compatible flattening operation using SOFIE.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible flattening operation using SOFIE.
    Flattening is the process of converting a multi-dimensional tensor into a
    one-dimensional tensor. Assumes layerDtype is float.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                name, data type, and other relevant information.

    Returns:
    ROperator_Reshape: A SOFIE operator representing the flattening operation.
    """
    import sofie

    keras_version = get_keras_version()

    finput = layer["layer_input"]
    foutput = layer["layer_output"]
    attributes = layer["layer_attributes"]
    if keras_version < "2.16":
        flayername = attributes["_name"]
    else:
        flayername = attributes["name"]
    op_mode = sofie.ReshapeOpMode.Flatten
    name_data = finput[0]
    name_output = foutput[0]
    name_shape = flayername + "_shape"
    op = sofie.ROperator_Reshape(op_mode, 0, name_data, name_shape, name_output)
    return op
