from .. import get_keras_version


def make_keras_reshape(layer):
    """
    Create a Keras-compatible reshaping operation using SOFIE.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible reshaping operation using SOFIE. Assumes layerDtype is float.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  name, data type, and other relevant information.

    Returns:
    ROperator_Reshape: A SOFIE operator representing the reshaping operation.
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
    op_mode = sofie.ReshapeOpMode.Reshape
    name_data = finput[0]
    name_output = foutput[0]
    name_shape = flayername + "_shape"
    op = sofie.ROperator_Reshape(op_mode, 0, name_data, name_shape, name_output)
    return op
