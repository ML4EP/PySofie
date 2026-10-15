def make_keras_leaky_relu(layer):
    """
    Create a Keras-compatible Leaky ReLU activation operation using SOFIE.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible Leaky ReLU activation operation using SOFIE.
    Leaky ReLU is a variation of the ReLU activation function that allows small negative
    values to pass through, introducing non-linearity while preventing "dying" neurons.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  attributes, and data type - must be float.

    Returns:
    ROperator_LeakyRelu: A SOFIE operator representing the Leaky ReLU activation operation.
    """
    import sofie

    finput = layer["layer_input"]
    foutput = layer["layer_output"]
    layer_dtype = layer["layer_dtype"]
    layer_input_name = finput[0]
    layer_output_name = foutput[0]
    attributes = layer["layer_attributes"]

    if "alpha" in attributes.keys():
        alpha = float(attributes["alpha"])
    elif "negative_slope" in attributes.keys():
        alpha = float(attributes["negative_slope"])
    elif "activation" in attributes.keys():
        alpha = 0.2
    else:
        raise RuntimeError("Failed to extract alpha value from LeakyReLU")

    if sofie.convert_string_to_type(layer_dtype) == sofie.ETensorType.FLOAT:
        op = sofie.ROperator_LeakyRelu("float")(alpha, layer_input_name, layer_output_name)
        return op
    else:
        raise RuntimeError(
            "SOFIE - Unsupported - Operator LeakyRelu does not yet support input type " + layer_dtype
        )
