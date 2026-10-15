def make_keras_swish(layer):
    """
    Create a Keras-compatible swish activation operation using SOFIE.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible swish activation operation using SOFIE.
    Swish is an activation function that aims to combine the benefits of ReLU and sigmoid,
    allowing some non-linearity while still keeping positive values unbounded.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  and data type.

    Returns:
    ROperator_Swish: A SOFIE operator representing the swish activation operation.
    """
    import sofie

    finput = layer["layer_input"]
    foutput = layer["layer_output"]
    layer_dtype = layer["layer_dtype"]
    layer_input_name = finput[0]
    layer_output_name = foutput[0]
    if sofie.convert_string_to_type(layer_dtype) == sofie.ETensorType.FLOAT:
        return sofie.ROperator_Swish(layer_input_name, layer_output_name)
    else:
        raise RuntimeError("SOFIE - Unsupported - Operator Swish does not yet support input type " + layer_dtype)
