def make_keras_tanh(layer):
    """
    Create a Keras-compatible hyperbolic tangent (tanh) activation operation using SOFIE.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible tanh activation operation using SOFIE.
    Tanh is an activation function that squashes input values to the range between -1 and 1,
    introducing non-linearity in neural networks.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  and data type - must be float.

    Returns:
    ROperator_Tanh: A SOFIE operator representing the tanh activation operation.
    """
    import sofie

    finput = layer["layer_input"]
    foutput = layer["layer_output"]
    layer_dtype = layer["layer_dtype"]
    layer_input_name = finput[0]
    layer_output_name = foutput[0]
    if sofie.convert_string_to_type(layer_dtype) == sofie.ETensorType.FLOAT:
        op = sofie.ROperator_Tanh("float")(layer_input_name, layer_output_name)
        return op
    else:
        raise RuntimeError("SOFIE - Unsupported - Operator Tanh does not yet support input type " + layer_dtype)
