def make_keras_softmax(layer):
    """
    Create a Keras-compatible softmax activation operation using SOFIE.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible softmax activation operation using SOFIE.
    Softmax is an activation function that converts input values into a probability
    distribution, often used in the output layer of a neural network for multi-class
    classification tasks.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  and data type - must be float.

    Returns:
    ROperator_Softmax: A SOFIE operator representing the softmax activation operation.
    """
    import sofie

    finput = layer["layer_input"]
    foutput = layer["layer_output"]
    layer_dtype = layer["layer_dtype"]
    layer_input_name = finput[0]
    layer_output_name = foutput[0]
    if sofie.convert_string_to_type(layer_dtype) == sofie.ETensorType.FLOAT:
        op = sofie.ROperator_Softmax(-1, layer_input_name, layer_output_name, False)
        return op
    else:
        raise RuntimeError(
            "SOFIE - Unsupported - Operator Softmax does not yet support input type " + layer_dtype
        )
