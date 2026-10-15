def make_keras_sigmoid(layer):
    """
    Create a Keras-compatible sigmoid activation operation using SOFIE.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible sigmoid activation operation using SOFIE.
    Sigmoid is a commonly used activation function that maps input values to the range
    between 0 and 1, providing a way to introduce non-linearity in neural networks.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  and data type - must be float.

    Returns:
    ROperator_Sigmoid: A SOFIE operator representing the sigmoid activation operation.
    """
    import sofie

    finput = layer["layer_input"]
    foutput = layer["layer_output"]
    layer_dtype = layer["layer_dtype"]
    layer_input_name = finput[0]
    layer_output_name = foutput[0]
    if sofie.convert_string_to_type(layer_dtype) == sofie.ETensorType.FLOAT:
        op = sofie.ROperator_Sigmoid("float")(layer_input_name, layer_output_name)
        return op
    else:
        raise RuntimeError(
            "SOFIE - Unsupported - Operator Sigmoid does not yet support input type " + layer_dtype
        )
