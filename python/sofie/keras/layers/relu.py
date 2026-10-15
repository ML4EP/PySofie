def make_keras_relu(layer):
    """
    Create a Keras-compatible rectified linear unit (ReLU) activation operation using SOFIE.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible ReLU activation operation using SOFIE.
    ReLU is a popular activation function that replaces all negative values in a tensor
    with zero, while leaving positive values unchanged.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  and data type, which must be float.

    Returns:
    ROperator_Relu: A SOFIE operator representing the ReLU activation operation.
    """
    import sofie

    finput = layer["layer_input"]
    foutput = layer["layer_output"]
    layer_dtype = layer["layer_dtype"]
    layer_input_name = finput[0]
    layer_output_name = foutput[0]
    if sofie.convert_string_to_type(layer_dtype) == sofie.ETensorType.FLOAT:
        op = sofie.ROperator_Relu("float")(layer_input_name, layer_output_name)
        return op
    else:
        raise RuntimeError("SOFIE - Unsupported - Operator Relu does not yet support input type " + layer_dtype)
