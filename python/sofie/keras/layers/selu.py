def make_keras_selu(layer):
    """
    Create a Keras-compatible scaled exponential linear unit (SeLU) activation operation using SOFIE.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible SeLU activation operation using SOFIE.
    SeLU is a type of activation function that introduces self-normalizing properties
    to the neural network.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  and data type - must be float32.

    Returns:
    ROperator_Selu: A SOFIE operator representing the SeLU activation operation.
    """
    import sofie

    finput = layer["layer_input"]
    foutput = layer["layer_output"]
    layer_dtype = layer["layer_dtype"]
    layer_input_name = finput[0]
    layer_output_name = foutput[0]
    if sofie.convert_string_to_type(layer_dtype) == sofie.ETensorType.FLOAT:
        op = sofie.ROperator_Selu("float")(layer_input_name, layer_output_name)
        return op
    else:
        raise RuntimeError("SOFIE - Unsupported - Operator Selu does not yet support input type " + layer_dtype)
