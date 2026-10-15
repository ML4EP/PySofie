def make_keras_elu(layer):
    """
    Create a Keras-compatible exponential linear Unit (ELU) activation operation using SOFIE.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible ELU activation operation using SOFIE.
    ELU is an activation function that modifies only the negative part of ReLU by
    applying an exponential curve. It allows small negative values instead of zeros.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  and data type, which must be float.

    Returns:
    ROperator_Elu: A SOFIE operator representing the ELU activation operation.
    """
    import sofie

    finput = layer["layer_input"]
    foutput = layer["layer_output"]
    layer_dtype = layer["layer_dtype"]
    layer_input_name = finput[0]
    layer_output_name = foutput[0]
    attributes = layer["layer_attributes"]
    if "alpha" in attributes.keys():
        alpha = attributes["alpha"]
    else:
        alpha = 1.0
    if sofie.convert_string_to_type(layer_dtype) == sofie.ETensorType.FLOAT:
        op = sofie.ROperator_Elu("float")(alpha, layer_input_name, layer_output_name)
        return op
    else:
        raise RuntimeError("SOFIE - Unsupported - Operator Relu does not yet support input type " + layer_dtype)
