def make_keras_permute(layer):
    """
    Create a Keras-compatible permutation operation using SOFIE.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible permutation operation using SOFIE.
    Permutation is an operation that rearranges the dimensions of a tensor based on
    specified dimensions.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  attributes, and data type - must be float.

    Returns:
    ROperator_Transpose: A SOFIE operator representing the permutation operation.
    """
    import sofie

    finput = layer["layer_input"]
    foutput = layer["layer_output"]
    layer_dtype = layer["layer_dtype"]
    layer_input_name = finput[0]
    layer_output_name = foutput[0]
    attributes = layer["layer_attributes"]
    attribute_permute = list(attributes["dims"])
    if sofie.convert_string_to_type(layer_dtype) == sofie.ETensorType.FLOAT:
        if len(attribute_permute) > 0:
            attribute_permute = [0] + attribute_permute  # for the batch dimension from the input
        op = sofie.ROperator_Transpose(attribute_permute, layer_input_name, layer_output_name)
        return op
    else:
        raise RuntimeError(
            "SOFIE - Unsupported - Operator Transpose does not yet support input type " + layer_dtype
        )
