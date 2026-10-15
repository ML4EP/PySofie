def make_keras_dense(layer):
    """
    Create a Keras-compatible dense (fully connected) layer operation using SOFIE.

    This function takes a dictionary representing a dense layer and its attributes and
    constructs a Keras-compatible dense (fully connected) layer operation using SOFIE.
    A dense layer applies a matrix multiplication between the input tensor and weight matrix,
    and adds a bias term.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  layer weight names, and data type - must be float.

    Returns:
    ROperator_Gemm: A SOFIE operator representing the dense layer operation.
    """
    import sofie

    finput = layer["layer_input"]
    foutput = layer["layer_output"]
    layer_dtype = layer["layer_dtype"]
    layer_input_name = finput[0]
    layer_output_name = foutput[0]
    weight_names = layer["layer_weight"]
    kernel_name = weight_names[0]
    bias_name = weight_names[1]
    attr_alpha = 1.0
    attr_beta = 1.0
    attr_trans_a = 0
    attr_trans_b = 0
    if sofie.convert_string_to_type(layer_dtype) == sofie.ETensorType.FLOAT:
        op = sofie.ROperator_Gemm["float"](
            attr_alpha, attr_beta, attr_trans_a, attr_trans_b, layer_input_name, kernel_name, bias_name, layer_output_name
        )
        return op
    else:
        raise RuntimeError("SOFIE - Unsupported - Operator Gemm does not yet support input type " + layer_dtype)
