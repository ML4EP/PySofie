def make_keras_identity(layer):
    import sofie

    input = layer["layer_input"]
    output = layer["layer_output"]
    layer_dtype = layer["layer_dtype"]
    layer_input_name = input[0]
    layer_output_name = output[0]
    if sofie.convert_string_to_type(layer_dtype) == sofie.ETensorType.FLOAT:
        op = sofie.ROperator_Identity("float")(layer_input_name, layer_output_name)
        return op
    else:
        raise RuntimeError(
            "SOFIE - Unsupported - Operator Identity does not yet support input type " + layer_dtype
        )
