def make_keras_concat(layer):
    import sofie

    finput = layer["layer_input"]
    foutput = layer["layer_output"]
    layer_dtype = layer["layer_dtype"]
    attributes = layer["layer_attributes"]
    input = [str(i) for i in finput]
    output = str(foutput[0])
    axis = int(attributes["axis"])
    if sofie.convert_string_to_type(layer_dtype) == sofie.ETensorType.FLOAT:
        op = sofie.ROperator_Concat(input, axis, 0, output)
    else:
        raise RuntimeError("SOFIE - Unsupported - Operator Concat does not yet support input type " + layer_dtype)
    return op
