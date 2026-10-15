# Keras layer type -> the BasicBinary operator of SOFIE
_binary_ops = {
    "Pow": "Pow",
    "Add": "Add",
    "Sub": "Sub",
    "Subtract": "Sub",
    "Mul": "Mul",
    "Multiply": "Mul",
    "Div": "Div",
}


def create_basic_binary(dtype, op_name, name_a, name_b, name_y):
    """Create an element-wise binary operator (Add, Sub/Subtract, Mul/Multiply, Div)."""
    import sofie

    if sofie.convert_string_to_type(dtype) != sofie.ETensorType.FLOAT:
        raise RuntimeError("SOFIE - Unsupported - Operator BasicBinary does not yet support input type " + dtype)
    if op_name not in _binary_ops:
        raise RuntimeError("SOFIE - parsing keras binary layer " + op_name + " is not supported")
    return getattr(sofie, "ROperator_BasicBinary_float_" + _binary_ops[op_name])(name_a, name_b, name_y)


def make_keras_binary(layer):
    inpt = layer["layer_input"]
    return create_basic_binary(layer["layer_dtype"], layer["layer_type"], inpt[0], inpt[1], layer["layer_output"][0])
