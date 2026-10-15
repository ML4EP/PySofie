def make_keras_rnn(layer):
    """
    Create a Keras-compatible RNN (Recurrent Neural Network) layer operation using SOFIE.

    This function takes a dictionary representing an RNN layer and its attributes and
    constructs a Keras-compatible RNN layer operation using SOFIE.
    RNN layers are used to model sequences, and they maintain internal states that are
    updated through recurrent connections.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  layer type, attributes, weights, and data type - must be float.

    Returns:
    ROperator_RNN: A SOFIE operator representing the RNN layer operation.
    """
    import sofie

    # Extract required information from the layer dictionary
    layer_dtype = layer["layer_dtype"]
    finput = layer["layer_input"]
    foutput = layer["layer_output"]
    attributes = layer["layer_attributes"]
    direction = attributes["direction"]
    hidden_size = attributes["hidden_size"]
    layout = int(attributes["layout"])
    name_x = finput[0]
    name_y = foutput[0]
    name_w = layer["layer_weight"][0]
    name_r = layer["layer_weight"][1]
    if len(layer["layer_weight"]) > 2:
        name_b = layer["layer_weight"][2]
    else:
        name_b = ""

    # Check if the provided activation function is supported
    p_activation = attributes["activation"]
    if p_activation.__name__ not in [
        "relu",
        "sigmoid",
        "tanh",
        "softsign",
        "softplus",
    ]:  # avoiding functions with parameters
        raise RuntimeError(
            "SOFIE - Unsupported - Operator RNN does not yet support activation function " + p_activation.__name__
        )

    activations = [p_activation.__name__[0].upper() + p_activation.__name__[1:]]

    # set default values
    activation_alpha = []
    activation_beta = []
    clip = 0.0
    name_y_h = ""
    name_initial_h = ""
    name_seq_len = ""

    if sofie.convert_string_to_type(layer_dtype) == sofie.ETensorType.FLOAT:
        if layer["layer_type"] == "SimpleRNN":
            op = sofie.ROperator_RNN["float"](
                activation_alpha,
                activation_beta,
                activations,
                clip,
                direction,
                hidden_size,
                layout,
                name_x,
                name_w,
                name_r,
                name_b,
                name_seq_len,
                name_initial_h,
                name_y,
                name_y_h,
            )

        elif layer["layer_type"] == "GRU":
            # an additional activation function is required, given by the user
            activations.insert(
                0,
                attributes["recurrent_activation"].__name__[0].upper()
                + attributes["recurrent_activation"].__name__[1:],
            )

            # new variable needed:
            linear_before_reset = attributes["linear_before_reset"]
            op = sofie.ROperator_GRU["float"](
                activation_alpha,
                activation_beta,
                activations,
                clip,
                direction,
                hidden_size,
                layout,
                linear_before_reset,
                name_x,
                name_w,
                name_r,
                name_b,
                name_seq_len,
                name_initial_h,
                name_y,
                name_y_h,
            )

        elif layer["layer_type"] == "LSTM":
            # an additional activation function is required, the first given by the user, the second set to tanh as default
            p_recurrent_activation = attributes["recurrent_activation"]
            if p_activation.__name__ not in [
                "relu",
                "sigmoid",
                "tanh",
                "softsign",
                "softplus",
            ]:  # avoiding functions with parameters
                raise RuntimeError(
                    "SOFIE - Unsupported - Operator RNN does not yet support recurrent activation function "
                    + p_activation.__name__
                )
            p_recurrent_activation_name = p_recurrent_activation.__name__[0].upper() + p_recurrent_activation.__name__[1:]
            activations.insert(0, p_recurrent_activation_name)
            activations.insert(2, "Tanh")

            # new variables needed:
            input_forget = 0
            name_initial_c = ""
            name_p = ""  # No peephole connections in keras LSTM model
            name_y_c = ""
            op = sofie.ROperator_LSTM["float"](
                activation_alpha,
                activation_beta,
                activations,
                clip,
                direction,
                hidden_size,
                input_forget,
                layout,
                name_x,
                name_w,
                name_r,
                name_b,
                name_seq_len,
                name_initial_h,
                name_initial_c,
                name_p,
                name_y,
                name_y_h,
                name_y_c,
            )

        else:
            raise RuntimeError(
                "SOFIE - Unsupported - Operator RNN does not yet support operator type " + layer["layer_type"]
            )
        return op
    else:
        raise RuntimeError("SOFIE - Unsupported - Operator RNN does not yet support input type " + layer_dtype)
