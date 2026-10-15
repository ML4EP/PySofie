def make_keras_pooling(layer):
    """
    Create a Keras-compatible pooling layer operation using SOFIE.

    This function takes a dictionary representing a pooling layer and its attributes and
    constructs a Keras-compatible pooling layer operation using SOFIE.
    Pooling layers downsample the input tensor by selecting a representative value from
    a group of neighboring values, either by taking the maximum or the average.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  layer type (the selection rule), the pool size, padding, strides, and data type.

    Returns:
    ROperator_Pool: A SOFIE operator representing the pooling layer operation.
    """
    import sofie

    # Extract attributes from layer data
    layer_dtype = layer["layer_dtype"]
    finput = layer["layer_input"]
    foutput = layer["layer_output"]
    layer_type = layer["layer_type"]
    layer_input_name = finput[0]
    layer_output_name = foutput[0]
    pool_atrr = sofie.RAttributes_Pool()
    attributes = layer["layer_attributes"]
    # Set default values for GlobalAveragePooling2D
    attr_kernel_shape = []
    keras_padding = "valid"
    attr_strides = []
    if layer_type != "GlobalAveragePooling2D":
        attr_kernel_shape = attributes["pool_size"]
        keras_padding = str(attributes["padding"])
        attr_strides = attributes["strides"]

    # Set default values
    attr_dilations = (1, 1)
    fpads = [0, 0, 0, 0, 0, 0]
    pool_atrr.ceil_mode = 0
    pool_atrr.count_include_pad = 0
    pool_atrr.storage_order = 0

    if keras_padding == "valid":
        attr_autopad = "VALID"
    elif keras_padding == "same":
        attr_autopad = "NOTSET"
    else:
        raise RuntimeError(
            "SOFIE - RModel Keras Parser doesn't yet support Pooling layer with padding " + keras_padding
        )
    pool_atrr.dilations = list(attr_dilations)
    pool_atrr.strides = list(attr_strides)
    pool_atrr.pads = fpads
    pool_atrr.kernel_shape = list(attr_kernel_shape)
    pool_atrr.auto_pad = attr_autopad

    # Choose pooling type
    if "Max" in layer_type:
        pool_mode = sofie.PoolOpMode.MaxPool
    elif "AveragePool" in layer_type:
        pool_mode = sofie.PoolOpMode.AveragePool
    elif "GlobalAverage" in layer_type:
        pool_mode = sofie.PoolOpMode.GloabalAveragePool
    else:
        raise RuntimeError(
            "SOFIE - Unsupported - Operator poolong does not yet support pooling type " + layer_type
        )

    # Create operator
    if sofie.convert_string_to_type(layer_dtype) == sofie.ETensorType.FLOAT:
        op = sofie.ROperator_Pool["float"](pool_mode, pool_atrr, layer_input_name, layer_output_name)
        return op
    else:
        raise RuntimeError(
            "SOFIE - Unsupported - Operator Pooling does not yet support input type " + layer_dtype
        )
