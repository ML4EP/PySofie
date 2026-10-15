import os
import time

from .._compat import add_initialized_tensor
from . import get_keras_version
from .layers.batchnorm import make_keras_batch_norm
from .layers.binary import make_keras_binary
from .layers.concat import make_keras_concat
from .layers.conv import make_keras_conv
from .layers.dense import make_keras_dense
from .layers.elu import make_keras_elu
from .layers.flatten import make_keras_flatten
from .layers.layernorm import make_keras_layer_norm
from .layers.leaky_relu import make_keras_leaky_relu
from .layers.permute import make_keras_permute
from .layers.pooling import make_keras_pooling
from .layers.relu import make_keras_relu
from .layers.reshape import make_keras_reshape
from .layers.selu import make_keras_selu
from .layers.sigmoid import make_keras_sigmoid
from .layers.softmax import make_keras_softmax
from .layers.swish import make_keras_swish
from .layers.tanh import make_keras_tanh


def make_keras_activation(layer):
    attributes = layer["layer_attributes"]
    activation = attributes["activation"]
    layer_activation = str(activation.__name__)

    if layer_activation in map_keras_layer.keys():
        return map_keras_layer[layer_activation](layer)
    else:
        raise Exception("TMVA.SOFIE - parsing keras activation layer " + layer_activation + " is not yet supported")


# Set global dictionaries, mapping layers to corresponding functions that create their ROperator instances
map_keras_layer = {
    "Activation": make_keras_activation,
    "Permute": make_keras_permute,
    "BatchNormalization": make_keras_batch_norm,
    "LayerNormalization": make_keras_layer_norm,
    "Reshape": make_keras_reshape,
    "Flatten": make_keras_flatten,
    "Concatenate": make_keras_concat,
    "swish": make_keras_swish,
    "silu": make_keras_swish,
    "Add": make_keras_binary,
    "Subtract": make_keras_binary,
    "Multiply": make_keras_binary,
    "Softmax": make_keras_softmax,
    "tanh": make_keras_tanh,
    #  "Identity": make_keras_identity,
    #  "Dropout": make_keras_identity,
    "ReLU": make_keras_relu,
    "relu": make_keras_relu,
    "ELU": make_keras_elu,
    "elu": make_keras_elu,
    "selu": make_keras_selu,
    "sigmoid": make_keras_sigmoid,
    "LeakyReLU": make_keras_leaky_relu,
    "leaky_relu": make_keras_leaky_relu,
    "softmax": make_keras_softmax,
    "MaxPooling2D": make_keras_pooling,
    "AveragePooling2D": make_keras_pooling,
    "GlobalAveragePooling2D": make_keras_pooling,
    #  "SimpleRNN": make_keras_rnn,
    #  "GRU": make_keras_rnn,
    #  "LSTM": make_keras_rnn,
}

map_keras_layer_with_activation = {"Dense": make_keras_dense, "Conv2D": make_keras_conv}


def add_layer_into_rmodel(rmodel, layer_data):
    """
    Add a Keras layer operation to an existing RModel using SOFIE.

    This function takes an existing RModel and a dictionary representing a Keras layer
    and its attributes, and adds the corresponding layer operation to the RModel using
    SOFIE. The function supports various types of Keras layers, including
    those with or without activation functions.

    Parameters:
    rmodel (RModel): An existing RModel to which the layer operation will be added.
    layer_data (dict): A dictionary containing layer information including type,
                      attributes, input, output, and layer data type.

    Returns:
    RModel: The updated RModel after adding the layer operation.

    Raises exception: If the provided layer type or activation function is not supported.
    """
    import numpy as np
    import sofie

    keras_version = get_keras_version()

    layer_type = layer_data["layer_type"]

    # reshape and flatten layers don't have weights, but they need constant tensor for the shape
    if layer_type == "Reshape" or layer_type == "Flatten":
        attributes = layer_data["layer_attributes"]
        if keras_version < "2.16":
            layer_name = attributes["_name"]
        else:
            layer_name = attributes["name"]

        if layer_type == "Reshape":
            target_shape = np.asarray(attributes["target_shape"]).astype("int64")
            target_shape = np.insert(target_shape, 0, 1)
        else:
            if "_build_input_shape" in attributes.keys():
                input_shape = attributes["_build_input_shape"]
            elif "_build_shapes_dict" in attributes.keys():
                input_shape = list(attributes["_build_shapes_dict"]["input_shape"])
            else:
                raise RuntimeError("Failed to extract build input shape from " + layer_type + " layer")
            target_shape = [sofie.convert_shape_to_length(input_shape[1:])]
            target_shape = np.asarray(target_shape)

        # since the add_initialized_tensor method in RModel requires unique pointer, we call a helper function
        # in c++ that does the conversion from a regular pointer to unique one in c++
        shape_tensor_name = layer_name + "_shape"
        add_initialized_tensor(rmodel, shape_tensor_name, sofie.ETensorType.UINT64, [len(target_shape)], target_shape)

    # These layers only have one operator - excluding the recurrent layers, in which the activation function(s)
    # are included in the recurrent operator
    if layer_type in map_keras_layer.keys():
        attributes = layer_data["layer_attributes"]
        inputs = layer_data["layer_input"]
        outputs = layer_data["layer_output"]
        if keras_version < "2.16":
            layer_name = attributes["_name"]
        else:
            layer_name = attributes["name"]

        # Convoltion/Pooling layers in keras by default assume the channels dimension is the
        # last one, while in onnx (and SOFIE's RModel) it is the first one (other than batch
        # size), so a transpose is needed before and after the pooling, if the data format is
        # channels last (can be set to channels first by the user). In case of MaxPool2D and
        # Conv2D (with linear activation) channels last, the transpose layers are added as:

        #                   input                   output
        # transpose layer   input_layer_name        layer_name + PreTrans
        # actual layer      layer_name + PreTrans   layer_name + PostTrans
        # transpose layer   layer_name + PostTrans  output_layer_name

        layer_output = outputs[0]
        if layer_type == "GlobalAveragePooling2D":
            if layer_data["channels_last"]:
                op = sofie.ROperator_Transpose([0, 3, 1, 2], inputs[0], layer_name + "PreTrans")
                rmodel.add_operator(op)
                inputs[0] = layer_name + "PreTrans"
            outputs[0] = layer_name + "Squeeze"
            rmodel.add_operator(map_keras_layer[layer_type](layer_data))
            op = sofie.ROperator_Reshape(sofie.ReshapeOpMode.Squeeze, [2, 3], layer_name + "Squeeze", layer_output)
            rmodel.add_operator(op)

        # Similar case is with Batchnorm, ONNX assumes that the 'axis' is always 1, but Keras
        # gives the user the choice of specifying it. So, we have to transpose the input layer
        # as 'axis' as the first dimension, apply the BatchNormalization operator and then
        # again tranpose it to bring back the original dimensions
        elif layer_type == "BatchNormalization":
            if "_build_input_shape" in attributes.keys():
                num_input_shapes = len(attributes["_build_input_shape"])
            elif "_build_shapes_dict" in attributes.keys():
                num_input_shapes = len(list(attributes["_build_shapes_dict"]["input_shape"]))

            axis = attributes["axis"]
            axis = axis[0] if isinstance(axis, list) else axis
            if axis < 0:
                axis += num_input_shapes
            attr_perm = list(range(0, num_input_shapes))
            attr_perm[1] = axis
            attr_perm[axis] = 1
            op = sofie.ROperator_Transpose(attr_perm, inputs[0], layer_name + "PreTrans")
            rmodel.add_operator(op)
            inputs[0] = layer_name + "PreTrans"
            outputs[0] = layer_name + "PostTrans"
            rmodel.add_operator(map_keras_layer[layer_type](layer_data))
            op = sofie.ROperator_Transpose(attr_perm, layer_name + "PostTrans", layer_output)
            rmodel.add_operator(op)

        elif layer_type == "MaxPooling2D" or layer_type == "AveragePooling2D":
            if layer_data["channels_last"]:
                op = sofie.ROperator_Transpose([0, 3, 1, 2], inputs[0], layer_name + "PreTrans")
                rmodel.add_operator(op)
                inputs[0] = layer_name + "PreTrans"
                outputs[0] = layer_name + "PostTrans"
            rmodel.add_operator(map_keras_layer[layer_type](layer_data))
            if layer_data["channels_last"]:
                op = sofie.ROperator_Transpose([0, 2, 3, 1], layer_name + "PostTrans", layer_output)
                rmodel.add_operator(op)

        else:
            rmodel.add_operator(map_keras_layer[layer_type](layer_data))

        return rmodel

    # These layers require two operators - dense/conv and their activation function
    elif layer_type in map_keras_layer_with_activation.keys():
        attributes = layer_data["layer_attributes"]
        if keras_version < "2.16":
            layer_name = attributes["_name"]
        else:
            layer_name = attributes["name"]
        p_activation = attributes["activation"]
        layer_activation = p_activation.__name__
        if layer_activation in ["selu", "sigmoid"]:
            rmodel.add_needed_std_lib("cmath")

        # if there is an activation function after the layer
        if layer_activation != "linear":
            if layer_activation not in map_keras_layer.keys():
                raise Exception(
                    "TMVA.SOFIE - parsing keras activation function " + layer_activation + " is not yet supported"
                )
            outputs = layer_data["layer_output"]
            inputs = layer_data["layer_input"]
            activation_layer_output = outputs[0]

            # like pooling, convolutional layer from keras requires transpose before and after to match
            # the onnx format
            # if the data format is channels last (can be set to channels first by the user).
            if layer_type == "Conv2D":
                if layer_data["channels_last"]:
                    op = sofie.ROperator_Transpose([0, 3, 1, 2], inputs[0], layer_name + "PreTrans")
                    rmodel.add_operator(op)
                    inputs[0] = layer_name + "PreTrans"
                    layer_data["layer_input"] = inputs
            outputs[0] = layer_name + layer_type
            layer_data["layer_output"] = outputs
            op = map_keras_layer_with_activation[layer_type](layer_data)
            rmodel.add_operator(op)
            activation_layer_input = layer_name + layer_type
            if layer_type == "Conv2D":
                if layer_data["channels_last"]:
                    op = sofie.ROperator_Transpose(
                        [0, 2, 3, 1], layer_name + layer_type, layer_name + "PostTrans"
                    )
                    rmodel.add_operator(op)
                    activation_layer_input = layer_name + "PostTrans"

            # Adding the activation function
            inputs[0] = activation_layer_input
            outputs[0] = activation_layer_output
            layer_data["layer_input"] = inputs
            layer_data["layer_output"] = outputs

            rmodel.add_operator(map_keras_layer[layer_activation](layer_data))

        else:  # if layer is conv and the activation is linear, we need to add transpose before and after
            if layer_type == "Conv2D":
                inputs = layer_data["layer_input"]
                outputs = layer_data["layer_output"]
                layer_output = outputs[0]
                if layer_data["channels_last"]:
                    op = sofie.ROperator_Transpose([0, 3, 1, 2], inputs[0], layer_name + "PreTrans")
                    rmodel.add_operator(op)
                    inputs[0] = layer_name + "PreTrans"
                    layer_data["layer_input"] = inputs
                    outputs[0] = layer_name + "PostTrans"
            rmodel.add_operator(map_keras_layer_with_activation[layer_type](layer_data))
            if layer_type == "Conv2D":
                if layer_data["channels_last"]:
                    op = sofie.ROperator_Transpose([0, 2, 3, 1], layer_name + "PostTrans", layer_output)
                    rmodel.add_operator(op)
        return rmodel
    else:
        raise Exception("TMVA.SOFIE - parsing keras layer " + layer_type + " is not yet supported")


def parse(filename, batch_size=1):  # If a model does not have a defined batch size, then assuming it is 1

    # TensorFlow/Keras is too fragile to import unconditionally. As its presence might break other
    # usecases and importing keras globally will slow down importing this package, which is not desired. For this,
    # we import keras within the functions instead of importing it at the start of the file.
    # So, whenever the parser function is called, only then keras will be imported.
    try:
        import keras
        import numpy as np
    except ImportError as e:
        raise ImportError(
            f"sofie.keras.parse needs keras, tensorflow and numpy ('{e.name}' is missing): "
            'install them with pip install "sofie[keras]"'
        ) from e
    import sofie

    keras_version = get_keras_version()

    # Check if file exists
    if not os.path.exists(filename):
        raise RuntimeError("Model file {} not found!".format(filename))

    # load model
    keras_model = keras.models.load_model(filename)
    keras_model.load_weights(filename)

    # create new RModel object
    sep = "/"
    if os.name == "nt":
        sep = "\\"

    isep = filename.rfind(sep)
    filename_nodir = filename
    if isep != -1:
        filename_nodir = filename[isep + 1 :]

    ttime = time.time()
    gmt_time = time.gmtime(ttime)
    parsetime = time.asctime(gmt_time)

    rmodel = sofie.RModel(filename_nodir, parsetime)

    print("sofie.keras: parsing model ", filename)

    # iterate over the layers and add them to the RModel
    # in case of keras 3.x (particularly in sequential models), the layer input and output name conventions are
    # different from keras 2.x. In keras 2.x, the layer input name is consistent with previous layer's output
    # name. For e.g., if the sequence of layers is dense -> maxpool, the input and output layer names would be:
    #           layer   |       name
    # input     dense   |   keras_tensor_1
    # output    dense   |   keras_tensor_2 --
    #                   |                    |=> layer names match
    # input     maxpool |   keras_tensor_2 --
    # output    maxpool |   keras_tensor_3
    #
    # but in case of keras 3.x, this changes.
    #           layer   |       name
    # input     dense   |   keras_tensor_1
    # output    dense   |   keras_tensor_2 --
    #                   |                    |=> different layer names
    # input     maxpool |   keras_tensor_3 --
    # output    maxpool |   keras_tensor_4
    #
    # hence, we need to add a custom layer iterator, which would replace the suffix of the layer's input
    # and output names
    layer_iter = 0
    is_functional_model = True if keras_model.__class__.__name__ == "Functional" else False
    for layer in keras_model.layers:
        layer_data = {}
        layer_data["layer_type"] = layer.__class__.__name__
        layer_data["layer_attributes"] = layer.__dict__
        # get input names for layer
        if keras_version < "2.16" or is_functional_model:
            if "input_layer" in layer.name:
                layer_data["layer_input"] = layer.name
            else:
                layer_data["layer_input"] = (
                    [x.name for x in layer.input] if isinstance(layer.input, list) else [layer.input.name]
                )
        else:
            # case of Keras3 Sequential model : in this case output of layer is input to following one, but names can be different
            if "input_layer" in layer.input.name:
                layer_data["layer_input"] = [layer.input.name]
            else:
                if layer_iter == 0:
                    input_layer_name = "tensor_input_" + layer.name
                else:
                    input_layer_name = "tensor_output_" + keras_model.layers[layer_iter - 1].name
                layer_data["layer_input"] = [input_layer_name]
        # get output names of layer
        if keras_version < "2.16" or is_functional_model:
            layer_data["layer_output"] = (
                [x.name for x in layer.output] if isinstance(layer.output, list) else [layer.output.name]
            )
        else:
            # sequential model in Keras3
            output_layer_name = "tensor_output_" + layer.name
            layer_data["layer_output"] = (
                [x.name for x in layer.output] if isinstance(layer.output, list) else [output_layer_name]
            )

        layer_iter += 1
        layer_type = layer_data["layer_type"]
        layer_data["layer_dtype"] = layer.dtype

        if len(layer.weights) > 0:
            if keras_version < "2.16":
                layer_data["layer_weight"] = [x.name for x in layer.weights]
            else:
                layer_data["layer_weight"] = [x.path for x in layer.weights]
        else:
            layer_data["layer_weight"] = []

        # for convolutional and pooling layers we need to know the format of the data
        if layer_data["layer_type"] in ["Conv2D", "MaxPooling2D", "AveragePooling2D", "GlobalAveragePooling2D"]:
            layer_data["channels_last"] = True if layer.data_format == "channels_last" else False

        # for recurrent type layers we need to extract additional unique information
        if layer_data["layer_type"] in ["SimpleRNN", "LSTM", "GRU"]:
            layer_data["layer_attributes"]["activation"] = layer.activation
            layer_data["layer_attributes"]["direction"] = "backward" if layer.go_backwards else "forward"
            layer_data["layer_attributes"]["units"] = layer.units
            layer_data["layer_attributes"]["layout"] = layer.input.shape[0] is None
            layer_data["layer_attributes"]["hidden_size"] = layer.output.shape[-1]

            # for GRU and LSTM we need to extract an additional activation function
            if layer_data["layer_type"] != "SimpleRNN":
                layer_data["layer_attributes"]["recurrent_activation"] = layer.recurrent_activation

            # for GRU there are two variants of the reset gate location, we need to know which one is it
            if layer_data["layer_type"] == "GRU":
                layer_data["layer_attributes"]["linear_before_reset"] = (
                    1 if layer.reset_after and layer.recurrent_activation.__name__ == "sigmoid" else 0
                )

        # Ignoring the input layer of the model
        if layer_type == "InputLayer":
            continue

        # Adding any required routines depending on the Layer types for generating inference code.
        if layer_type == "Dense":
            rmodel.add_blas_routines(["Gemm", "Gemv"])
        elif layer_type == "BatchNormalization":
            rmodel.add_blas_routines(["Copy", "Axpy"])
        elif layer_type == "Conv1D" or layer_type == "Conv2D" or layer_type == "Conv3D":
            rmodel.add_blas_routines(["Gemm", "Axpy"])
        rmodel = add_layer_into_rmodel(rmodel, layer_data)

    # Extracting model's weights
    weight = []
    for idx in range(len(keras_model.get_weights())):
        weight_prop = {}
        if keras_version < "2.16":
            weight_prop["name"] = keras_model.weights[idx].name
        else:
            weight_prop["name"] = keras_model.weights[idx].path
        weight_prop["dtype"] = keras_model.get_weights()[idx].dtype.name
        if "conv" in weight_prop["name"] and keras_model.weights[idx].shape.ndims == 4:
            weight_prop["value"] = keras_model.get_weights()[idx].transpose((3, 2, 0, 1)).copy()
        else:
            weight_prop["value"] = keras_model.get_weights()[idx]
        weight.append(weight_prop)

    # Traversing through all the Weight tensors
    for weight_iter in range(len(weight)):
        weight_tensor = weight[weight_iter]
        weight_name = weight_tensor["name"]
        weight_dtype = sofie.convert_string_to_type(weight_tensor["dtype"])
        weight_tensor_value = weight_tensor["value"]
        weight_tensor_size = 1
        weight_tensor_shape = []

        # IS IT BATCH SIZE? CHECK ONNX
        if (
            "simple_rnn" in weight_name
            or "lstm" in weight_name
            or ("gru" in weight_name and "bias" not in weight_name)
        ):
            weight_tensor_shape.append(1)

        # Building the shape vector and finding the tensor size
        for j in range(len(weight_tensor_value.shape)):
            weight_tensor_shape.append(weight_tensor_value.shape[j])
            weight_tensor_size *= weight_tensor_value.shape[j]

        if weight_dtype == sofie.ETensorType.FLOAT:
            weight_array = weight_tensor_value

            # weights conversion format between keras and onnx for lstm: the order of the different
            # elements (input, output, forget, cell) inside the vector/matrix is different
            if "lstm" in weight_name:
                if "kernel" in weight_name:
                    units = int(weight_array.shape[1] / 4)
                    w_f = weight_array[:, units : units * 2].copy()
                    w_c = weight_array[:, units * 2 : units * 3].copy()
                    w_o = weight_array[:, units * 3 :].copy()
                    weight_array[:, units : units * 2] = w_o
                    weight_array[:, units * 2 : units * 3] = w_f
                    weight_array[:, units * 3 :] = w_c
                else:  # bias
                    units = int(weight_array.shape[0] / 4)
                    w_f = weight_array[units : units * 2].copy()
                    w_c = weight_array[units * 2 : units * 3].copy()
                    w_o = weight_array[units * 3 :].copy()
                    weight_array[units : units * 2] = w_o
                    weight_array[units * 2 : units * 3] = w_f
                    weight_array[units * 3 :] = w_c

            # need to make specific adjustments for recurrent weights and biases
            if "simple_rnn" in weight_name or "lstm" in weight_name or "gru" in weight_name:
                # reshaping weight matrices for recurrent layers due to keras-onnx inconsistencies
                if "kernel" in weight_name:
                    weight_array = np.transpose(weight_array)
                    weight_tensor_shape[1], weight_tensor_shape[2] = weight_tensor_shape[2], weight_tensor_shape[1]

                data = weight_array.flatten()

                # the recurrent bias and the cell bias can be the same, in which case we need to add a
                # vector of zeros for the recurrent bias
                if "bias" in weight_name and len(data.shape) == 1:
                    weight_tensor_shape[1] *= 2
                    rbias = data.copy() * 0
                    data = np.concatenate((data, rbias))

            else:
                data = weight_array.flatten()
            add_initialized_tensor(rmodel, weight_name, sofie.ETensorType.FLOAT, weight_tensor_shape, data)
        else:
            raise TypeError("Type error: TMVA SOFIE does not yet support data layer type: " + weight_dtype)

    # Extracting input tensor info
    if keras_version < "2.16":
        p_inputs = keras_model.input_names
    else:
        p_inputs = [x.name for x in keras_model.inputs]

    p_input_shape = (
        keras_model.input_shape if isinstance(keras_model.input_shape, list) else [keras_model.input_shape]
    )
    p_input_dtype = []
    for idx in range(len(keras_model.inputs)):
        dtype = keras_model.inputs[idx].dtype.__str__()
        if dtype == "float32":
            p_input_dtype.append(dtype)
        else:
            p_input_dtype.append(dtype[9:-2])

    if len(p_input_shape) == 1:
        input_name = p_inputs[0]
        input_dtype = sofie.convert_string_to_type(p_input_dtype[0])
        # convert ot a list to modify batch size
        input_shape = list(p_input_shape[0])
        # set the batch size in case of -1 or None as first value
        if input_shape[0] is None or input_shape[0] <= 0:
            input_shape[0] = batch_size
        rmodel.add_input_tensor_info(input_name, input_dtype, input_shape)
        rmodel.add_input_tensor_name(input_name)
    else:
        # Iterating through multiple input tensors
        for input_name, input_dtype, input_shape_tuple in zip(p_inputs, p_input_dtype, p_input_shape):
            input_dtype = sofie.convert_string_to_type(input_dtype)
            input_shape = list(input_shape_tuple)
            if input_shape[0] is None or input_shape[0] <= 0:
                input_shape[0] = batch_size
            rmodel.add_input_tensor_info(input_name, input_dtype, input_shape)
            rmodel.add_input_tensor_name(input_name)

    # Adding OutputTensorInfos
    output_names = []
    if keras_version < "2.16" or is_functional_model:
        for layer_name in keras_model.output_names:
            final_layer = keras_model.get_layer(layer_name)
            output_layer_name = final_layer.output.name
            output_names.append(output_layer_name)
    else:
        output_layer_name = "tensor_output_" + keras_model.layers[-1].name
        output_names.append(output_layer_name)

    rmodel.add_output_tensor_name_list(output_names)
    return rmodel
