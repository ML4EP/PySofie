"""Compatibility helpers to support multiple ROOT versions."""


def add_initialized_tensor(rmodel, tensor_name, tensor_type, shape, data):
    """Call RModel::AddInitializedTensor in a ROOT-version-independent way.

    ROOT master after 6.40 (see root-project/root commit 9a7f3e776c4) has a
    non-templated overload that takes the tensor type and the data as a raw
    pointer. Released ROOT versions only have a templated overload without the
    tensor type parameter, which is deduced from the data pointer type instead.

    Parameters:
    rmodel (RModel): The RModel to add the tensor to.
    tensor_name (str): Name of the tensor.
    tensor_type (SOFIE.ETensorType): Type of the tensor (used by new ROOT).
    shape (list): Shape of the tensor.
    data (numpy.ndarray): The tensor data.
    """
    try:
        rmodel.AddInitializedTensor(tensor_name, tensor_type, shape, data)
    except TypeError:
        cpp_type = {
            "float32": "float",
            "int64": "int64_t",
            "uint64": "uint64_t",
        }[data.dtype.name]
        rmodel.AddInitializedTensor[cpp_type](tensor_name, shape, data)
