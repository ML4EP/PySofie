"""Helpers shared by the parsers."""


def add_initialized_tensor(rmodel, tensor_name, tensor_type, shape, data):
    """Add a weight/constant tensor to an RModel.

    Parameters:
    rmodel (RModel): The RModel to add the tensor to.
    tensor_name (str): Name of the tensor.
    tensor_type (sofie.ETensorType): Type of the tensor.
    shape (list): Shape of the tensor.
    data (array-like): The tensor data, copied into the model.
    """
    rmodel.add_initialized_tensor(tensor_name, tensor_type, [int(d) for d in shape], data)
