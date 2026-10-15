import os

import sofie


def parse(filename, verbose=False):
    """
    Parse an ONNX model (.onnx file) into a RModel object

    Parameters:
    filename (str): File location of the ONNX model.
    verbose (bool): Print what the parser does.

    Returns:
    RModel: The parsed model, from which the inference code is generated.

    Raises an exception if the model uses an operator that SOFIE does not support;
    ``check_model`` lists them.

    Example usage:
    ~~~ {.py}
    import sofie

    rmodel = sofie.onnx.parse("model.onnx")
    rmodel.generate()
    rmodel.output_generated("model.hxx")
    ~~~
    """
    filename = os.fspath(filename)
    if not os.path.exists(filename):
        raise RuntimeError("Model file {} not found!".format(filename))
    return sofie.RModelParser_ONNX().parse(filename, verbose)


def check_model(filename, verbose=False):
    """
    Check whether SOFIE supports all the operators of an ONNX model (.onnx file).

    The missing operators, with how many times the model uses them, are printed.

    Returns:
    bool: True if every operator is supported.
    """
    filename = os.fspath(filename)
    if not os.path.exists(filename):
        raise RuntimeError("Model file {} not found!".format(filename))
    return sofie.RModelParser_ONNX().check_model(filename, verbose)
