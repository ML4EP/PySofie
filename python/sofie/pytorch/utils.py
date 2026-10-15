"""Utilities for the translation of ATen operators into SOFIE operators
"""

import re

import numpy as np

import sofie

def import_torch():
    """Import torch; if it is missing, say how to install it."""
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            'sofie.pytorch needs torch >= 2.5, which is not installed: install it with pip install "sofie[torch]"'
        ) from e
    return torch


# ---------------------------------------------------------------------------------------
# names and types
# ---------------------------------------------------------------------------------------
def clean(name):
    """Make a torch.fx name or state_dict key usable as a SOFIE tensor name."""
    return re.sub(r"\W", "_", str(name))


def etype(dtype):
    """SOFIE tensor type of a torch dtype."""
    name = str(dtype).replace("torch.", "")
    names = {
        "float32": "FLOAT",
        "float64": "DOUBLE",
        "float16": "FLOAT16",
        "bfloat16": "BFLOAT16",
        "int64": "INT64",
        "int32": "INT32",
        "int16": "INT16",
        "int8": "INT8",
        "uint8": "UINT8",
        "bool": "BOOL",
    }
    if name not in names:
        raise RuntimeError(f"SOFIE - Unsupported - tensor data type {dtype}")
    return getattr(sofie.ETensorType, names[name])


def numpy_dtype(dtype):
    """numpy dtype of a torch dtype."""
    return {"float32": np.float32, "float64": np.float64, "int64": np.int64, "int32": np.int32, "bool": np.bool_}.get(
        str(dtype).replace("torch.", ""), np.float32
    )


def bind_args(node):
    """
    Return the arguments of an ATen call_function node as a dict {name: value}.

    The (positional and keyword) arguments are matched against the operator schema, so
    arguments that torch.export left at their default value are filled in.
    """
    schema = node.target._schema
    values = {}
    for i, arg in enumerate(schema.arguments):
        if i < len(node.args):
            values[arg.name] = node.args[i]
        elif arg.name in node.kwargs:
            values[arg.name] = node.kwargs[arg.name]
        elif arg.has_default_value():
            values[arg.name] = arg.default_value
    return values

# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def forward(node, src, ctx):
    """
    The value of `node` is the one of `src`, unchanged: use the same tensor, or copy it when `node`
    is an output of the model (an output tensor needs an operator that produces it).
    """
    if node.name in ctx.graph_outputs:
        return sofie.ROperator_Identity["float"](ctx.name(src), ctx.name(node))
    ctx.alias(node, ctx.name(src))
    return []


def norm_dim(dim, rank):
    """A dimension index, which may be negative, as an index in [0, rank)."""
    return int(dim) % rank if rank else 0


def norm_dims(dims, rank):
    """Dimension indices as indices in [0, rank); all the dimensions if there are none."""
    if dims is None or len(dims) == 0:
        return list(range(rank))
    return [norm_dim(d, rank) for d in dims]


def ints(value, n, default):
    """An attribute that is an int or a list of ints, as a list of n ints."""
    if value is None or (not isinstance(value, int) and len(value) == 0):
        return [default] * n
    if isinstance(value, int):
        return [value] * n
    value = [int(v) for v in value]
    return value * n if len(value) == 1 else value


def transposed_source(arg):
    """If `arg` is a 2D ``aten.t`` node, return the node it transposes, else None."""
    import torch

    if isinstance(arg, torch.fx.Node) and arg.op == "call_function" and str(arg.target) == "aten.t.default":
        return arg.args[0]
    return None


def slice_op(ctx, node, src, out_name, dim, start, end, step):
    """The Slice operator that takes ``src[start:end:step]`` along `dim`, with negative and out-of-range bounds."""
    size = ctx.shape(src)[dim]
    start = 0 if start is None else int(start)
    end = size if end is None else int(end)
    start = max(start + size, 0) if start < 0 else min(start, size)
    end = max(end + size, 0) if end < 0 else min(end, size)
    names = [
        ctx.shape_tensor(node.name + "_starts", [start]),
        ctx.shape_tensor(node.name + "_ends", [end]),
        ctx.shape_tensor(node.name + "_axes", [dim]),
        ctx.shape_tensor(node.name + "_steps", [int(step)]),
    ]
    return sofie.ROperator_Slice_int64(ctx.name(src), names, out_name)


def pool_attrs(args, n, dilation=False):
    """The attributes of a pooling operator over n dimensions, from the arguments of the ATen operator."""
    attr = sofie.RAttributes_Pool()
    kernel = ints(args["kernel_size"], n, 1)
    attr.kernel_shape = kernel
    attr.strides = ints(args.get("stride") or kernel, n, 1)
    pads = ints(args.get("padding"), n, 0)
    attr.pads = pads + pads
    attr.dilations = ints(args.get("dilation"), n, 1) if dilation else [1] * n
    attr.ceil_mode = int(bool(args.get("ceil_mode", False)))
    attr.count_include_pad = int(bool(args.get("count_include_pad", True)))
    attr.auto_pad = "NOTSET"
    return attr


def scale_bias(ctx, node, weight, bias, shape):
    """Names of the scale and bias tensors of a normalization; they are constants 1 and 0 when the module has none."""
    scale = ctx.name(weight) if weight is not None else ctx.const(node.name + "_scale", np.ones(shape), shape)
    shift = ctx.name(bias) if bias is not None else ctx.const(node.name + "_bias", np.zeros(shape), shape)
    return scale, shift


# ---------------------------------------------------------------------------------------
# handler factories, for groups of operators that differ in SOFIE class only
# ---------------------------------------------------------------------------------------
def binary(kind, swap=False):
    """Element-wise binary operator ``ROperator_BasicBinary_float_<kind>``; `swap` exchanges the operands."""

    def handler(node, ctx):
        args = bind_args(node)
        if args.get("alpha", 1) != 1:
            raise RuntimeError(f"SOFIE - Unsupported - {node.target} with alpha != 1")
        if args.get("rounding_mode") is not None:
            raise RuntimeError(f"SOFIE - Unsupported - {node.target} with a rounding mode")
        a, b = ctx.operand(args["self"]), ctx.operand(args["other"] if "other" in args else args["exponent"])
        if swap:
            a, b = b, a
        return getattr(sofie, "ROperator_BasicBinary_float_" + kind)(a, b, ctx.name(node))

    return handler


def nary(kind):
    """Element-wise operator on two inputs, ``ROperator_BasicNary_float_<kind>``."""

    def handler(node, ctx):
        args = bind_args(node)
        return getattr(sofie, "ROperator_BasicNary_float_" + kind)(
            [ctx.operand(args["self"]), ctx.operand(args["other"])], ctx.name(node)
        )

    return handler


def unary(cls_name):
    """Operator with one input and one output and no attributes: the class `cls_name` of the sofie module."""

    def handler(node, ctx):
        return getattr(sofie, cls_name)(ctx.name(node.args[0]), ctx.name(node))

    return handler


def comparison(kind):
    """Comparison operator ``ROperator_Comparison_float_<kind>``."""

    def handler(node, ctx):
        args = bind_args(node)
        return getattr(sofie, "ROperator_Comparison_float_" + kind)(
            ctx.operand(args["self"]), ctx.operand(args["other"]), ctx.name(node)
        )

    return handler


def reduce(mode):
    """Reduction over the ``dim`` and ``keepdim`` arguments: the class `mode` of the sofie module."""

    def handler(node, ctx):
        args = bind_args(node)
        src = args["self"]
        rank = len(ctx.shape(src))
        dims = args.get("dim")
        if isinstance(dims, int):
            dims = [dims]
        axes = norm_dims(dims, rank)
        keep = 1 if args.get("keepdim", False) else 0
        return getattr(sofie, mode)(keep, axes, ctx.name(src), "", ctx.name(node))

    return handler


def creation(value_of):
    """Tensor creation (full, zeros, ...): the shapes are static, so it is a constant known at parse time."""

    def handler(node, ctx):
        ctx.add_node_constant(node, np.full(ctx.shape(node), value_of(node), dtype=numpy_dtype(ctx.dtype(node))))
        return []

    return handler
