"""Translation of ATen operators into SOFIE operators: one declaration per operator.

A declaration is a function ``handler(node, ctx)`` registered with ``@aten("aten.<name>.<overload>")``,
with the operator name as printed in the exported graph. It takes the FX ``call_function`` node and
the :class:`~sofie.pytorch.parser.parse_context` of the model being built, and returns SOFIE
ROperator (or the list of them, in execution order) that the node translates to. Operators without a
declaration are lowered to core ATen by ``ExportedProgram.run_decompositions()``.
"""

import math

import numpy as np

import sofie
from .utils import (
    binary,
    bind_args,
    comparison,
    creation,
    etype,
    forward,
    ints,
    nary,
    norm_dim,
    norm_dims,
    numpy_dtype,
    pool_attrs,
    reduce,
    scale_bias,
    slice_op,
    transposed_source,
    unary,
)


class handler:
    """The translation of an ATen operator, and the BLAS routines that the generated code needs for it."""

    def __init__(self, fn, blas_routines):
        self.fn = fn
        self.blas_routines = list(blas_routines)


# Maps the name of an ATen operator (as printed in the graph) to its Handler
aten_handlers = {}

# Operators that only have a compile-time effect: no SOFIE operator is needed
ignored_targets = {
    "aten._assert_tensor_metadata.default",
    "aten._assert_scalar.default",
    "aten.sym_constrain_range_for_size.default",
}


def aten(*targets, blas=()):
    """Register a function ``handler(node, ctx)`` as the translation of the given ATen operators."""

    def register(fn):
        for target in targets:
            aten_handlers[target] = handler(fn, blas)
        return fn

    return register


# SELU constants as defined by torch.nn.SELU
selu_alpha = 1.6732632423543772848170429916717
selu_scale = 1.0507009873554804934193349852946

float_max = float(np.finfo(np.float32).max)


# ---------------------------------------------------------------------------------------
# element-wise binary operators
# ---------------------------------------------------------------------------------------
for _kind, _names in {
    "Add": ["add.Tensor", "add.Scalar"],
    "Sub": ["sub.Tensor", "sub.Scalar"],
    "Mul": ["mul.Tensor", "mul.Scalar"],
    "Div": ["div.Tensor", "div.Scalar"],
    "Pow": ["pow.Tensor_Tensor", "pow.Tensor_Scalar"],
}.items():
    aten(*["aten." + n for n in _names])(binary(_kind))
aten("aten.rsub.Scalar", "aten.rsub.Tensor")(binary("Sub", swap=True))


aten("aten.maximum.default")(nary("Max"))
aten("aten.minimum.default")(nary("Min"))


# ---------------------------------------------------------------------------------------
# element-wise unary operators and activations
# ---------------------------------------------------------------------------------------
for _target, _cls in {
    "neg": "ROperator_BasicUnary_float_Neg",
    "exp": "ROperator_BasicUnary_float_Exp",
    "log": "ROperator_BasicUnary_float_Log",
    "sqrt": "ROperator_BasicUnary_float_Sqrt",
    "abs": "ROperator_BasicUnary_float_Abs",
    "sin": "ROperator_BasicUnary_float_Sin",
    "cos": "ROperator_BasicUnary_float_Cos",
    "floor": "ROperator_BasicUnary_float_Floor",
    "reciprocal": "ROperator_BasicUnary_float_Reciprocal",
    "erf": "ROperator_Erf_float",
    "relu": "ROperator_Relu_float",
    "sigmoid": "ROperator_Sigmoid_float",
    "tanh": "ROperator_Tanh_float",
    "hardswish": "ROperator_HardSwish_float",
    "logical_not": "ROperator_Not",
}.items():
    aten(f"aten.{_target}.default")(unary(_cls))


@aten("aten.rsqrt.default")
def _rsqrt(node, ctx):
    root = ctx.new_name(node.name + "_sqrt")
    return [
        sofie.ROperator_BasicUnary_float_Sqrt(ctx.name(node.args[0]), root),
        sofie.ROperator_BasicUnary_float_Reciprocal(root, ctx.name(node)),
    ]


@aten("aten.expm1.default")
def _expm1(node, ctx):
    e = ctx.new_name(node.name + "_exp")
    return [
        sofie.ROperator_BasicUnary_float_Exp(ctx.name(node.args[0]), e),
        sofie.ROperator_BasicBinary_float_Sub(e, ctx.operand(1.0), ctx.name(node)),
    ]


@aten("aten.log1p.default")
def _log1p(node, ctx):
    s = ctx.new_name(node.name + "_sum")
    return [
        sofie.ROperator_BasicBinary_float_Add(ctx.name(node.args[0]), ctx.operand(1.0), s),
        sofie.ROperator_BasicUnary_float_Log(s, ctx.name(node)),
    ]


@aten("aten.leaky_relu.default")
def _leaky_relu(node, ctx):
    args = bind_args(node)
    return sofie.ROperator_LeakyRelu["float"](float(args["negative_slope"]), ctx.name(args["self"]), ctx.name(node))


@aten("aten.elu.default")
def _elu(node, ctx):
    args = bind_args(node)
    alpha, scale, input_scale = float(args["alpha"]), float(args["scale"]), float(args["input_scale"])
    x, y = ctx.name(args["self"]), ctx.name(node)
    if input_scale != 1.0:
        raise RuntimeError(f"SOFIE - Unsupported - elu with input_scale != 1 (node {node.name})")
    if scale == 1.0:
        return sofie.ROperator_Elu["float"](alpha, x, y)
    if math.isclose(alpha, selu_alpha, rel_tol=1e-6) and math.isclose(scale, selu_scale, rel_tol=1e-6):
        return sofie.ROperator_Selu["float"](x, y)
    e = ctx.new_name(node.name + "_elu")
    return [
        sofie.ROperator_Elu["float"](alpha, x, e),
        sofie.ROperator_BasicBinary_float_Mul(e, ctx.operand(scale), y),
    ]


@aten("aten.selu.default")
def _selu(node, ctx):
    return sofie.ROperator_Selu["float"](ctx.name(node.args[0]), ctx.name(node))


@aten("aten.gelu.default")
def _gelu(node, ctx):
    args = bind_args(node)
    return sofie.ROperator_Gelu(ctx.name(args["self"]), ctx.name(node), args.get("approximate", "none"))


@aten("aten.hardsigmoid.default")
def _hardsigmoid(node, ctx):
    return sofie.ROperator_HardSigmoid["float"](1.0 / 6.0, 0.5, ctx.name(node.args[0]), ctx.name(node))


@aten("aten.hardtanh.default", "aten.clamp.default")
def _clip(node, ctx):
    args = bind_args(node)
    lo = args.get("min_val", args.get("min"))
    hi = args.get("max_val", args.get("max"))
    lo = -float_max if lo is None else float(lo)
    hi = float_max if hi is None else float(hi)
    return sofie.ROperator_Clip["float"](ctx.name(args["self"]), ctx.name(node), lo, hi)


@aten("aten._softmax.default", "aten.softmax.int", "aten._log_softmax.default", "aten.log_softmax.int")
def _softmax(node, ctx):
    args = bind_args(node)
    rank = len(ctx.shape(args["self"]))
    log = "log" in str(node.target)
    return sofie.ROperator_Softmax(norm_dim(args["dim"], rank), ctx.name(args["self"]), ctx.name(node), log)

for _kind, _name in {"Greater": "gt", "GreaterEq": "ge", "Less": "lt", "LessEq": "le", "Eq": "eq"}.items():
    aten(f"aten.{_name}.Tensor", f"aten.{_name}.Scalar")(comparison(_kind))


@aten("aten.ne.Tensor", "aten.ne.Scalar")
def _ne(node, ctx):
    args = bind_args(node)
    eq = ctx.new_name(node.name + "_eq")
    return [
        sofie.ROperator_Comparison_float_Eq(ctx.operand(args["self"]), ctx.operand(args["other"]), eq),
        sofie.ROperator_Not(eq, ctx.name(node)),
    ]


@aten("aten.where.self", "aten.where.ScalarOther", "aten.where.ScalarSelf", "aten.where.Scalar")
def _where(node, ctx):
    args = bind_args(node)
    return sofie.ROperator_Where["float"](
        ctx.name(args["condition"]), ctx.operand(args["self"]), ctx.operand(args["other"]), ctx.name(node)
    )


@aten("aten._to_copy.default", "aten.to.dtype", "aten.to.dtype_layout")
def _cast(node, ctx):
    args = bind_args(node)
    src = args["self"]
    target = etype(ctx.dtype(node))
    if etype(ctx.dtype(src)) == target:
        return forward(node, src, ctx)
    return sofie.ROperator_Cast(target, ctx.name(src), ctx.name(node))


@aten(
    "aten.clone.default",
    "aten.alias.default",
    "aten.detach.default",
    "aten.contiguous.default",
    "aten.lift_fresh_copy.default",
    "aten.dropout.default",
)
def _identity(node, ctx):
    return forward(node, node.args[0], ctx)


aten("aten.full.default", "aten.full_like.default")(creation(lambda node: bind_args(node)["fill_value"]))
aten("aten.zeros.default", "aten.zeros_like.default", "aten.new_zeros.default")(creation(lambda node: 0))
aten("aten.ones.default", "aten.ones_like.default", "aten.new_ones.default")(creation(lambda node: 1))


@aten("aten.arange.default", "aten.arange.start", "aten.arange.start_step")
def _arange(node, ctx):
    args = bind_args(node)
    values = np.arange(args.get("start", 0), args["end"], args.get("step", 1), dtype=numpy_dtype(ctx.dtype(node)))
    ctx.add_node_constant(node, values)
    return []


# ---------------------------------------------------------------------------------------
# shape manipulation
# ---------------------------------------------------------------------------------------
@aten(
    "aten.view.default",
    "aten.reshape.default",
    "aten._unsafe_view.default",
    "aten.flatten.using_ints",
    "aten.unflatten.int",
    "aten.squeeze.dim",
    "aten.squeeze.dims",
    "aten.squeeze.default",
    "aten.unsqueeze.default",
    "aten.view_as.default",
)
def _reshape(node, ctx):
    # a change of shape that keeps the order of the elements is a Reshape to the static output shape
    src = node.args[0]
    shape = ctx.shape(node)
    if ctx.shape(src) == shape:
        return forward(node, src, ctx)
    shape_name = ctx.shape_tensor(node.name + "_shape", shape)
    return sofie.ROperator_Reshape(sofie.ReshapeOpMode.Reshape, 0, ctx.name(src), shape_name, ctx.name(node))


@aten("aten.permute.default", "aten.transpose.int", "aten.t.default")
def _transpose(node, ctx):
    args = bind_args(node)
    rank = len(ctx.shape(node.args[0]))
    if "dims" in args:
        perm = [norm_dim(d, rank) for d in args["dims"]]
    elif "dim0" in args:
        perm = list(range(rank))
        d0, d1 = norm_dim(args["dim0"], rank), norm_dim(args["dim1"], rank)
        perm[d0], perm[d1] = perm[d1], perm[d0]
    else:  # aten.t
        perm = [1, 0] if rank == 2 else list(range(rank))
    if perm == list(range(rank)):
        return forward(node, node.args[0], ctx)
    return sofie.ROperator_Transpose(perm, ctx.name(node.args[0]), ctx.name(node))


@aten("aten.expand.default", "aten.expand_as.default")
def _expand(node, ctx):
    src = node.args[0]
    shape = ctx.shape(node)
    if ctx.shape(src) == shape:
        return forward(node, src, ctx)
    return sofie.ROperator_Expand["float"](ctx.name(src), ctx.shape_tensor(node.name + "_shape", shape), ctx.name(node))

@aten("aten.slice.Tensor")
def _slice(node, ctx):
    args = bind_args(node)
    src = args["self"]
    dim = norm_dim(args["dim"], len(ctx.shape(src)))
    return slice_op(ctx, node, src, ctx.name(node), dim, args.get("start"), args.get("end"), args.get("step", 1))


@aten("aten.select.int")
def _select(node, ctx):
    args = bind_args(node)
    src = args["self"]
    dim = norm_dim(args["dim"], len(ctx.shape(src)))
    index = int(args["index"])
    if index < 0:
        index += ctx.shape(src)[dim]
    sliced = ctx.new_name(node.name + "_slice")
    shape_name = ctx.shape_tensor(node.name + "_shape", ctx.shape(node))
    return [
        slice_op(ctx, node, src, sliced, dim, index, index + 1, 1),
        sofie.ROperator_Reshape(sofie.ReshapeOpMode.Reshape, 0, sliced, shape_name, ctx.name(node)),
    ]


@aten("aten.cat.default")
def _cat(node, ctx):
    args = bind_args(node)
    tensors = [t for t in args["tensors"] if 0 not in ctx.shape(t)]
    rank = len(ctx.shape(node))
    return sofie.ROperator_Concat([ctx.name(t) for t in tensors], norm_dim(args["dim"], rank), 0, ctx.name(node))


@aten("aten.split_with_sizes.default", "aten.split.Tensor", "aten.split.sizes")
def _split(node, ctx):
    args = bind_args(node)
    src = args["self"]
    dim = norm_dim(args.get("dim", 0), len(ctx.shape(src)))
    outs = node.meta["val"]
    sizes = [int(o.shape[dim]) for o in outs]
    names = [ctx.new_name(f"{node.name}_{i}") for i in range(len(sizes))]
    ctx.set_outputs(node, names)
    return sofie.ROperator_Split(ctx.name(src), ctx.shape_tensor(node.name + "_sizes", sizes), dim, names)


@aten("aten.repeat.default")
def _repeat(node, ctx):
    args = bind_args(node)
    src = args["self"]
    repeats = [int(r) for r in args["repeats"]]
    in_shape = ctx.shape(src)
    ops, x = [], ctx.name(src)
    if len(repeats) > len(in_shape):  # the missing leading dimensions of the input are 1
        x = ctx.new_name(node.name + "_x")
        padded = [1] * (len(repeats) - len(in_shape)) + in_shape
        ops.append(
            sofie.ROperator_Reshape(
                sofie.ReshapeOpMode.Reshape, 0, ctx.name(src), ctx.shape_tensor(node.name + "_in_shape", padded), x
            )
        )
    ops.append(sofie.ROperator_Tile["float"](ctx.shape_tensor(node.name + "_repeats", repeats), x, ctx.name(node)))
    return ops


@aten("aten.constant_pad_nd.default")
def _pad(node, ctx):
    args = bind_args(node)
    rank = len(ctx.shape(args["self"]))
    pad = [int(p) for p in args["pad"]]
    begin, end = [0] * rank, [0] * rank
    for i in range(len(pad) // 2):  # torch lists the pads from the last dimension backward
        begin[rank - 1 - i], end[rank - 1 - i] = pad[2 * i], pad[2 * i + 1]
    value = ctx.const(node.name + "_value", [float(args.get("value") or 0.0)], [1])
    return sofie.ROperator_Pad["float"](
        ctx.name(args["self"]), ctx.shape_tensor(node.name + "_pads", begin + end), value, "", ctx.name(node), "constant"
    )


@aten("aten.cumsum.default")
def _cumsum(node, ctx):
    args = bind_args(node)
    rank = len(ctx.shape(args["self"]))
    return sofie.ROperator_CumSum["float"](norm_dim(args["dim"], rank), 0, 0, ctx.name(args["self"]), ctx.name(node))


@aten("aten.embedding.default")
def _embedding(node, ctx):
    args = bind_args(node)
    return sofie.ROperator_Gather(0, ctx.name(args["weight"]), ctx.name(args["indices"]), ctx.name(node))


@aten("<built-in function getitem>")
def _getitem(node, ctx):
    source, index = node.args
    names = ctx.outputs_of(source)
    if names is None or names[index] is None:
        raise RuntimeError(
            f"SOFIE - Unsupported - output {index} of {source.target} (node {node.name}) is not available"
        )
    ctx.alias(node, names[index])
    return []

aten("aten.mean.dim", "aten.mean.default")(reduce("ROperator_ReduceMean"))
aten("aten.sum.dim_IntList", "aten.sum.default")(reduce("ROperator_ReduceSum"))
aten("aten.amax.default")(reduce("ROperator_ReduceMax"))
aten("aten.amin.default")(reduce("ROperator_ReduceMin"))
aten("aten.prod.dim_int")(reduce("ROperator_ReduceProd"))


@aten("aten.var.correction", "aten.var.dim")
def _var(node, ctx):
    # var = sum((x - mean(x))^2) / (N - correction)
    args = bind_args(node)
    src = args["self"]
    shape = ctx.shape(src)
    dims = args.get("dim")
    if isinstance(dims, int):
        dims = [dims]
    axes = norm_dims(dims, len(shape))
    n = int(np.prod([shape[a] for a in axes]))
    correction = args.get("correction")
    correction = 1 if correction is None else float(correction)
    if "unbiased" in args:
        correction = 1 if args["unbiased"] else 0
    keep = 1 if args.get("keepdim", False) else 0
    x = ctx.name(src)
    mean, diff, sq, total = (ctx.new_name(node.name + s) for s in ("_mean", "_diff", "_sq", "_sum"))
    return [
        sofie.ROperator_ReduceMean(1, axes, x, "", mean),
        sofie.ROperator_BasicBinary_float_Sub(x, mean, diff),
        sofie.ROperator_BasicBinary_float_Mul(diff, diff, sq),
        sofie.ROperator_ReduceSum(keep, axes, sq, "", total),
        sofie.ROperator_BasicBinary_float_Div(total, ctx.operand(n - correction), ctx.name(node)),
    ]

@aten("aten.linear.default", blas=["Gemm", "Gemv"])
def _linear(node, ctx):
    args = bind_args(node)
    x, w, bias = args["input"], args["weight"], args.get("bias")
    in_shape = ctx.shape(x)
    out_shape = ctx.shape(node)
    if len(in_shape) == 2:
        x_name, y_name, ops = ctx.name(x), ctx.name(node), []
    else:
        # collapse the leading dimensions into the batch dimension
        x_name, y_name = ctx.new_name(node.name + "_x2d"), ctx.new_name(node.name + "_y2d")
        flat = [int(np.prod(in_shape[:-1])), in_shape[-1]]
        ops = [
            sofie.ROperator_Reshape(
                sofie.ReshapeOpMode.Reshape, 0, ctx.name(x), ctx.shape_tensor(node.name + "_in_shape", flat), x_name
            )
        ]
    if bias is None:
        ops.append(sofie.ROperator_Gemm["float"](1.0, 1.0, 0, 1, x_name, ctx.name(w), y_name))
    else:
        ops.append(sofie.ROperator_Gemm["float"](1.0, 1.0, 0, 1, x_name, ctx.name(w), ctx.name(bias), y_name))
    if len(in_shape) != 2:
        ops.append(
            sofie.ROperator_Reshape(
                sofie.ReshapeOpMode.Reshape, 0, y_name, ctx.shape_tensor(node.name + "_out_shape", out_shape), ctx.name(node)
            )
        )
    return ops


@aten("aten.addmm.default", blas=["Gemm", "Gemv"])
def _addmm(node, ctx):
    # addmm(bias, x, t(weight)): a transpose of the second matrix is folded into transB
    args = bind_args(node)
    source = transposed_source(args["mat2"])
    return sofie.ROperator_Gemm["float"](
        float(args["alpha"]),
        float(args["beta"]),
        0,
        1 if source is not None else 0,
        ctx.name(args["mat1"]),
        ctx.name(source if source is not None else args["mat2"]),
        ctx.name(args["self"]),
        ctx.name(node),
    )


@aten("aten.mm.default", "aten.bmm.default", "aten.matmul.default", blas=["Gemm", "Gemv"])
def _matmul(node, ctx):
    args = bind_args(node)
    b = args.get("mat2", args.get("other"))
    source = transposed_source(b)
    return sofie.ROperator_Gemm["float"](
        1.0,
        0.0,
        0,
        1 if source is not None else 0,
        ctx.name(args["self"]),
        ctx.name(source if source is not None else b),
        ctx.name(node),
    )

@aten(
    "aten.convolution.default",
    "aten.conv1d.default",
    "aten.conv2d.default",
    "aten.conv3d.default",
    blas=["Gemm", "Axpy"],
)
def _convolution(node, ctx):
    args = bind_args(node)
    weight = args["weight"]
    kernel = [int(d) for d in weight.meta["val"].shape[2:]]
    n = len(kernel)
    if args.get("transposed", False):
        pads = ints(args["padding"], n, 0)
        bias = args.get("bias")
        return sofie.ROperator_ConvTranspose["float"](
            "NOTSET",
            ints(args.get("dilation"), n, 1),
            int(args["groups"]),
            kernel,
            ints(args.get("output_padding"), n, 0),
            [],
            pads + pads,
            ints(args.get("stride"), n, 1),
            ctx.name(args["input"]),
            ctx.name(weight),
            ctx.name(bias) if bias is not None else "",
            ctx.name(node),
        )
    padding = args["padding"]
    if isinstance(padding, str):
        if padding != "valid":
            raise RuntimeError(f"SOFIE - Unsupported - convolution padding '{padding}' (node {node.name})")
        padding = 0
    pads = ints(padding, n, 0)
    common = (
        "NOTSET",
        ints(args.get("dilation"), n, 1),
        int(args["groups"]),
        kernel,
        pads + pads,
        ints(args.get("stride"), n, 1),
        ctx.name(args["input"]),
        ctx.name(weight),
    )
    bias = args.get("bias")
    if bias is None:
        return sofie.ROperator_Conv["float"](*common, ctx.name(node))
    return sofie.ROperator_Conv["float"](*common, ctx.name(bias), ctx.name(node))


@aten("aten.avg_pool2d.default")
def _avg_pool(node, ctx):
    args = bind_args(node)
    if args.get("divisor_override") is not None:
        raise RuntimeError("SOFIE - Unsupported - avg_pool2d with divisor_override")
    return sofie.ROperator_Pool["float"](
        sofie.PoolOpMode.AveragePool, pool_attrs(args, 2), ctx.name(args["self"]), ctx.name(node)
    )


@aten("aten.max_pool2d.default", "aten.max_pool2d_with_indices.default")
def _max_pool(node, ctx):
    args = bind_args(node)
    if "with_indices" in str(node.target):
        ctx.set_outputs(node, [ctx.name(node), None])
    return sofie.ROperator_Pool["float"](
        sofie.PoolOpMode.MaxPool, pool_attrs(args, 2, dilation=True), ctx.name(args["self"]), ctx.name(node)
    )


@aten("aten.adaptive_avg_pool2d.default")
def _adaptive_avg_pool(node, ctx):
    args = bind_args(node)
    in_shape, out_shape = ctx.shape(args["self"]), ctx.shape(node)
    if out_shape[-2:] == [1, 1]:
        attr = sofie.RAttributes_Pool()
        attr.kernel_shape, attr.strides, attr.pads, attr.dilations = in_shape[-2:], [1, 1], [0] * 4, [1, 1]
        return sofie.ROperator_Pool["float"](
            sofie.PoolOpMode.GlobalAveragePool, attr, ctx.name(args["self"]), ctx.name(node)
        )
    if any(i % o for i, o in zip(in_shape[-2:], out_shape[-2:])):
        raise RuntimeError(f"SOFIE - Unsupported - adaptive_avg_pool2d {in_shape} -> {out_shape}")
    kernel = [i // o for i, o in zip(in_shape[-2:], out_shape[-2:])]
    pool_args = {"kernel_size": kernel, "stride": kernel, "padding": 0, "ceil_mode": False}
    return sofie.ROperator_Pool["float"](
        sofie.PoolOpMode.AveragePool, pool_attrs(pool_args, 2), ctx.name(args["self"]), ctx.name(node)
    )


@aten("aten._native_batch_norm_legit_no_training.default", "aten.batch_norm.default")
def _batch_norm(node, ctx):
    args = bind_args(node)
    if args.get("training", False):
        raise RuntimeError("SOFIE - Unsupported - batch normalization in training mode")
    channels = [ctx.shape(args["input"])[1]]
    scale, shift = scale_bias(ctx, node, args.get("weight"), args.get("bias"), channels)
    if "legit" in str(node.target):
        ctx.set_outputs(node, [ctx.name(node), None, None])
    return sofie.ROperator_BatchNormalization["float"](
        float(args["eps"]),
        float(args["momentum"]),
        0,
        ctx.name(args["input"]),
        scale,
        shift,
        ctx.name(args["running_mean"]),
        ctx.name(args["running_var"]),
        ctx.name(node),
    )


@aten("aten.native_layer_norm.default", "aten.layer_norm.default")
def _layer_norm(node, ctx):
    args = bind_args(node)
    normalized = [int(d) for d in args["normalized_shape"]]
    rank = len(ctx.shape(args["input"]))
    scale, shift = scale_bias(ctx, node, args.get("weight"), args.get("bias"), normalized)
    if "native" in str(node.target):
        ctx.set_outputs(node, [ctx.name(node), None, None])
    return sofie.ROperator_LayerNormalization["float"](
        rank - len(normalized), float(args["eps"]), 1, ctx.name(args["input"]), scale, shift, ctx.name(node), "", ""
    )


@aten("aten.native_group_norm.default", "aten.group_norm.default")
def _group_norm(node, ctx):
    args = bind_args(node)
    channels = [ctx.shape(args["input"])[1]]
    scale, shift = scale_bias(ctx, node, args.get("weight"), args.get("bias"), channels)
    if "native" in str(node.target):
        ctx.set_outputs(node, [ctx.name(node), None, None])
    groups = int(args["group"] if "group" in args else args["num_groups"])
    return sofie.ROperator_GroupNorm["float"](groups, float(args["eps"]), ctx.name(args["input"]), scale, shift, ctx.name(node))


@aten("aten.instance_norm.default")
def _instance_norm(node, ctx):
    args = bind_args(node)
    if not args.get("use_input_stats", True) or args.get("running_mean") is not None:
        raise RuntimeError("SOFIE - Unsupported - instance normalization with running statistics")
    channels = [ctx.shape(args["input"])[1]]
    scale, shift = scale_bias(ctx, node, args.get("weight"), args.get("bias"), channels)
    return sofie.ROperator_InstanceNormalization["float"](
        float(args["eps"]), ctx.name(args["input"]), scale, shift, ctx.name(node)
    )


@aten("aten.scaled_dot_product_attention.default")
def _sdpa(node, ctx):
    args = bind_args(node)
    if args.get("is_causal", False):
        raise RuntimeError("SOFIE - Unsupported - scaled_dot_product_attention with is_causal")
    if args.get("dropout_p", 0.0):
        raise RuntimeError("SOFIE - Unsupported - scaled_dot_product_attention with dropout")
    mask = args.get("attn_mask")
    return sofie.ROperator_SDPA["float"](
        ctx.name(args["query"]),
        ctx.name(args["key"]),
        ctx.name(args["value"]),
        ctx.name(node),
        ctx.name(mask) if mask is not None else "",
        float(args["scale"]) if args.get("scale") is not None else 0.0,
        1 if len(ctx.shape(args["query"])) == 3 else 0,  # PyTorch's rank-3 inputs have a single head
    )
