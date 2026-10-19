import os
import time
import zipfile

import numpy as np

import sofie
from .._compat import add_initialized_tensor
from .aten import aten_handlers, ignored_targets, handler
from .utils import bind_args, clean, etype, import_torch

# Operators of the graph that need no translation
_transpose_consumers = ("aten.addmm.default", "aten.mm.default", "aten.bmm.default", "aten.matmul.default")


class parse_context:
    """State of the translation of one exported program into an RModel."""

    def __init__(self, rmodel):
        self.rmodel = rmodel
        self._aliases = {}  # node name -> tensor name, for the outputs of getitem nodes
        self._multi = {}  # node name -> tensor names of its outputs, for multi-output nodes
        self._used = set()
        self._scalars = {}
        self.consts = {}  # node name -> torch tensor, for tensors that are known at parse time
        self.graph_outputs = set()  # names of the nodes that produce an output of the model

    # -- names ---------------------------------------------------------------------------
    def name(self, arg):
        """Name of SOFIE tensor holding the value of the FX node `arg`."""
        import torch

        if not isinstance(arg, torch.fx.Node):
            raise RuntimeError(f"SOFIE - Unsupported - expected a tensor argument, got {arg!r}")
        return self._aliases.get(arg.name, clean(arg.name))

    def new_name(self, hint):
        """A fresh tensor name, for intermediate tensors of operators made of several SOFIE operators."""
        i = 0
        while f"{clean(hint)}_{i}" in self._used:
            i += 1
        name = f"{clean(hint)}_{i}"
        self._used.add(name)
        return name

    def set_outputs(self, node, names):
        """Declare the tensors that the outputs of the multi-output `node` are stored in."""
        self._multi[node.name] = names

    def outputs_of(self, node):
        return self._multi.get(node.name)

    def alias(self, node, name):
        self._aliases[node.name] = name

    # -- tensors -------------------------------------------------------------------------
    @staticmethod
    def shape(node):
        """Static shape of the output of `node`."""
        val = node.meta["val"]
        shape = []
        for dim in val.shape:
            if not isinstance(dim, int):
                raise RuntimeError(f"SOFIE - Unsupported - dynamic dimension {dim} in node {node.name}")
            shape.append(int(dim))
        return shape

    @staticmethod
    def dtype(node):
        return node.meta["val"].dtype

    def const(self, hint, values, shape, dtype="float32"):
        """Add a constant tensor and return its name."""
        name = self.new_name(hint)
        self.rmodel.add_constant_tensor(name, etype(dtype), [int(d) for d in shape], np.asarray(values).reshape(-1))
        return name

    def shape_tensor(self, hint, values):
        """Add an INT64 tensor that is used to define a shape (not written to the weight file)."""
        name = self.new_name(hint)
        values = [int(v) for v in values]
        self.rmodel.add_initialized_tensor(name, sofie.ETensorType.INT64, [len(values)], values)
        self.rmodel.set_not_writable_initialized_tensor(name)
        return name

    def add_node_constant(self, node, values):
        """Register the output of `node` as a constant tensor known at parse time."""
        import torch

        values = torch.as_tensor(values).reshape(self.shape(node)).to(node.meta["val"].dtype)
        self.consts[node.name] = values
        self._used.add(self.name(node))
        shape = self.shape(node)
        self.rmodel.add_constant_tensor(self.name(node), etype(values.dtype), shape, values.numpy().reshape(-1))

    def try_fold(self, node):
        """
        Evaluate `node` at parse time if all its tensor inputs are such constants (e.g. a mask
        built from arange), and register the result as a constant. Returns whether it did so.

        This also covers integer and boolean computations, which have no SOFIE operators.
        """
        import torch

        inputs = node.all_input_nodes
        if not inputs or any(i.name not in self.consts for i in inputs):
            return False
        if not isinstance(node.meta.get("val"), torch.Tensor):
            return False
        args = torch.fx.node.map_arg(node.args, lambda n: self.consts[n.name])
        kwargs = torch.fx.node.map_arg(node.kwargs, lambda n: self.consts[n.name])
        self.add_node_constant(node, node.target(*args, **kwargs))
        return True

    def operand(self, arg):
        """Name of the tensor for an operand that is either a node or a Python scalar."""
        import torch

        if isinstance(arg, torch.fx.Node):
            return self.name(arg)
        if isinstance(arg, (bool, int, float)):
            key = float(arg)
            if key not in self._scalars:
                self._scalars[key] = self.const("scalar", [key], [1])
            return self._scalars[key]
        raise RuntimeError(f"SOFIE - Unsupported - operand {arg!r}")


def register_operator(target, make_op, blas_routines=()):
    """
    Register the translation of an ATen operator, or override the one the parser has.

    Parameters:
    target (str): The operator as printed in the exported graph
    make_op (callable): Function ``make_op(node, ctx)`` taking the torch.fx.Node and the
                        :class:`parse_context`, and returning the
                        SOFIE ROperator (or a list of them, in execution order). The inputs of the
                        node are ``ctx.name(arg)``, its output is ``ctx.name(node)`` and its output
                        shape ``ctx.shape(node)``.
    blas_routines: BLAS routines the generated code needs for the operator.
    """
    aten_handlers[target] = handler(make_op, blas_routines)


def _load_exported_program(model, input_shapes):
    """Turn the user-provided model into a torch.export.ExportedProgram."""
    torch = import_torch()

    if isinstance(model, torch.export.ExportedProgram):
        return model

    if isinstance(model, torch.nn.Module):
        if input_shapes is None:
            raise RuntimeError("input_shapes is required to export a torch.nn.Module")
        example_inputs = tuple(torch.rand(*[int(dim) for dim in shape]) for shape in input_shapes)
        return torch.export.export(model.cpu().eval(), example_inputs)

    filename = os.fspath(model)
    if not os.path.exists(filename):
        raise RuntimeError("Model file {} not found!".format(filename))

    if zipfile.is_zipfile(filename):
        with zipfile.ZipFile(filename) as archive:
            names = archive.namelist()
        if not any(n.endswith("models/model.json") for n in names):
            if any(n.endswith(".so") or "/data/aotinductor/" in n for n in names):
                raise RuntimeError(
                    f"{filename} is a plain AOTInductor package (torch._inductor.aoti_compile_and_package): it only "
                    "holds fused, vectorized kernels and a call sequence, from which the operator graph cannot be "
                    "recovered. Package the exported program together with the compiled artifacts instead, with "
                    "torch.export.pt2_archive._package.package_pt2(path, exported_programs=ep, aoti_files=files); "
                )
            raise RuntimeError(
                f"{filename} is a TorchScript archive, which is not supported anymore. Export the model with "
                "torch.export.save(torch.export.export(model, example_inputs), 'model.pt2') instead."
            )
    return torch.export.load(filename)


def _overload(target):
    """The torch OpOverload for an operator name such as "aten.linear.default", or None."""
    import torch

    op = torch.ops
    try:
        for part in target.split("."):
            op = getattr(op, part)
    except (AttributeError, RuntimeError):
        return None
    return op if isinstance(op, torch._ops.OpOverload) else None


def _lower(ep, decompositions=None):
    """
    Lower the operators of the exported program that have no SOFIE translation to core ATen.

    This is the decomposition that AOTInductor applies, except for the operators that the parser
    translates directly, which are kept as they are. `decompositions`
    maps further operators to the functions that decompose them.
    """
    import torch

    table = torch.export.default_decompositions()
    for target in aten_handlers:
        op = _overload(target)
        if op is not None and op in table:
            del table[op]
    for op, decomposition in (decompositions or {}).items():
        table[op] = decomposition
    return ep.run_decompositions(table)


def _calls(ep):
    return [node for node in ep.graph_module.graph.nodes if node.op == "call_function"]


def _supported(target):
    return target in aten_handlers or target in ignored_targets


def list_operators(model, input_shapes=None, decompose=True, decompositions=None):
    """
    List the ATen operators of a model and whether the parser can translate them.

    The operators are the ones the parser would see in :func:`parse`, i.e. after the
    lowering of ``decompose`` is applied. Accepts the same models as ``parse``.

    Returns:
    list of dict: One entry per operator, in order of first appearance, with the keys
                  ``"op"``.
    """
    ep = _load_exported_program(model, input_shapes)
    if decompose:
        ep = _lower(ep, decompositions)
    ops = {}
    for node in _calls(ep):
        target = str(node.target)
        ops.setdefault(target, {"op": target, "count": 0, "supported": _supported(target)})["count"] += 1
    return list(ops.values())


def _folded(graph):
    """Names of the 2D transposes of weights that a Gemm absorbs into its transB (see aten._addmm)."""
    folded = set()
    for node in graph.nodes:
        if node.op != "call_function" or str(node.target) != "aten.t.default" or not node.users:
            continue
        if all(
            user.op == "call_function"
            and str(user.target) in _transpose_consumers
            and bind_args(user).get("mat2", bind_args(user).get("other")) is node
            for user in node.users
        ):
            folded.add(node.name)
    return folded


def parse(model, input_shapes=None, input_dtypes=None, decompose=True, decompositions=None):
    """
    Parse a PyTorch model exported with ``torch.export`` into a RModel object.

    The model is given either as a ``.pt2`` file written by ``torch.export.save``, as a
    ``torch.export.ExportedProgram``, or as a ``torch.nn.Module``. An exported program is a graph of ATen
    operators with its weights attached, and with the input shapes fixed.

    The nodes of the graph are traversed to add the equivalent ROperator instances to the
    RModel (an ATen operator may translate to several SOFIE operators), the weights
    (parameters and buffers) become its initialized tensors, and the graph inputs and
    outputs become its input and output tensors.

    By default, all the operators that have no direct SOFIE translation are lowered to core
    ATen first, the way AOTInductor does (``ExportedProgram.run_decompositions()``), so that
    everything that decomposes into supported ATen operators can be parsed. Operators that
    remain unsupported are all reported in a single error; ``list_operators`` shows them in
    advance. Everything in a model has to break down into ATen operators: a custom operator
    (registered with ``torch.library``) stays an opaque node, so give its decomposition in
    `decompositions`.

    Parameters:
    model (str, ExportedProgram or nn.Module): The model to parse.
    input_shapes: Optional list of input shape lists. By default the shapes the model
                  was exported with are used. When given, they must match those.
    input_dtypes: Optional list of sofie.ETensorType for the data types of the
                  input tensors. By default the data types of the exported inputs are used.
    decompose (bool): Lower the operators without a SOFIE translation to core ATen.
    decompositions (dict): Additional decompositions, mapping an operator (an OpOverload, e.g.
                           ``torch.ops.mylib.my_op.default``) to a function that computes it with
                           torch operations, for operators that PyTorch cannot decompose itself,
                           such as custom operators. Needs decompose=True.

    Returns:
    RModel: The parsed model.

    Example usage:
    ~~~ {.py}
    torch = import_torch()
    import sofie

    ep = torch.export.export(model.eval(), (torch.rand(120, 1),))
    torch.export.save(ep, "trained_model_dense.pt2")
    rmodel = sofie.pytorch.parse("trained_model_dense.pt2")
    ~~~
    """

    torch = import_torch()

    if decompositions is not None and not decompose:
        raise ValueError("decompositions need decompose=True")
    ep = _load_exported_program(model, input_shapes)
    if decompose:
        ep = _lower(ep, decompositions)

    unsupported = {}
    for node in _calls(ep):
        if not _supported(str(node.target)):
            unsupported.setdefault(str(node.target), 0)
            unsupported[str(node.target)] += 1
    if unsupported:
        raise RuntimeError(
            "SOFIE - The model uses operators that the PyTorch parser cannot translate: "
            + ", ".join(f"{op} (x{n})" for op, n in unsupported.items())
            + ". Everything in a model has to break down into ATen operators: "
            + ("" if decompose else "try decompose=True; ")
            + "for a custom operator, pass its decomposition in `decompositions`. "
            + "sofie.pytorch.list_operators shows the operators of a model."
        )

    # create new RModel object
    name = os.path.basename(os.fspath(model)) if isinstance(model, (str, os.PathLike)) else "PyTorchModel"
    parsetime = time.asctime(time.gmtime(time.time()))
    rmodel = sofie.RModel(name, parsetime)

    print("Torch Version: " + torch.__version__)

    graph = ep.graph_module.graph
    sig = ep.graph_signature
    ctx = parse_context(rmodel)
    output_node = next(n for n in graph.nodes if n.op == "output")
    outputs = output_node.args[0] if isinstance(output_node.args[0], (tuple, list)) else [output_node.args[0]]
    ctx.graph_outputs = {o.name for o in outputs}

    # Classify the placeholders of the graph: weights/buffers/constants vs. user inputs
    weights = {}  # placeholder name -> tensor
    user_inputs = []
    for spec in sig.input_specs:
        placeholder = spec.arg.name
        kind = spec.kind.name
        if kind in ("PARAMETER", "BUFFER"):
            weights[placeholder] = ep.state_dict[spec.target]
        elif kind == "CONSTANT_TENSOR":
            weights[placeholder] = ep.constants[spec.target]
        elif kind == "USER_INPUT":
            user_inputs.append(placeholder)
        else:
            raise RuntimeError(f"SOFIE - Unsupported - graph input of kind {kind}")
    for placeholder in list(weights) + user_inputs:
        ctx._used.add(clean(placeholder))

    nodes = {n.name: n for n in graph.nodes}
    folded = _folded(graph)

    # Iterate over the nodes of the graph and add the equivalent operators to the RModel
    rmodel.add_needed_std_lib("cmath")
    for node in graph.nodes:
        if node.op != "call_function" or node.name in folded or str(node.target) in ignored_targets:
            continue
        handler = aten_handlers[str(node.target)]
        ctx._used.add(ctx.name(node))
        if ctx.try_fold(node):
            continue
        ops = handler.fn(node, ctx)
        if handler.blas_routines:
            rmodel.add_blas_routines(handler.blas_routines)
        for op in ops if isinstance(ops, list) else [ops]:
            rmodel.add_operator(op)

    # Add the weights as initialized tensors. Only the ones that are used by some operator
    # are added.
    used = {arg.name for node in graph.nodes if node.op == "call_function" for arg in node.all_input_nodes}
    for placeholder, tensor in weights.items():
        if placeholder not in used:
            continue
        value = tensor.detach().cpu().numpy()
        add_initialized_tensor(rmodel, clean(placeholder), etype(tensor.dtype), list(value.shape), value.flatten())

    # Add the input tensor infos
    if input_shapes is not None and len(input_shapes) != len(user_inputs):
        raise RuntimeError(f"The model has {len(user_inputs)} inputs, but {len(input_shapes)} input shapes were given")
    if input_dtypes is not None and len(input_dtypes) != len(user_inputs):
        raise RuntimeError(f"The model has {len(user_inputs)} inputs, but {len(input_dtypes)} input types were given")
    for i, input_name in enumerate(user_inputs):
        fake = nodes[input_name].meta["val"]
        exported_shape = []
        for dim in fake.shape:
            if not isinstance(dim, int):
                raise RuntimeError(
                    f"SOFIE - Unsupported - input {input_name} has the dynamic dimension {dim}; "
                    "export the model with static input shapes"
                )
            exported_shape.append(int(dim))
        if input_shapes is not None and [int(d) for d in input_shapes[i]] != exported_shape:
            raise RuntimeError(
                f"The shape {list(input_shapes[i])} given for input {input_name} does not match the shape "
                f"{exported_shape} the model was exported with"
            )
        input_dtype = input_dtypes[i] if input_dtypes is not None else etype(fake.dtype)
        rmodel.add_input_tensor_info(clean(input_name), input_dtype, exported_shape)
        rmodel.add_input_tensor_name(clean(input_name))

    # Extract the output tensor names
    output_names = []
    for out in outputs:
        if out.op == "placeholder":
            # a graph input that is also an output: SOFIE needs an operator to produce the output
            copy = ctx.new_name(out.name + "_out")
            rmodel.add_operator(sofie.ROperator_Identity["float"](ctx.name(out), copy))
            output_names.append(copy)
        else:
            output_names.append(ctx.name(out))
    rmodel.add_output_tensor_name_list(output_names)

    return rmodel
