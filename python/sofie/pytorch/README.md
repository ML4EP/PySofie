## SOFIE-PyTorch parser

SOFIE needs the operator graph of the model together with its weights. The
parser reads the `torch.export` program (an `ExportedProgram`: an FX graph of
ATen operators, the weights and the input shapes), either from a `.pt2` file,
from an `ExportedProgram` object, or by exporting an `nn.Module` itself.

Each ATen operator is translated to one or several SOFIE `ROperator`s. Operators that have no direct translation are first lowered to core ATen
with `ExportedProgram.run_decompositions()`, the decomposition AOTInductor
applies, so that everything that decomposes into supported operators can be
parsed. Operators that have a SOFIE equivalent (`linear`, `gelu`, `layer_norm`,
`scaled_dot_product_attention`, ...) are kept out of the decomposition so that
they map to a single SOFIE operator. Constants that are known at parse time are folded into constant tensors.

Supported ATen operators:

- arithmetic: `add`, `sub`, `rsub`, `mul`, `div`, `pow`, `neg`, `abs`, `exp`,
  `log`, `log1p`, `expm1`, `sqrt`, `rsqrt`, `reciprocal`, `sin`, `cos`,
  `floor`, `erf`, `maximum`, `minimum`, `clamp`, `hardtanh`
- activations: `relu`, `sigmoid`, `tanh`, `gelu`, `elu`, `selu`, `leaky_relu`,
  `hardsigmoid`, `hardswish`, `softmax`, `log_softmax`
- comparison and selection: `gt`, `ge`, `lt`, `le`, `eq`, `ne`,
  `logical_not`, `where`, type conversion (`_to_copy`)
- shapes: `view`, `reshape`, `flatten`, `squeeze`, `unsqueeze`, `permute`,
  `transpose`, `t`, `expand`, `repeat`, `slice`, `select`, `cat`, `split`,
  `constant_pad_nd`, `embedding`, `clone` and friends
- reductions: `mean`, `sum`, `amax`, `amin`, `prod`, `var`, `cumsum`
- linear algebra: `linear`, `addmm`, `mm`, `bmm`, `matmul`
- neural networks: `convolution` (1D to 3D, and transposed), `avg_pool2d`,
  `max_pool2d`, `adaptive_avg_pool2d`, batch, layer, group and instance
  normalization, `scaled_dot_product_attention`
- tensor creation: `full`, `zeros`, `ones`, `arange`

This covers, e.g., MLPs, CNNs with batch norm and residual connections,
transformer encoder layers, `nn.MultiheadAttention`, and `nn.RNN`, `nn.GRU` and `nn.LSTM`. The tests check the numerical
results of all of them against PyTorch. Arguments that SOFIE
cannot express raise an error naming the node.

### Listing the operators of a model

```python
import sofie

for op in sofie.pytorch.list_operators("model.pt2"):    # also takes an ExportedProgram or nn.Module
    print(op["op"], op["count"], op["supported"])
```

or from the command line (the exit status is 1 if an operator is not supported):

```sh
python -m sofie.pytorch model.pt2
```

The list is that of the operators the parser would see, i.e. after the
decomposition (`--no-decompose` / `decompose=False` shows the program as
exported). It works on the operator names; unsupported arguments of a supported
operator are only reported when parsing. `sofie.pytorch.parse` also reports all the
unsupported operators of a model at once.

### Custom operators

Everything in a model has to break down into ATen operators. A function written
with torch operations does (it is traced into ATen operators when the model is
exported). A *custom operator* (`torch.library.custom_op`, ...) does not: it
stays an opaque node such as `torch.ops.mylib.my_op.default`, and `parse` stops
with an error that names it. Give its decomposition, a function that computes it
with torch operations:

```python
rmodel = sofie.pytorch.parse("model.pt2", decompositions={torch.ops.mylib.my_op.default: lambda x: x * 3.0})
```

`sofie.pytorch.list_operators(..., decompositions=...)` takes the same argument.
`register_operator(target, make_op)` adds or overrides the translation of an ATen
operator by building SOFIE operators from the FX node (see `pytorch/aten.py` for
examples and `pytorch/utils.py` for the helpers).
