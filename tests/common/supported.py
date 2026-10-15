onnx_cpu = (
    "linear", "linear3d", "conv2d", "conv1d", "bn2d", "bn1d", "ln", "GELU", "SiLU", "Softplus", "ELU",
    "LeakyReLU", "ReLU6", "Hardtanh", "Hardswish", "Hardsigmoid", "Softsign", "SELU", "Tanh", "Sigmoid",
    "ReLU", "Tanhshrink", "LogSigmoid", "softmax", "logsoftmax", "maxpool", "avgpool", "adaptavg", "flatten",
    "dropout", "embedding", "add", "sub", "mul", "div", "addscalar", "mulscalar", "rsub", "pow", "neg", "exp",
    "log", "sqrt", "rsqrt", "abs", "erf", "clamp", "mean", "sum", "amax", "var", "cat", "stack", "reshape",
    "view", "squeeze", "permute", "transpose", "slice", "select", "matmul", "bmm", "where", "gt", "expand",
    "to", "chunk", "maximum", "floor", "sin", "cos", "zeros", "resnet_block", "user_defined_function",
    "transformer_encoder", "multihead_attention", "lstm", "gru", "rnn", "rmsnorm", "maxpool1d", "avgpool1d",
    "std", "conv_transpose2d", "instance_norm", "repeat", "pad", "cumsum", "normalize",
)

torch_alpaka_cuda = (
    "linear", "conv2d", "conv1d", "bn2d", "bn1d", "ln", "gn", "SiLU", "Softplus", "ELU", "LeakyReLU", "ReLU6",
    "Hardtanh", "Mish", "Softsign", "SELU", "Tanh", "Sigmoid", "ReLU", "Tanhshrink", "LogSigmoid", "CELU",
    "softmax", "logsoftmax", "maxpool", "avgpool", "adaptavg", "dropout", "add", "sub", "mul", "div",
    "addscalar", "mulscalar", "rsub", "pow", "neg", "exp", "log", "sqrt", "rsqrt", "abs", "clamp", "mean",
    "sum", "amax", "var", "cat", "stack", "squeeze", "permute", "transpose", "slice", "select", "matmul",
    "bmm", "where", "gt", "expand", "to", "chunk", "sdpa", "maximum", "floor", "sin", "cos", "zeros",
    "resnet_block", "user_defined_function", "transformer_encoder", "multihead_attention", "rnn", "rmsnorm",
    "prelu", "std", "repeat", "pad", "cumsum", "tril", "normalize",
)
