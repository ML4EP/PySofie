"""Modules that exercise the ATen operators (after decomposition) that the PyTorch parser supports."""

import torch, torch.nn as nn, torch.nn.functional as F, collections
class fn(nn.Module):
    """A module that calls a function."""

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, *args):
        return self.f(*args)


def mods():
    yield 'linear', nn.Linear(4,5), (torch.randn(2,4),)
    yield 'linear3d', nn.Linear(4,5), (torch.randn(2,3,4),)
    yield 'conv2d', nn.Conv2d(3,4,3,padding=1,stride=2), (torch.randn(2,3,8,8),)
    yield 'conv1d', nn.Conv1d(3,4,3), (torch.randn(2,3,8),)
    yield 'bn2d', nn.BatchNorm2d(3).eval(), (torch.randn(2,3,4,4),)
    yield 'bn1d', nn.BatchNorm1d(3).eval(), (torch.randn(2,3),)
    yield 'ln', nn.LayerNorm(4), (torch.randn(2,3,4),)
    yield 'gn', nn.GroupNorm(2,4), (torch.randn(2,4,3,3),)
    for n in ['GELU','SiLU','Softplus','ELU','LeakyReLU','ReLU6','Hardtanh','Hardswish','Hardsigmoid','Mish','Softsign','SELU','Tanh','Sigmoid','ReLU','Tanhshrink','LogSigmoid','CELU']:
        yield n, getattr(nn,n)(), (torch.randn(2,4),)
    yield 'softmax', nn.Softmax(dim=1), (torch.randn(2,4),)
    yield 'logsoftmax', nn.LogSoftmax(dim=1), (torch.randn(2,4),)
    yield 'maxpool', nn.MaxPool2d(2), (torch.randn(1,2,4,4),)
    yield 'avgpool', nn.AvgPool2d(2), (torch.randn(1,2,4,4),)
    yield 'adaptavg', nn.AdaptiveAvgPool2d(1), (torch.randn(1,2,4,4),)
    yield 'flatten', nn.Flatten(), (torch.randn(2,3,4),)
    yield 'dropout', nn.Dropout(0.5).eval(), (torch.randn(2,4),)
    yield 'embedding', nn.Embedding(10,3), (torch.tensor([[1,2],[3,4]]),)
    x=torch.randn(2,3,4); y=torch.randn(2,3,4)
    for name,f,args in [
      ('add',lambda a,b:a+b,(x,y)),('sub',lambda a,b:a-b,(x,y)),('mul',lambda a,b:a*b,(x,y)),('div',lambda a,b:a/b,(x,y)),
      ('addscalar',lambda a:a+1.5,(x,)),('mulscalar',lambda a:a*2,(x,)),('rsub',lambda a:1-a,(x,)),('pow',lambda a:a**2,(x,)),
      ('neg',lambda a:-a,(x,)),('exp',torch.exp,(x,)),('log',lambda a:torch.log(a.abs()+1),(x,)),('sqrt',lambda a:torch.sqrt(a.abs()),(x,)),
      ('rsqrt',lambda a:torch.rsqrt(a.abs()+1),(x,)),('abs',torch.abs,(x,)),('erf',torch.erf,(x,)),('clamp',lambda a:a.clamp(-1,1),(x,)),
      ('mean',lambda a:a.mean(-1),(x,)),('sum',lambda a:a.sum(1,keepdim=True),(x,)),('amax',lambda a:a.amax(-1),(x,)),('var',lambda a:a.var(-1),(x,)),
      ('cat',lambda a,b:torch.cat([a,b],1),(x,y)),('stack',lambda a,b:torch.stack([a,b]),(x,y)),
      ('reshape',lambda a:a.reshape(6,4),(x,)),('view',lambda a:a.view(2,-1),(x,)),('squeeze',lambda a:a.unsqueeze(0).squeeze(0),(x,)),
      ('permute',lambda a:a.permute(2,0,1),(x,)),('transpose',lambda a:a.transpose(1,2),(x,)),('slice',lambda a:a[:,1:],(x,)),('select',lambda a:a[:,0],(x,)),
      ('matmul',lambda a,b:a@b.transpose(1,2),(x,y)),('bmm',lambda a,b:torch.bmm(a,b.transpose(1,2)),(x,y)),
      ('where',lambda a,b:torch.where(a>0,a,b),(x,y)),('gt',lambda a:(a>0).float(),(x,)),('expand',lambda a:a[:, :1].expand(2,3,4),(x,)),
      ('to',lambda a:a.to(torch.float64).to(torch.float32),(x,)),('chunk',lambda a:a.chunk(2,-1)[0],(x,)),('sdpa',lambda a,b:F.scaled_dot_product_attention(a,b,b),(x,y)),
      ('maximum',torch.maximum,(x,y)),('floor',lambda a:torch.floor(a),(x,)),('sin',torch.sin,(x,)),('cos',torch.cos,(x,)),('zeros',lambda a:a+torch.zeros(3,4),(x,)),
    ]:
        yield name, fn(f), args


def models():
    """Larger models: blocks made of many operators, with user-defined functions."""

    class res(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
            self.b1 = nn.BatchNorm2d(4)
            self.c2 = nn.Conv2d(4, 4, 3, padding=1)
            self.b2 = nn.BatchNorm2d(4)

        def forward(self, x):
            y = F.relu(self.b1(self.c1(x)))
            y = self.b2(self.c2(y))
            return F.relu(x + y)

    class net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(1, 4, 3)
            self.res = res()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(4, 10)

        def forward(self, x):
            return self.fc(self.pool(self.res(F.relu(self.stem(x)))).flatten(1))

    def gelu_ish(x):  # a user-defined function: it is traced into ATen operators
        return x * torch.sigmoid(1.702 * x)

    class user_defined(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(6, 6)

        def forward(self, x):
            return F.softmax(gelu_ish(self.fc(x)) + torch.tanh(x), -1)

    class first(nn.Module):
        """The first output of a module that returns a tuple."""

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args):
            return self.module(*args)[0]

    yield "resnet_block", net(), (torch.randn(2, 1, 8, 8),)
    yield "user_defined_function", user_defined(), (torch.randn(2, 6),)
    yield "transformer_encoder", nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True), (torch.randn(2, 5, 8),)
    yield "multihead_attention", first(nn.MultiheadAttention(8, 2, batch_first=True)), (torch.randn(2, 5, 8),) * 3
    yield "lstm", first(nn.LSTM(4, 6, batch_first=True)), (torch.randn(2, 3, 4),)
    yield "gru", first(nn.GRU(4, 6, batch_first=True)), (torch.randn(2, 3, 4),)
    yield "rnn", first(nn.RNN(4, 6, batch_first=True)), (torch.randn(2, 3, 4),)
    yield "rmsnorm", nn.RMSNorm(4), (torch.randn(2, 4),)
    yield "prelu", nn.PReLU(), (torch.randn(2, 4),)
    yield "maxpool1d", nn.MaxPool1d(2), (torch.randn(1, 2, 6),)
    yield "avgpool1d", nn.AvgPool1d(2), (torch.randn(1, 2, 6),)
    yield "std", fn(lambda x: x.std(1)), (torch.randn(2, 4),)
    yield "conv_transpose2d", nn.ConvTranspose2d(2, 3, 3, stride=2), (torch.randn(1, 2, 4, 4),)
    yield "instance_norm", nn.InstanceNorm2d(3, affine=True), (torch.randn(2, 3, 4, 4),)
    yield "repeat", fn(lambda x: x.repeat(2, 1, 3)), (torch.randn(2, 4),)
    yield "pad", nn.ZeroPad2d((1, 2, 0, 1)), (torch.randn(1, 2, 3, 3),)
    yield "cumsum", fn(lambda x: torch.cumsum(x, 1)), (torch.randn(2, 4),)
    yield "tril", fn(lambda x: torch.tril(x)), (torch.randn(4, 4),)
    yield "normalize", fn(lambda x: F.normalize(x, dim=1)), (torch.randn(2, 4),)


def cases(names=None):
    available = list(mods()) + list(models())
    if names is not None:
        unknown = set(names) - {name for name, _, _ in available}
        if unknown:
            raise ValueError(f"unknown cases: {sorted(unknown)}")
    for name, module, args in available:
        if names is None or name in names:
            yield name, module, args
