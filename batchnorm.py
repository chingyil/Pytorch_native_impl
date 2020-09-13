import torch
import torch.nn as nn

batch_size, c, w, h = 8, 1, 1, 1
input = torch.rand((batch_size, c, w, h))

class NativeBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super( NativeBatchNorm2d, self ).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = eps

        self.reset_parameters()

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        self.bias = nn.Parameter(torch.zeros_like(self.bias))
        self.weight = nn.Parameter(torch.ones_like(self.weight))

    def forward(self, x):
        if self.training:
            momentum = self.momentum
            mu = x.mean()
            var = ((x - mu) ** 2).sum() / (x.numel() - 1)
            sigma_sq = ((x - mu) ** 2).mean()
            self.running_mean = momentum * mu + (1-momentum) * self.running_mean
            self.running_var = momentum * var + (1-momentum) * self.running_var
            return (x - mu) / (sigma_sq + self.eps).sqrt()

        else:
            mu = self.running_mean.reshape((1, self.num_features, 1, 1))
            var = self.running_var.reshape((1, self.num_features, 1, 1))
            gamma = self.weight.reshape((1, self.num_features, 1, 1))
            beta = self.bias.reshape((1, self.num_features, 1, 1))
            return ((x - mu) / (var + self.eps).sqrt()) * gamma + beta

# Instantiate PyTorch's BatchNorm2d and NativePython's BatchNorm2d
pytorch_bn = nn.modules.BatchNorm2d(c)
native_bn = NativeBatchNorm2d(c)

# Test in training model
res1 = pytorch_bn(input)
res2 = native_bn(input)
assert (res1.norm() - res2.norm()) < 1e-3
assert (pytorch_bn.running_mean - native_bn.running_mean).abs() < 1e-3
assert (pytorch_bn.running_var - native_bn.running_var).abs() < 1e-3

native_bn.eval()
pytorch_bn.eval()
res3 = pytorch_bn(input)
res4 = native_bn(input)
assert (res3.norm() - res4.norm()) < 1e-3
