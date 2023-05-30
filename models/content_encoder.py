import torch
import torch.nn as nn
from utils import add_sp


class ContentEncoder(nn.Module):
    def __init__(self, ngf=16, num_layers_r=3, use_sp=False) -> None:
        """Content Encoder

        Args:
            ngf (int, optional): number of convolution filters after the first convolution, subsequent filter number doubles each layer. Defaults to 16.
            num_layers_r (int, optional): number of convolution layers. Defaults to 3.
            use_sp (bool, optional): whether to use spectral normalization. Defaults to False.
        """
        super().__init__()
        out_c = ngf

        self.conv2d_first = nn.Conv2d(
            in_channels=3, out_channels=out_c, kernel_size=7, stride=1, padding=3)
        self.in_first = nn.InstanceNorm2d(out_c, affine=True)
        self.lrelu = nn.LeakyReLU(0.2)

        self.conv2d_base = nn.ModuleList()
        self.in_base = nn.ModuleList()
        for i in range(num_layers_r):
            in_c, out_c = out_c, min(ngf * 2 ** (i + 1), 128)
            self.conv2d_base.append(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1))
            self.in_base.append(nn.InstanceNorm2d(out_c, affine=True))

        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=256)

        self.use_sp = use_sp
        if use_sp:
            self.apply(add_sp)

    def forward(self, input_x):
        x = input_x
        x = self.lrelu(self.in_first(self.conv2d_first(x)))
        for conv, in_layer in zip(self.conv2d_base, self.in_base):
            x = self.lrelu(in_layer(conv(x)))
        x = x.view(x.size(0), -1)
        bottleneck = self.fc1(x)
        return bottleneck
