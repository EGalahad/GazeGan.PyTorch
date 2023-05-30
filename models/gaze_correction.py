import torch
import torch.nn as nn
from utils import add_sp

class GazeCorrection(nn.Module):
    def __init__(self, ngf=16, num_layers=3, use_sp=False) -> None:
        super().__init__()
        out_c = ngf

        self.conv2d_first = nn.Conv2d(in_channels=3 + 3, out_channels=out_c, kernel_size=7, stride=1, padding=3)
        self.in_first = nn.InstanceNorm2d(out_c, affine=True)
        self.lrelu = nn.LeakyReLU(0.2)

        self.conv2d_base = nn.ModuleList()
        self.in_base = nn.ModuleList()
        u_out_c_list = []
        for i in range(num_layers):
            in_c, out_c = out_c, min(ngf * 2 ** (i + 1), 256)
            self.conv2d_base.append(nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1))
            self.in_base.append(nn.InstanceNorm2d(out_c, affine=True))
            u_out_c_list.append(out_c)

        self.fc1 = nn.Linear(in_features=256 * 8 * 8, out_features=256)

        self.fc2 = nn.Linear(in_features=256 + 256 * 2, out_features=256 * 8 * 8)

        # u net upsampling
        self.deconv_base = nn.ModuleList()
        self.in_deconv_base = nn.ModuleList()
        ngf = out_c
        for i in range(num_layers):
            in_c, out_c = out_c + u_out_c_list[num_layers - i - 1], max(ngf // (2 ** i), 16)
            self.deconv_base.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1))
            self.in_deconv_base.append(nn.InstanceNorm2d(out_c, affine=True))
        self.relu = nn.ReLU()
        self.conv2d_final = nn.Conv2d(in_channels=out_c, out_channels=3, kernel_size=7, stride=1, padding=3)

        if use_sp:
            self.apply(add_sp)
    
    def forward(self, input_x, img_mask, content_fp):
        # repeat img_mask for 3 channels
        img_mask = img_mask.repeat(1, 3, 1, 1)
        x = torch.cat([input_x, img_mask], dim=1)
        x = self.lrelu(self.in_first(self.conv2d_first(x)))
        u_fp_list = []
        for conv, in_layer in zip(self.conv2d_base, self.in_base):
            x = self.lrelu(in_layer(conv(x)))
            u_fp_list.append(x)

        h, w = x.shape[2], x.shape[3]
        x = x.view(x.shape[0], -1)
        bottleneck = self.fc1(x)
        bottleneck = torch.cat([bottleneck, content_fp], dim=1)

        de_x = self.lrelu(self.fc2(bottleneck))
        de_x = de_x.view(de_x.shape[0], -1, h, w)
        for deconv, in_layer in zip(self.deconv_base, self.in_deconv_base):
            de_x = torch.cat([de_x, u_fp_list.pop()], dim=1)
            de_x = self.relu(in_layer(deconv(de_x)))

        de_x = self.conv2d_final(de_x)
        return input_x + torch.tanh(de_x) * img_mask

