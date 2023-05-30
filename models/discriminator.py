import torch
import torch.nn as nn
from utils import add_sp

class Discriminator(nn.Module):
    def __init__(self, ndf=16, num_layers=5, use_sp=True, version="V2") -> None:
        super().__init__()
        self.d1 = LocalDiscriminator(ndf=ndf, num_layers=num_layers, version=version)
        self.d2 = GlobalDiscriminator(ndf=ndf, num_layers=num_layers)

        if version == "V1":
            self.fc1 = nn.Linear(in_features=256 + 256, out_features=512)
        elif version == "V2":
            self.fc1 = nn.Linear(in_features=256 + 512, out_features=1024)

        self.relu = nn.ReLU()

        if version == "V1":
            self.fc2 = nn.Linear(in_features=512, out_features=1)
        elif version == "V2":
            self.fc2 = nn.Linear(in_features=1024, out_features=1)

        if use_sp:
            self.apply(add_sp)
    
    def forward(self, x, left_eye, right_eye):
        xg_fp = self.d2(x)
        xl_fp = self.d1(torch.cat([left_eye, right_eye], dim=1))
        x = torch.cat([xg_fp, xl_fp], dim=1)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


class LocalDiscriminator(nn.Module):
    def __init__(self, in_c=6, ndf=16, num_layers=5, version="V2") -> None:
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2)

        self.conv2d_base = nn.ModuleList()
        out_c = in_c
        for i in range(num_layers):
            if version == "V1":
                in_c, out_c = out_c, min(ndf * 2 ** (i + 1), 256)
            elif version == "V2":
                in_c, out_c = out_c, min(ndf * 2 ** (i + 1), 512)
            self.conv2d_base.append(nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1))
        
        self.fp = nn.Linear(in_features=out_c * 2 * 2, out_features=out_c)

    def forward(self, x):
        for conv in self.conv2d_base:
            x = self.lrelu(conv(x))
        x = x.view(x.shape[0], -1)
        x = self.fp(x)
        return x

class GlobalDiscriminator(nn.Module):
    def __init__(self, in_c=3, ndf=16, num_layers=5) -> None:
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2)

        self.conv2d_base = nn.ModuleList()
        out_c = in_c
        for i in range(num_layers):
            in_c, out_c = out_c, min(ndf * 2 ** (i + 1), 256)
            self.conv2d_base.append(nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1))
        
        self.fp = nn.Linear(in_features=out_c * 8 * 8, out_features=out_c)

    def forward(self, x):
        for conv in self.conv2d_base:
            x = self.lrelu(conv(x))
        x = x.view(x.shape[0], -1)
        x = self.fp(x)
        return x

        
