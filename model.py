import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_gan_losses_fn, VGGLoss, crop, add_sp
from models.content_encoder import ContentEncoder
from models.gaze_correction import GazeCorrection
from models.discriminator import Discriminator

import sys
import os

class GazeGan(nn.Module):
    V1_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "pretrained", "checkpointv1.pt")
    V2_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "pretrained", "checkpointv2.pt")
    V1_CHECKPOINT_URL = ""
    V2_CHECKPOINT_URL = ""

    def __init__(self, ngf=16, ndf=16, num_layers_r=3, num_layers_g=5, num_layers_d=5, version="V2") -> None:
        super().__init__()
        self.Gr = ContentEncoder(ngf=ngf, num_layers_r=num_layers_r, use_sp=False)
        
        for param in self.Gr.parameters():
            param.requires_grad = False

        self.Gx = GazeCorrection(ngf=ngf, num_layers_g=num_layers_g, use_sp=False)
        self.Dx = Discriminator(ndf=ndf, num_layers_d=num_layers_d, use_sp=True, version=version)

        self.d_loss_fn, self.g_loss_fn = get_gan_losses_fn()
        self.vgg_loss = VGGLoss()

        # load the model from the checkpoint, pretrained/checkpointv1.pt for V1 and pretrained/checkpointv2.pt for V2
        if version == "V1":
            if not os.path.exists(self.V1_CHECKPOINT_PATH):
                # download the checkpoint from url
                raise FileNotFoundError("Pretrained checkpoint for V1 not found.")
            self.load_state_dict(torch.load(self.V1_CHECKPOINT_PATH))
        elif version == "V2":
            if not os.path.exists(self.V2_CHECKPOINT_PATH):
                raise FileNotFoundError("Pretrained checkpoint for V2 not found.")
            self.load_state_dict(torch.load(self.V2_CHECKPOINT_PATH))


    def inpaiting(self, x, x_mask, x_left_eye, x_right_eye):
        xc = x * (1 - x_mask)

        # extract content feature
        left_eye_content_fp = self.Gr(x_left_eye)
        right_eye_content_fp = self.Gr(x_right_eye)
        x_content_fp = torch.cat([left_eye_content_fp, right_eye_content_fp], dim=1)
        # x_content_fp: [batch_size, 256 * 2]

        # reconstruct
        xr = self.Gx(xc, x_mask, x_content_fp)
        # xr: [batch_size, n_channels=3, h=64, w=64]

        return xr
    
    def forward(self, x, x_mask, x_left_pos, x_right_pos):
        x_left_eye, x_right_eye = crop(x, x_left_pos, x_right_pos)
        xr = self.inpaiting(x, x_mask, x_left_eye, x_right_eye)
        xr_left_eye, xr_right_eye = crop(xr, x_left_pos, x_right_pos)
        return xr, xr_left_eye, xr_right_eye
    
    def get_loss(self, x, x_mask, x_left_pos, x_right_pos):
        # x: [batch_size, n_channels=3, h=64, w=64]
        # x_mask: [batch_size, n_channels=1, h=64, w=64]
        lam_rec = 10
        lam_vgg = 0.1

        x_left_eye, x_right_eye = crop(x, x_left_pos, x_right_pos)

        xr = self.inpaiting(x, x_mask, x_left_eye, x_right_eye)

        xr_left_eye, xr_right_eye = crop(xr, x_left_pos, x_right_pos)

        # real and fake loss
        dx_logits = self.Dx(x, x_left_eye, x_right_eye)
        gx_logits = self.Dx(xr, xr_left_eye, xr_right_eye)

        d_loss = self.d_loss_fn(dx_logits, gx_logits)
        g_loss = self.g_loss_fn(gx_logits)

        # reconstruction loss
        recon_loss = F.l1_loss(xr, x)

        # vgg loss
        vgg_loss = self.vgg_loss(x_left_eye, xr_left_eye) \
                    + self.vgg_loss(x_right_eye, xr_right_eye)
        
        G_loss = g_loss + lam_rec * recon_loss + lam_vgg * vgg_loss
        D_loss = d_loss
        return G_loss, D_loss


# class AngleEncoder(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         nef = 32
#         num_layers = 3
#         out_c = nef

#         self.conv2d_first = nn.Conv2d(in_channels=3, out_channels=out_c, kernel_size=7, stride=1, padding=3)
#         self.in_first = nn.InstanceNorm2d(out_c)
#         self.relu = nn.ReLU()

#         self.conv2d_base = nn.ModuleList()
#         self.in_base = nn.ModuleList()
#         for i in range(num_layers):
#             in_c, out_c = out_c, min(nef * 2 ** (i + 1), 128)
#             self.conv2d_base.append(nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1))
#             self.in_base.append(nn.InstanceNorm2d(out_c))
        
#         self.fc_en = nn.LazyLinear(out_features=2)
    
#     def forward(self, input_x):
#         x = input_x
#         x = self.relu(self.in_first(self.conv2d_first(x)))
#         for conv, in_layer in zip(self.conv2d_base, self.in_base):
#             x = self.relu(in_layer(conv(x)))
#         x = x.view(x.size(0), -1)
#         bottleneck = self.fc_en(x)
#         return bottleneck
        


# class GazeAnimation(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         ngf = 16
#         num_layers = 5
#         out_c = ngf
#         self.h, self.w = h, w = 256, 256

#         self.conv2d_first = nn.Conv2d(in_channels=3 + 1, out_channels=out_c, kernel_size=7, stride=1, padding=3)
#         self.in_first = nn.InstanceNorm2d(out_c)
#         self.lrelu = nn.LeakyReLU(0.2)

#         self.conv2d_base = nn.ModuleList()
#         self.in_base = nn.ModuleList()
#         u_out_c_list = []
#         for i in range(num_layers):
#             u_out_c_list.append(out_c)
#             in_c, out_c = out_c, min(ngf * 2 ** (i + 1), 256)
#             self.conv2d_base.append(nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1))
#             self.in_base.append(nn.InstanceNorm2d(out_c))

#         self.fc1 = nn.LazyLinear(out_features=256)

#         self.fc2 = nn.LazyLinear(out_features=256 * h * w)

#         # u net upsampling
#         self.deconv_base = nn.ModuleList()
#         self.in_deconv_base = nn.ModuleList()
#         for i in range(num_layers):
#             in_c, out_c = out_c + u_out_c_list[num_layers - i - 1], max(out_c // 2, 16)
#             self.deconv_base.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1))
#             self.in_deconv_base.append(nn.InstanceNorm2d(out_c))
#         self.relu = nn.ReLU()
#         self.conv2d_final = nn.Conv2d(in_channels=out_c, out_channels=3, kernel_size=7, stride=1, padding=3)

#     def forward(self, input_x, img_mask, content_fp, angle_fp):
#         x = torch.cat([input_x, img_mask], dim=1)
#         x = self.lrelu(self.in_first(self.conv2d_first(x)))
#         u_fp_list = []
#         for conv, in_layer in zip(self.conv2d_base, self.in_base):
#             x = self.lrelu(in_layer(conv(x)))
#             u_fp_list.append(x)

#         x = x.view(x.shape[0], -1)
#         bottleneck = self.fc1(x)
#         bottleneck = torch.cat([bottleneck, content_fp, angle_fp], dim=1)

#         de_x = self.lrelu(self.fc2(bottleneck))
#         h, w = x.shape[2], x.shape[3]
#         de_x = de_x.view(de_x.shape[0], -1, h, w)
#         for deconv, in_layer in zip(self.deconv_base, self.in_deconv_base):
#             de_x = torch.cat([de_x, u_fp_list.pop()], dim=1)
#             de_x = self.relu(in_layer(deconv(de_x)))

#         de_x = self.conv2d_final(de_x)
#         return input_x + torch.tanh(de_x) * img_mask

