import torch
import torch.nn as nn
import clip

from transformers import CLIPVisionModel
from torchsummary import summary

import warnings
warnings.filterwarnings("ignore")

class Hook:

    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class Model(nn.Module):
    def __init__(
        self,
        backbone,
        nproj,
        proj_dim,
        device,
    ):
        super().__init__()

        self.device = device

        self.clip = CLIPVisionModel.from_pretrained(backbone[0]).to(device)
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        self.visiual_encoder = self.clip.vision_model
        
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.vision_model.named_modules()
            if "layer_norm2" in name
        ]

        self.alpha = nn.Parameter(torch.randn([1, len(self.hooks), proj_dim]))
        proj1_layers = [nn.Dropout()]
        for i in range(nproj):
            proj1_layers.extend(
                [
                    nn.Linear(backbone[1] if i == 0 else proj_dim, proj_dim, bias=False),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj1 = nn.Sequential(*proj1_layers)
        proj2_layers = [nn.Dropout()]
        for _ in range(nproj):
            proj2_layers.extend(
                [
                    nn.Linear(proj_dim, proj_dim, bias=False),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj2 = nn.Sequential(*proj2_layers)
        self.head = nn.Sequential(
            *[
                nn.Linear(proj_dim, proj_dim, bias=False),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, 1),
            ]
        )


    def forward(self, x):
        with torch.no_grad():
            self.visiual_encoder(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)[:, 0, :, :]

        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=1) * g
        z = torch.sum(z, dim=1)
        z = self.proj2(z)

        p = self.head(z)
        return p, z

