import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset

import os
from io import BytesIO
import pickle
import copy
import json
import random
import time

import numpy as np

from Network import Model, Hook
from AdversarialAttacks import Model as AdvModel
from PIL import Image

from time import monotonic_ns as perf_counter_ns

import warnings
warnings.filterwarnings("ignore")

class TestDataset(Dataset):
    def __init__(self, datapath, transforms=None):
        self.file_exts = ("png", "jpg")
        
        self.images = list(os.path.join(datapath, item) for item in os.listdir(datapath) if item.split(".")[-1].lower() in self.file_exts)
        self.transforms = transforms

    
    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        imgn = image_path.split("/")[-1].split(".")[0].split()[0].strip()

        return (image, imgn)

class TestTransform():
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            # transforms.TenCrop(224),
            # transforms.Lambda(
            #     lambda crops: torch.stack(
            #         [transforms.PILToTensor()(crop) for crop in crops]
            #     )
            # ),
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x / 255),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

def test_dataloader(
    test_dir, transforms_test, workers
):
    return DataLoader(
        TestDataset(test_dir, transforms=transforms_test),
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        drop_last=False,
    )



def testing(model, test, device, outfile=None):
    accs = []
    aps = []
    log = {}

    model.eval()
    y_pred = []

    times, timen = 0, 0

    with torch.no_grad():
        for data in test:
            images, imgn = data
            imgn = imgn[0]
            images = images.to(device)
            t1 = perf_counter_ns()
            outputs = model(images.view(1, 3, 224, 224))[0]
            t2 = perf_counter_ns()
            #print("Time:", (t2 - t1)/(1000000))
            times += (t2 - t1)/(1000000)
            timen += 1

            y_out = torch.sigmoid(outputs.view(-1)).item()
            y_out_pred = (1 if y_out > 0.5 else 0)

            y_img = {
                "index" : imgn,
                "prediction" : "fake" if y_out_pred else "real"
            }
            y_pred.append(y_img)
            print(imgn.ljust(20), "FAKE" if y_out_pred else "REAL")

    #print("Final Time:", times/timen)
    if outfile is not None:
        print("Creating JSON output.")
        with open(outfile, "w") as h:
            json.dump(y_pred, h, indent=0)
            print("JSON saved.")

def load_trained_model(model, backbone, ckpt_path, advs_path=None, device="cpu"):
    if(ckpt_path):
        opts = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(opts)
        print("Loaded Checkpoint:", ckpt_path)

    if(advs_path):
        advmodel = AdvModel(backbone, num_classes=10, training=False, device=device)
        model_advs = torch.load(advs_path, weights_only=True)
        advmodel.load_state_dict(model_advs["model_state_dict"])

        model.visiual_encoder = advmodel.visual_encoder.to(device).requires_grad_(False)

        model.hooks = [
            Hook(name, module)
            for name, module in model.visiual_encoder.named_modules()
            if "layer_norm2" in name
        ]

        print("Loaded Adversarial Model:", ckpt_path)


def main():
    test_dir = "enter_your_path"
    out_file = "enter_your_path"
    
    backbone = ("wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M", 512)
    nprojs = 2
    proj_dims = backbone[1]
    workers = 1
    #device = "cuda:0"
    device = "cpu"

    model = Model(backbone, nprojs, proj_dims, device).to(device)
    #print(model)

    ckpt_path = "enter_your_path"
    advs_path = "enter_your_path"

    load_trained_model(model, backbone[0], ckpt_path, advs_path, device)

    transform = TestTransform.transform
    dataloader = test_dataloader(test_dir, transform, workers)

    testing(model, dataloader, device, out_file)

if __name__=="__main__":
    main()