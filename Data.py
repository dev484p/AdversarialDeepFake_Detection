import os
import torch
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

data  = "enter_your_path"

class TrainingDatasetLDM(Dataset):
    def __init__(self, split, transforms=None):
        self.real = [
            (f"{data}/train/{x.split('_')[0]}/0_real/{x.split('_')[1]}", 0)
            for x in pd.read_csv(
                f"{data}/latent_diffusion_trainingset/{split}/real_lsun.txt",
                header=None,
            )
            .values.reshape(-1)
            .tolist()
            if x.endswith('.jpg')
        ] + [
            (
                (
                    f"{data}/coco/train2017/{x}"
                ),
                0,
            )
            for x in pd.read_csv(
                f"{data}/latent_diffusion_trainingset/{split}/real_coco.txt", header=None
            )
            .values.reshape(-1)
            .tolist()
            if x.endswith('.jpg')
        ]

        fake_dir = f"{data}/latent_diffusion_trainingset/"
        self.fake = [
            (f"{fake_dir}{split}/{x}/{y}", 1)
            for x in os.listdir(f"{fake_dir}{split}")
            if os.path.isdir(f"{fake_dir}{split}/{x}")
            for y in os.listdir(f"{fake_dir}{split}/{x}")
            if y.endswith('.jpg')
        ]


        self.images = self.real + self.fake
        random.shuffle(self.images)

        self.transforms = transforms

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, target = self.images[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)
        return [image, target]
    


class EvaluationDataset(Dataset):
    def __init__(self, generator, transforms=None, perturb=None):
        if generator in ["cyclegan", "progan", "stylegan", "stylegan2"]:
            self.real = [
                (f"{data}/test/{generator}/{y}/0_real/{x}", 0)
                for y in os.listdir(f"{data}/test/{generator}")
                for x in os.listdir(f"{data}/test/{generator}/{y}/0_real")
                if x.endswith('.jpg')
            ]
            self.fake = [
                (f"{data}/test/{generator}/{y}/1_fake/{x}", 1)
                for y in os.listdir(f"{data}/test/{generator}")
                for x in os.listdir(f"{data}/test/{generator}/{y}/1_fake")
                if x.endswith('.jpg')
            ]
        elif "diffusion_datasets/guided" in generator:
            self.real = [
                (f"{data}/test/diffusion_datasets/imagenet/0_real/{x}", 0)
                for x in os.listdir(f"{data}/test/diffusion_datasets/imagenet/0_real")
                if x.endswith('.jpg')
            ]
            self.fake = [
                (f"{data}/test/{generator}/1_fake/{x}", 1)
                for x in os.listdir(f"{data}/test/{generator}/1_fake")
                if x.endswith('.jpg')
            ]
        elif (
            "diffusion_datasets/ldm" in generator
            or "diffusion_datasets/glide" in generator
            or "diffusion_datasets/dalle" in generator
        ):
            self.real = [
                (f"{data}/test/diffusion_datasets/laion/0_real/{x}", 0)
                for x in os.listdir(f"{data}/test/diffusion_datasets/laion/0_real")
                if x.endswith('.jpg')
            ]
            self.fake = [
                (f"{data}/test/{generator}/1_fake/{x}", 1)
                for x in os.listdir(f"{data}/test/{generator}/1_fake")
                if x.endswith('.jpg')
            ]
        elif any(
            [
                x in generator
                for x in [
                    "biggan",
                    "stargan",
                    "gaugan",
                    "deepfake",
                    "seeingdark",
                    "san",
                    "crn",
                    "imle",
                ]
            ]
        ):
            self.real = [
                (f"{data}/test/{generator}/0_real/{x}", 0)
                for x in os.listdir(f"{data}/test/{generator}/0_real")
                if x.endswith('.jpg')
            ]
            self.fake = [
                (f"{data}/test/{generator}/1_fake/{x}", 1)
                for x in os.listdir(f"{data}/test/{generator}/1_fake")
                if x.endswith('.jpg')
        ]
        elif any(
            [
                x in generator
                for x in [
                    "dalle2",
                    "dalle3",
                    "stable-diffusion-1-3",
                    "stable-diffusion-1-4",
                    "stable-diffusion-2",
                    "stable-diffusion-xl",
                    "glide",
                    "firefly",
                    "midjourney-v5",
                ]
            ]
        ):
            self.real = [(f"{data}/RAISEpng/{x}", 0) for x in os.listdir(f"{data}/RAISEpng") if x.endswith('.jpg')]
            self.fake = [
                (f"{data}/synthbuster/{generator}/{x}", 1)
                for x in os.listdir(f"{data}/synthbuster/{generator}")
                if all([y not in x for y in [".txt", ".py"]]) and x.endswith('.jpg')
            ]

        self.images = self.real + self.fake

        self.transforms = transforms
        self.perturb = perturb

    
    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, target = self.images[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transforms is not None and self.perturb is None:
            image = self.transforms(image)

        return [image, target]
        


