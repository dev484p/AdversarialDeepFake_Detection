import torch
import torch.nn as nn
from transformers import CLIPVisionModel
import torchvision
import torchattacks
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

def generate_adversarial_examples(model, images, labels, attack_type, device, **params):
    model.eval()

    # Create the specified attack
    if attack_type == "FGSM":
        attack = torchattacks.FGSM(model, eps=params.get("eps", 8 / 255))
    elif attack_type == "PGD":
        attack = torchattacks.PGD(
            model, 
            eps=params.get("eps", 8 / 255), 
            alpha=params.get("alpha", 2 / 255), 
            steps=params.get("steps", 10)
        )
    elif attack_type == "DeepFool":
        attack = torchattacks.DeepFool(model, steps=params.get("steps", 50))
    elif attack_type == "CW":
        attack = torchattacks.CW(
            model, 
            c=params.get("c", 1), 
            kappa=params.get("kappa", 0), 
            steps=params.get("steps", 10), 
            lr=params.get("lr", 0.01)
        )
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    adversarial_images = attack(images, labels)
    return adversarial_images


def train_with_progressive_adversarial_examples(
    clean_model, robust_model, train_loader, optimizer, device, attack_schedule, checkpoint_dir="./checkpoints/adv_cosine", lossfn = F.mse_loss, batch_size=1
):
    import os

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    robust_model.to(device)
    clean_model.to(device)
    robust_model.train()
    clean_model.eval().requires_grad_(False)

    for attack_idx, (attack_type, attack_params) in enumerate(attack_schedule):
        print(f"\nTraining with {attack_type} attack: {attack_params}\n")

        for epoch in range(attack_params["epochs"]):
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                adversarial_images = generate_adversarial_examples(
                    clean_model, images, labels, attack_type, device, **attack_params
                )

                optimizer.zero_grad()
                clean_logits, clean_features = clean_model(images, return_features=True)
                adversarial_logits, adversarial_features = robust_model(
                    adversarial_images, return_features=True
                )

                #feature_loss = F.mse_loss(clean_features, adversarial_features)
                one = torch.ones((labels.shape[0])).to(device).float()
                feature_loss = lossfn(clean_features.flatten(start_dim=1), adversarial_features.flatten(start_dim=1), one)
                classification_loss = F.cross_entropy(adversarial_logits, labels)
                total_loss = feature_loss + classification_loss

                total_loss.backward()
                optimizer.step()

                if batch_idx % 2 == 0:
                    print(
                        f"Epoch {epoch + 1}/{attack_params['epochs']}, Batch {batch_idx + 1}/{len(train_loader)}: "
                        f"Feature Loss = {feature_loss.item():.4f}, "
                        f"Classification Loss = {classification_loss.item():.4f}, "
                        f"Total Loss = {total_loss.item():.4f}"
                    )

        # Save checkpoint after each attack phase
        checkpoint_path = os.path.join(checkpoint_dir, f"attack_{attack_idx + 1}_{attack_type}.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "attack_type": attack_type,
                "attack_params": attack_params,
                "model_state_dict": robust_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved after {attack_type} attack at {checkpoint_path}\n")


class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)
        self.input = None
        self.output = None

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class Model(nn.Module):
    def __init__(self, backbone, num_classes, training, device):
        super().__init__()
        self.device = device

        self.clip = CLIPVisionModel.from_pretrained(backbone).to(device)
        
        for name, param in self.clip.named_parameters():
            param.requires_grad = training

        self.visual_encoder = self.clip.vision_model

        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.vision_model.named_modules()
            if "layer_norm2" in name
        ]

        self.classifier = nn.Linear(self.visual_encoder.config.hidden_size, num_classes)
    
    def forward(self, x, return_features=False):
        features = self.visual_encoder(x).last_hidden_state

        intermediate_features = torch.stack([h.output for h in self.hooks], dim=2)[:, 0, :, :]
        logits = self.classifier(features[:, 0, :])

        if return_features:
            return logits, intermediate_features
        return logits


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
    #backbone = "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"

    frozen_clip = Model(backbone, num_classes=10, training=False, device=device)  # Parameters frozen
    robust_clip = Model(backbone, num_classes=10,  training=True, device=device)   # Parameters trainable

    ckpt_path = "enter_your_path"
    ckpt_path = ""
    if(ckpt_path):
        modelp = torch.load(ckpt_path, weights_only=True)
        robust_clip.load_state_dict(modelp["model_state_dict"])

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 64
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.Adam(robust_clip.parameters(), lr=5e-6)
    cosineloss = torch.nn.CosineEmbeddingLoss()

    attack_schedule = [
        ("FGSM", {"eps": 4 / 255, "epochs": 5}),  # Weak FGSM attack
        ("FGSM", {"eps": 8 / 255, "epochs": 1}),  # Stronger FGSM attack
        #("PGD", {"eps": 8 / 255, "alpha": 2 / 255, "steps": 10, "epochs": 5}),  # Weak PGD
        ("PGD", {"eps": 16 / 255, "alpha": 4 / 255, "steps": 20, "epochs": 1}),  # Stronger PGD
        #("DeepFool", {"steps": 30, "epochs": 1}),  # DeepFool attack
        ("CW", {"c": 1, "kappa": 0, "steps": 10, "lr": 0.01, "epochs": 1}),  # CW attack
    ]

    # Train the robust model progressively
    train_with_progressive_adversarial_examples(
        frozen_clip, robust_clip, train_loader, optimizer, device, attack_schedule, lossfn=cosineloss, batch_size=batch_size
    )
    

if __name__ == "__main__":
    main()
