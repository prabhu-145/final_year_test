import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from PIL import Image

from generator import Generator
from discriminator import Discriminator

# ------------------
# CONFIG
# ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS =  80          # increase later to 50 or 100
BATCH_SIZE = 4
LR = 0.0002
IMG_SIZE = 256
LAMBDA_L1 = 100

TRAIN_A = "data/paired/trainA"
TRAIN_B = "data/paired/trainB"

CHECKPOINT_DIR = "experiments/checkpoints"
RESULT_DIR = "experiments/results"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ------------------
# DATASET
# ------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dir_A, dir_B, transform):
        self.dir_A = dir_A
        self.dir_B = dir_B
        self.files = sorted(os.listdir(dir_A))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_A = Image.open(os.path.join(self.dir_A, self.files[idx])).convert("RGB")
        img_B = Image.open(os.path.join(self.dir_B, self.files[idx])).convert("RGB")

        return self.transform(img_A), self.transform(img_B)

dataset = PairedDataset(TRAIN_A, TRAIN_B, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------
# MODELS
# ------------------
generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)


criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# ------------------
# TRAINING
# ------------------
for epoch in range(1, EPOCHS + 1):
    for i, (sketch, photo) in enumerate(loader):
        sketch, photo = sketch.to(DEVICE), photo.to(DEVICE)

        # ------------------
        # Train Generator
        # ------------------
        fake_photo = generator(sketch)
        pred_fake = discriminator(sketch, fake_photo)
        valid = torch.ones_like(pred_fake).to(DEVICE)

        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_L1 = criterion_L1(fake_photo, photo)
        loss_G = loss_GAN + LAMBDA_L1 * loss_L1

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # ------------------
        # Train Discriminator
        # ------------------
        pred_real = discriminator(sketch, photo)
        loss_real = criterion_GAN(pred_real, valid)

        pred_fake = discriminator(sketch, fake_photo.detach())
        fake = torch.zeros_like(pred_fake).to(DEVICE)
        loss_fake = criterion_GAN(pred_fake, fake)

        loss_D = (loss_real + loss_fake) * 0.5

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        if epoch % 5 == 0:

            print(
                f"[Epoch {epoch}/{EPOCHS}] "
                f"[Batch {i}/{len(loader)}] "
                f"[D loss: {loss_D.item():.4f}] "
                f"[G loss: {loss_G.item():.4f}]"
            )

    # ------------------
    # SAVE RESULTS
    # ------------------
    save_path = os.path.join(RESULT_DIR, f"epoch_{epoch}")
    os.makedirs(save_path, exist_ok=True)
    save_image(fake_photo[:4], f"{save_path}/generated.png", normalize=True)

    torch.save(generator.state_dict(), f"{CHECKPOINT_DIR}/generator_epoch_{epoch}.pth")

print("✅ Pix2Pix training completed successfully")
