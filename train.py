import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataloader import SegmentationDataset
from model import ViTUNet

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', nargs='?', const=10, default = 10, type=int)
parser.add_argument('--output_dir', nargs='?', const="models", default = "models", type=str)
parser.add_argument('--use_tensorboard', nargs='?', const=True, default = True, type=bool)
args = parser.parse_args()

# === Paths ===
train_img_dir = "dataset/train/images"
train_mask_dir = "dataset/train/masks"
val_img_dir = "dataset/eval/images"
val_mask_dir = "dataset/eval/masks"
output_dir = args.output_dir

# === Hyperparams ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = args.epochs
lr = 1e-4
batch_size = 32

# === Setup ===
train_dataset = SegmentationDataset(train_img_dir, train_mask_dir)
val_dataset = SegmentationDataset(val_img_dir, val_mask_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = ViTUNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

writer = None
if(args.use_tensorboard):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter("runs")

# === Training Loop ===
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
        loss = criterion(outputs, masks)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    if(writer):
        writer.add_scalar("Loss/Train", avg_loss, epoch)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    if(writer):
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)

    print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'model.pth')
    torch.save(model.state_dict(), output_path)

