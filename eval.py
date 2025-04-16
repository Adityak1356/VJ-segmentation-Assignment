import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataloader import SegmentationDataset
from model import ViTUNet
import os
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', nargs='?', const="models/model.pth", default = "models/model.pth", type=str)
parser.add_argument('--output_dir', nargs='?', const="results", default = "results", type=str)
args = parser.parse_args()
model_path = args.model_path

def calculate_metrics(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection / (union + 1e-7)).mean().item()

    dice = (2 * intersection / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-7)).mean().item()

    correct = (preds == targets).float().sum()
    total = torch.numel(preds)
    pixel_accuracy = (correct / total).item()

    return iou, dice, pixel_accuracy

def evaluate_model(model, test_loader, device):
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    total_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
            outputs = torch.sigmoid(outputs)

            iou, dice, acc = calculate_metrics(outputs, masks)

            total_iou += iou
            total_dice += dice
            total_acc += acc
            num_batches += 1

    print(f"Test IoU: {total_iou / num_batches:.4f}")
    print(f"Test Dice Coefficient: {total_dice / num_batches:.4f}")
    print(f"Test Pixel Accuracy: {total_acc / num_batches:.4f}")
    
    return total_iou / num_batches, total_dice / num_batches, total_acc / num_batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTUNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)  

test_img_dir = "dataset/test/images"
test_mask_dir = "dataset/test/masks"
test_dataset = SegmentationDataset(test_img_dir, test_mask_dir)
test_loader = DataLoader(test_dataset, batch_size=8)

# Evaluate
iou, dice_coeff, pixel_acc = evaluate_model(model, test_loader, device)

results = {
    "Test IoU": round(iou, 4),
    "Test Dice Coefficient": round(dice_coeff, 4),
    "Test Pixel Accuracy": round(pixel_acc, 4)
}


# Set your desired path
save_path = args.output_dir
os.makedirs(save_path, exist_ok=True)  # Make sure directory exists
json_file = os.path.join(save_path, "results.json")

# Write to JSON file
with open(json_file, 'w') as f:
    json.dump(results, f)

print(f"Saved results to {json_file}")
