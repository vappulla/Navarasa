# ============================
# DeiT Small – Train from Scratch on Color Dataset
# With: Grayscale(15%), Saturation(-25% to +25%), Cutout(3×10%), 3 Augs per Image
# ============================

import os, math, time, random
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from timm.models.vision_transformer import vit_small_patch16_224

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================
# PATHS
# ============================

drive_path = "/content/drive/MyDrive"   # change if needed

train_dir = f"{drive_path}/NEW_COLOR_DATASET/train"
val_dir   = f"{drive_path}/NEW_COLOR_DATASET/valid"
test_dir  = f"{drive_path}/NEW_COLOR_DATASET/test"

ckpt_dir = f"{drive_path}/deit_color_checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

checkpoint_path = os.path.join(ckpt_dir, "deit_color_checkpoint.pth")
best_model_path = os.path.join(ckpt_dir, "deit_color_best.pth")

# ============================
# MODEL (FROM SCRATCH)
# ============================

num_classes = 9
img_size = 224
batch_size = 32
num_epochs = 40

model = vit_small_patch16_224(
    num_classes=num_classes,
    img_size=img_size,
    pretrained=False     # IMPORTANT: train from scratch
).to(device)

print("Training DeiT Small from scratch!")


# ============================
# CUSTOM AUGMENTATIONS
# ============================

class RandomFixedGrayscale:
    def __init__(self, p=0.15):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return transforms.functional.rgb_to_grayscale(img, num_output_channels=3)
        return img

class RandomCutout:
    def __init__(self, num_holes=3, size_frac=0.10):
        self.num_holes = num_holes
        self.size_frac = size_frac
    def __call__(self, img):
        h, w = img.size[1], img.size[0]
        mask = np.array(img).copy()

        hole_h = int(h * self.size_frac)
        hole_w = int(w * self.size_frac)

        for _ in range(self.num_holes):
            y = random.randint(0, h - hole_h)
            x = random.randint(0, w - hole_w)
            mask[y:y+hole_h, x:x+hole_w, :] = 0
        return transforms.ToPILImage()(mask)


# ============================
# 3 Augmentations per training sample
# ============================

class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, transforms_aug, times=3):
        self.base_dataset = base_dataset
        self.transforms_aug = transforms_aug
        self.times = times
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        imgs = []
        for _ in range(self.times):
            imgs.append(self.transforms_aug(img))
        return torch.stack(imgs), torch.tensor(label)


# ============================
# TRANSFORMS
# ============================

train_base = datasets.ImageFolder(
    train_dir,
    transform=transforms.Resize((640, 640))    # stretch resize
)

transform_train = transforms.Compose([
    RandomFixedGrayscale(0.15),
    transforms.ColorJitter(saturation=0.25),
    RandomCutout(3, 0.10),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

train_dataset = AugmentedDataset(train_base, transform_train, times=3)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
)

# ============================
# DATALOADERS
# ============================

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

# ============================
# CLASS WEIGHTS
# ============================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_base.targets),
    y=train_base.targets
)
cw = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=cw)

# ============================
# OPTIMIZER + SCHEDULER
# ============================

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

total_steps = len(train_loader) * num_epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

scaler = torch.cuda.amp.GradScaler()

best_val_acc = 0.0

# ============================
# TRAIN LOOP
# ============================

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for imgs_batch, labels in pbar:
        # imgs_batch shape: [B, 3, C, H, W]
        # Flatten into 3× batch:
        B, T, C, H, W = imgs_batch.shape
        imgs_batch = imgs_batch.view(B*T, C, H, W).to(device)
        labels = labels.unsqueeze(1).repeat(1, T).view(-1).to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(imgs_batch)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{running_loss/(pbar.n+1):.4f}"})

    # ============================
    # VALIDATION
    # ============================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(), torch.cuda.amp.autocast():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Save checkpoint
    torch.save({
        "epoch": epoch+1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_acc": val_acc
    }, checkpoint_path)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print("🔥 New best model saved!")

# ============================
# FINAL TEST
# ============================

model.load_state_dict(torch.load(best_model_path))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad(), torch.cuda.amp.autocast():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nFINAL TEST REPORT:")
print(classification_report(all_labels, all_preds,
                            target_names=test_dataset.classes))

