import os
import csv
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ==============================
# 1️⃣ SETUP
# ==============================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(BASE_DIR, "logs", "results.csv")

# ==============================
# 2️⃣ BEST CONFIGURATION
# ==============================

LR = 0.0001
BATCH_SIZE = 16
DROPOUT = 0.3
OPTIMIZER_NAME = "RMSprop"
EPOCHS = 9
RUN_ID = f"improve_{int(time.time())}"

# ==============================
# 3️⃣ DATA
# ==============================

IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ==============================
# 4️⃣ MODEL (Partial Unfreeze)
# ==============================

model = models.mobilenet_v2(pretrained=True)

# Freeze all layers first
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze last two blocks
for param in model.features[-2:].parameters():
    param.requires_grad = True

model.classifier = nn.Sequential(
    nn.Linear(1280, 128),
    nn.ReLU(),
    nn.Dropout(DROPOUT),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

model = model.to(device)

# ==============================
# 5️⃣ LOSS & OPTIMIZER
# ==============================

criterion = nn.BCELoss()

optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# ==============================
# 6️⃣ TRAINING
# ==============================

start_time = time.time()

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

training_time = time.time() - start_time

# ==============================
# 7️⃣ EVALUATION
# ==============================

model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images).squeeze().cpu()
        preds = (outputs > 0.5).int()

        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
defects = sum(np.array(y_true) != np.array(y_pred))
total = len(y_true)
dpmo = (defects / total) * 1_000_000

print("\n===== IMPROVED RESULTS =====")
print("Accuracy:", accuracy)
print("F1:", f1)
print("Defects:", defects)
print("DPMO:", dpmo)
print("Training Time:", training_time)

# ==============================
# 8️⃣ SAVE MODEL
# ==============================

model_dir = os.path.join(BASE_DIR, "models", "best")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, f"{RUN_ID}.pt")
torch.save(model.state_dict(), model_path)

# ==============================
# 9️⃣ COMPARE WITH BASELINE
# ==============================

df = None
if os.path.exists(CSV_PATH):
    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    baseline_dpmo = df[df["run_type"]=="baseline"]["dpmo"].values[0]
    reduction = baseline_dpmo - dpmo
    percent = (reduction / baseline_dpmo) * 100

    print("\n===== DEFECT REDUCTION =====")
    print("Baseline DPMO:", baseline_dpmo)
    print("New DPMO:", dpmo)
    print("Reduction:", reduction)
    print("Reduction %:", percent)

# ==============================
# 🔟 LOG TO CSV
# ==============================

with open(CSV_PATH, mode='a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        RUN_ID,
        "improve",
        LR,
        BATCH_SIZE,
        DROPOUT,
        OPTIMIZER_NAME,
        EPOCHS,
        train_loss,
        val_loss,
        accuracy,
        f1,
        defects,
        dpmo,
        training_time,
        model_path
    ])

print("Improve phase completed and logged.")