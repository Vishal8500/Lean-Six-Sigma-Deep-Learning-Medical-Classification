import os
import csv
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ==============================
# 2️⃣ HYPERPARAMETERS
# ==============================

LR = 0.001
BATCH_SIZE = 16
DROPOUT = 0.5
EPOCHS = 10
OPTIMIZER_NAME = "Adam"
RUN_ID = f"baseline_{int(time.time())}"

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


# Get project root directory (one level above scripts folder)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(BASE_DIR, "data")

print("Base Directory:", BASE_DIR)
print("Data Directory:", DATA_DIR)

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ==============================
# 4️⃣ MODEL
# ==============================

model = models.mobilenet_v2(pretrained=True)

for param in model.features.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(1280, 128),
    nn.ReLU(),
    nn.Dropout(DROPOUT),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

model = model.to(device)

# ==============================
# 5️⃣ OPTIMIZER & LOSS
# ==============================

criterion = nn.BCELoss()

if OPTIMIZER_NAME == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=LR)
else:
    optimizer = optim.RMSprop(model.parameters(), lr=LR)

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

    # Validation Loss
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

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

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
cm = confusion_matrix(y_true, y_pred)

defects = sum(np.array(y_true) != np.array(y_pred))
total = len(y_true)
dpmo = (defects / total) * 1_000_000

print("\n===== BASELINE RESULTS =====")
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Defects:", defects)
print("DPMO:", dpmo)
print("Training Time (sec):", training_time)
print("Confusion Matrix:\n", cm)

# ==============================
# 8️⃣ SAVE MODEL
# ==============================

os.makedirs("models/baseline", exist_ok=True)
model_path = f"models/baseline/{RUN_ID}.pt"
torch.save(model.state_dict(), model_path)

# ==============================
# 9️⃣ SAVE TO CSV
# ==============================

os.makedirs("logs", exist_ok=True)
csv_path = "logs/results.csv"
file_exists = os.path.isfile(csv_path)

with open(csv_path, mode='a', newline='') as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "run_id",
            "run_type",
            "lr",
            "batch_size",
            "dropout",
            "optimizer",
            "epochs",
            "train_loss",
            "val_loss",
            "accuracy",
            "f1",
            "defects",
            "dpmo",
            "training_time_sec",
            "model_path"
        ])

    writer.writerow([
        RUN_ID,
        "baseline",
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

print("Model saved at:", model_path)
print("Results logged to CSV.")