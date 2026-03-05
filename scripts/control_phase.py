import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd

# ==============================
# CONFIG
# ==============================

LR = 0.0001
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
DROPOUT = 0.3
EPOCHS = 15
PATIENCE = 3
NUM_RUNS = 3

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(BASE_DIR, "logs", "results.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# DATA
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
# CONTROL RUNS
# ==============================

results = []

for run in range(NUM_RUNS):

    print(f"\n===== CONTROL RUN {run+1} =====")

    seed = 42 + run
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = models.mobilenet_v2(weights="DEFAULT")

    for param in model.features.parameters():
        param.requires_grad = False

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

    criterion = nn.BCELoss()

    optimizer = optim.RMSprop(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):

        print(f"\n===== Run {run+1} | Epoch {epoch+1}/{EPOCHS} =====")

        model.train()
        train_loss = 0

        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                images = images.to(device)
                labels = labels.float().to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    # Test Evaluation
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
    defects = sum(np.array(y_true) != np.array(y_pred))
    dpmo = (defects / len(y_true)) * 1_000_000

    print("Accuracy:", accuracy)
    print("DPMO:", dpmo)

    results.append(accuracy)

# ==============================
# CONTROL STATISTICS
# ==============================

mean_acc = np.mean(results)
std_acc = np.std(results)

UCL = mean_acc + 3 * std_acc
LCL = mean_acc - 3 * std_acc

print("\n===== CONTROL CHART METRICS =====")
print("Accuracies:", results)
print("Mean Accuracy:", mean_acc)
print("Std Dev:", std_acc)
print("Upper Control Limit:", UCL)
print("Lower Control Limit:", LCL)

if all(LCL <= x <= UCL for x in results):
    print("\nProcess is statistically stable.")
else:
    print("\nProcess shows instability.")