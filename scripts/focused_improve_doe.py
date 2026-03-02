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
import itertools
import pandas as pd

# ==============================
# SETUP
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
# FIXED BEST CONFIG
# ==============================

BATCH_SIZE = 16
DROPOUT = 0.3
OPTIMIZER_NAME = "RMSprop"
EPOCHS = 10   # safer than 20 for fine-tuning

# Focused factors
learning_rates = [0.001, 0.0005, 0.0001]
weight_decays = [0, 1e-4]

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
# RUN MINI DOE
# ==============================

for lr, wd in itertools.product(learning_rates, weight_decays):

    run_id = f"focused_{int(time.time())}_{lr}_{wd}"
    print(f"\nStarting Run: {run_id}")

    # Model
    model = models.mobilenet_v2(weights="DEFAULT")

    # Freeze all layers
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

    criterion = nn.BCELoss()

    optimizer = optim.RMSprop(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=wd
    )

    # ==============================
    # TRAINING
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

        # Validation
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
    # EVALUATION
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

    print("Accuracy:", accuracy)
    print("DPMO:", dpmo)

    # ==============================
    # SAVE MODEL
    # ==============================

    model_dir = os.path.join(BASE_DIR, "models", "focused")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{run_id}.pt")
    torch.save(model.state_dict(), model_path)

    # ==============================
    # LOG RESULTS
    # ==============================

    with open(CSV_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run_id,
            "focused_doe",
            lr,
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

    print("Run logged successfully.")

print("\nFocused Improve DOE Completed Successfully.")