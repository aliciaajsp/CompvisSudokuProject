import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# config
BATCH_SIZE = 128
EPOCHS = 20
PATIENCE = 4
LR = 0.001
TEST_SPLIT = 0.1
RANDOM_STATE = 42
EPSILON = 1e-7
MODEL_PATH = "digit_cnn_tmnist.pt"
NORM_PATH = "norm_constants_tmnist.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load & clean data
df = pd.read_csv("TMNIST_Data.csv")
df = df.dropna()

y_all = df["labels"].values.astype("int64")
x_all = df.drop(columns=["names", "labels"]).values.astype("float32")

nan_x = np.isnan(x_all).any(axis=1)
nan_y = np.isnan(y_all.astype("float32"))
if nan_x.any() or nan_y.any():
    valid_mask = ~nan_x & ~nan_y
    x_all = x_all[valid_mask]
    y_all = y_all[valid_mask]

# initial reshape & scale to (0.0 - 1.0)
x_all = x_all.reshape(-1, 28, 28) / 255.0

# split dataset (done before normalization to prevent data leakage)
x_train, x_test, y_train, y_test = train_test_split(
    x_all, y_all, test_size=TEST_SPLIT, random_state=RANDOM_STATE, stratify=y_all
)

# calculate normalization constants strictly from train set
TRAIN_MEAN = float(np.mean(x_train))
TRAIN_STD = float(np.std(x_train))

# apply training constants to both splits
x_train = (x_train - TRAIN_MEAN) / (TRAIN_STD + EPSILON)
x_test = (x_test - TRAIN_MEAN) / (TRAIN_STD + EPSILON)

# reshape to pytorch format: [Batch, Channel, Height, Width]
x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)

# post-normalization nan protection
if np.isnan(x_train).any(): x_train = np.nan_to_num(x_train, nan=0.0)
if np.isnan(x_test).any(): x_test = np.nan_to_num(x_test, nan=0.0)

# pytorch dataloaders
train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# model definition
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

# init model & optimizer
model = DigitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)

best_val_loss = float('inf')
patience_counter = 0

# train
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total

    # validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / val_total
    epoch_val_acc = val_correct / val_total

    print(f"[{epoch+1:02d}/{EPOCHS}] Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")

    scheduler.step(epoch_val_loss)

    # early stopping & save model
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# save normalization constants (using training constants)
np.save(NORM_PATH, np.array([TRAIN_MEAN, TRAIN_STD]))

# test final performance
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

final_acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
print(f"Final Test Accuracy: {final_acc:.2f}%")