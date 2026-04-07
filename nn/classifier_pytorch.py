"""
PyTorch implementation of the simple MNIST classifier that was
originally written with Keras/TensorFlow.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ---------------------------
# 1.  Hyper‑parameters
# ---------------------------
BATCH_SIZE   = 128
EPOCHS       = 5
LR           = 0.01         # Keras SGD default
VAL_SPLIT    = 0.10
NUM_CLASSES  = 10
IMAGE_SIZE   = 28 * 28      # 784

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 2.  Dataset & Dataloaders
# ---------------------------
#  -> ToTensor() gives [0,1] float32 tensor of shape (1,28,28)
#  -> Lambda flattens to (784,)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))   # flatten
])

train_full = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
test_set   = datasets.MNIST(root="data", train=False, download=True, transform=transform)

val_len   = int(len(train_full) * VAL_SPLIT)
train_len = len(train_full) - val_len
train_set, val_set = random_split(train_full, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE)

# ---------------------------
# 3.  Model definition
# ---------------------------
model = nn.Sequential(
    nn.Linear(IMAGE_SIZE, 32),
    # choose ONE of these …
    # nn.ReLU(),          # piece‑wise linear (most common)
    # nn.Tanh(),        # bounded, zero‑centred
    # nn.LeakyReLU(),   # “fixed” ReLU with small negative slope
    # nn.ELU(),         # smooth ReLU variant
    # nn.GELU(),        # popular in transformers
    nn.Sigmoid(),     # original choice (bounded, not zero‑centred)
    nn.Linear(32, NUM_CLASSES)   # CrossEntropyLoss adds softmax internally
).to(device)

# ---------------------------
# 4.  Loss & Optimizer
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01,
                      momentum=0.9, nesterov=True)

# ---------------------------
# 5.  Training loop
# ---------------------------
history = {"train_acc": [], "val_acc": []}

for epoch in range(EPOCHS):
    # ---- train ----
    model.train()
    correct, total = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    train_acc = correct / total

    # ---- validate ----
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    val_acc = correct / total

    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    print(f"Epoch {epoch+1}/{EPOCHS}  —  train acc: {train_acc:.3f}  |  val acc: {val_acc:.3f}")

# ---------------------------
# 6.  Test‑set evaluation
# ---------------------------
model.eval()
with torch.no_grad():
    correct, total = 0, 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
test_acc = correct / total
print(f"\nTest accuracy: {test_acc:.3f}")

# ---------------------------
# 7.  Accuracy curves
# ---------------------------
plt.plot(history["train_acc"], label="training")
plt.plot(history["val_acc"],   label="validation")
plt.title("Model accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()

# ---------------------------
# 8.  Quick prediction demo
# ---------------------------
def show_prediction(idx=0):
    """Visualise one test image and the network’s prediction."""
    img, label = test_set[idx]
    with torch.no_grad():
        pred = model(img.to(device)).argmax().item()

    plt.imshow(img.view(28, 28).cpu(), cmap="gray")
    plt.title(f"True label: {label} — Predicted: {pred}")
    plt.axis("off")
    plt.show()

show_prediction(idx=1)
