import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4

# -------------------------
# Transforms
# -------------------------
train_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# Dataset
# -------------------------
train_dataset = datasets.ImageFolder(
    root="datasets/scene/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root="datasets/scene/val",
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print("Classes:", train_dataset.classes)

# -------------------------
# Model
# -------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(train_dataset.classes))
model.to(device)

# -------------------------
# Loss with class weights
# -------------------------
class_weights = torch.tensor([1.0, 1.2, 1.5, 1.0], device=device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Training
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.3f} | Val Acc: {acc:.2f}%")

# -------------------------
# Save model
# -------------------------
os.makedirs("scene/models", exist_ok=True)
torch.save(model.state_dict(), "scene/models/scene_model.pth")

print("✅ Scene model retrained and saved")
# -------------------------