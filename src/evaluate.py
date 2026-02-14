import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

device = "cpu"

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

val_data = datasets.ImageFolder("data/val", transform=transform)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=16)

model = models.mobilenet_v2(weights=None)
num_classes = len(val_data.classes)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load("model/image_classifier.pth", map_location=device))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
cm = confusion_matrix(y_true, y_pred).tolist()

os.makedirs("results", exist_ok=True)

metrics = {
    "accuracy": float(acc),
    "precision_weighted": float(prec),
    "recall_weighted": float(rec),
    "confusion_matrix": cm
}

with open("results/metrics.json", "w") as f:
    json.dump(metrics, f)

print("metrics saved")