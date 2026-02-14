import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

device = "cpu"

train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder("data/train", transform=train_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

model = models.mobilenet_v2(weights=None)

num_classes = len(train_data.classes)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    break

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/image_classifier.pth")

print("model saved")