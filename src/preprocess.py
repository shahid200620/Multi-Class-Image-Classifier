import os
import random
from torchvision.datasets import CIFAR10
from PIL import Image

root = "data"
train_dir = os.path.join(root, "train")
val_dir = os.path.join(root, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

dataset = CIFAR10(root="data", train=True, download=True)

classes = dataset.classes

for c in classes:
    os.makedirs(os.path.join(train_dir, c), exist_ok=True)
    os.makedirs(os.path.join(val_dir, c), exist_ok=True)

data = list(zip(dataset.data, dataset.targets))
random.shuffle(data)

split = int(0.8 * len(data))
train_data = data[:split]
val_data = data[split:]

for idx, (img, label) in enumerate(train_data):
    cls = classes[label]
    im = Image.fromarray(img)
    im.save(os.path.join(train_dir, cls, f"{idx}.png"))

for idx, (img, label) in enumerate(val_data):
    cls = classes[label]
    im = Image.fromarray(img)
    im.save(os.path.join(val_dir, cls, f"{idx}.png"))

print("done")