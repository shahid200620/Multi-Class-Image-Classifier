import io
import os
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms, models
from PIL import Image

app = FastAPI()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

classes = sorted(os.listdir("data/train"))

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))
model.load_state_dict(torch.load("model/image_classifier.pth", map_location="cpu"))
model.eval()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        return {"error": "no file"}

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        return {"error": "invalid image"}

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return {
        "predicted_class": classes[pred.item()],
        "confidence": float(conf.item())
    }