import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


DATA_DIR  = "chest_xray/chest_xray"
MODEL_DIR = "models"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transforms)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
classes      = test_dataset.classes
print(f"Sınıflar: {classes}")


model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "pneumonia_model.pth"), map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model yüklendi!")


all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)


print("\n--- SINIFLANDIRMA RAPORU ---")
print(classification_report(all_labels, all_preds, target_names=classes))


cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
plt.show()
print("Confusion matrix kaydedildi!")