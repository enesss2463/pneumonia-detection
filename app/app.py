import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/pneumonia_model.pth"

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model

model = load_model()
classes = ["NORMAL", "PNEUMONIA"]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


st.set_page_config(page_title="Pnömoni Tespiti", page_icon="🫁", layout="centered")

st.title("🫁 Göğüs Röntgeni Pnömoni Tespiti")
st.markdown("Bu uygulama, göğüs röntgeni görüntüsünü analiz ederek **pnömoni (zatürre)** olup olmadığını tespit eder.")
st.divider()

uploaded_file = st.file_uploader("Röntgen görüntüsü yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Röntgen", use_column_width=True)
    st.divider()

    with st.spinner("Analiz ediliyor..."):
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item() * 100

    label = classes[predicted_class]

    if label == "PNEUMONIA":
        st.error(f"⚠️ SONUÇ: PNÖMONİ TESPİT EDİLDİ")
        st.metric("Güven Oranı", f"%{confidence:.2f}")
        st.warning("⚕️ Lütfen bir sağlık uzmanına başvurunuz.")
    else:
        st.success(f"✅ SONUÇ: NORMAL")
        st.metric("Güven Oranı", f"%{confidence:.2f}")
        st.info("Röntgen görüntüsünde pnömoni belirtisi tespit edilmedi.")

    st.divider()
    st.subheader("📊 Olasılık Dağılımı")
    col1, col2 = st.columns(2)
    col1.metric("Normal", f"%{probabilities[0].item()*100:.2f}")
    col2.metric("Pnömoni", f"%{probabilities[1].item()*100:.2f}")