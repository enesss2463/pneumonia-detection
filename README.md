#  Göğüs Röntgeni ile Pnömoni Tespiti

Derin öğrenme kullanarak göğüs röntgeni görüntülerinden pnömoni (zatürre) tespiti yapan bir yapay zeka uygulaması.

---

##  Proje Hakkında

Bu proje, ResNet18 derin öğrenme modeli kullanarak göğüs röntgeni görüntülerini analiz eder ve **Normal** ya da **Pnömoni** olarak sınıflandırır. Streamlit ile geliştirilen web arayüzü sayesinde kullanıcılar röntgen görüntüsü yükleyerek anında sonuç alabilir.

---

##  Model Performansı

| Metrik | Değer |
|--------|-------|
| Accuracy | %86 |
| Precision (PNEUMONIA) | %82 |
| Recall (PNEUMONIA) | %100 |
| F1-Score (PNEUMONIA) | %90 |

---

##  Veri Seti

- **Kaynak:** [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Toplam Görüntü:** 5.863
- **Sınıflar:** Normal, Pneumonia
- **Bölme:** %80 Eğitim / %20 Test

---

##  Kullanılan Teknolojiler

- Python 3.x
- PyTorch & TorchVision
- ResNet18 (Transfer Learning)
- Streamlit
- OpenCV
- Scikit-learn
- Matplotlib & Seaborn

---

##  Kurulum ve Çalıştırma

### 1. Repoyu klonla
```bash
git clone https://github.com/enesss2463/pneumonia-detection.git
cd pneumonia-detection
```

### 2. Sanal ortam oluştur
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Kütüphaneleri kur
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install streamlit pillow matplotlib scikit-learn seaborn
```

### 4. Modeli eğit
```bash
python train.py
```

### 5. Uygulamayı başlat
```bash
streamlit run app/app.py
```