# Rakam Sınıflandırma (MNIST) — ANN

Bu proje, **MNIST** veri setindeki el yazısı rakamları (0–9) **Yapay Sinir Ağı (ANN)** ile sınıflandırır. Eğitim sırasında **EarlyStopping** ve **ModelCheckpoint** kullanılır, en iyi model `ann_best_model.keras` olarak kaydedilir ve eğitim/validasyon eğrileri görselleştirilir.  

---

## Kurulum

```bash
# Bağımlılıklar
pip install -r requirements.txt
```

---

## Kullanım

```bash
python Rakam_sınıflandırma_ANN.py
```

* Eğitim sırasında **en iyi ağırlıklar** `ann_best_model.keras` dosyasına kaydedilir. 
* Konsolda **test doğruluğu ve kaybı** yazdırılır. 
* **Eğitim/validasyon doğruluk ve kayıp grafikleri** ekranda gösterilir. 

**İsteğe bağlı:**

* Nihai modeli ayrıca kaydetmek isterseniz ilgili satırı açabilirsiniz (`model.save("final_mnist_ann_model.keras")`). 
* Kaydedilmiş modeli tekrar yükleme örneği de dosyada bulunur (`load_model(...)`). 
