# Veri setinin hazırlanması ve Preprocessing

from keras.datasets import mnist # Mnist veri setini (Rakamlar veri seti) içe aktardım.
from keras.utils import to_categorical #(integer değerleri kategorik verilere ayırabilmek için import ettim)
from keras.callbacks import EarlyStopping, ModelCheckpoint # model eğitimini durdurma ve kaydetmek için içe aktardım.
from keras.models import Sequential # sıralı modeli içe aktardım.
from keras.layers import Dense # Bağlı katmanlar
from keras.models import load_model # Modeli kaydettikten sonra tekrar içe aktararak kullanabilmek için...
from keras.layers import Dense, BatchNormalization, Dropout

import matplotlib.pyplot as plt

# Mnist veri setini test ve train olacak şekilde yüklüyorum.
(x_train, y_train), (x_test, y_test) = mnist.load_data() 


#ilk birkac ornegi gorsellestir 
plt.figure(figsize=(10,5)) 
for i in range(6): 
    plt.subplot(2, 3, i+1) 
    plt.imshow(x_train[i], cmap="gray") 
    plt.title(f"index: {i}, Label: {y_train[i]}") 
    plt.axis("off") 
plt.show() 


#veri setinin normalize edelim, 0-255 araligindaki pixel degerlerini 0-1 arasina olceklendiriyoruz 
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2])).astype("float32")/255 
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2])).astype("float32")/255 


# etiketleri kategorik hale cevir (0-9 arasindaki rakamlari one-hot encoding yapiyoruz) 
y_train = to_categorical(y_train, 10) #10 sinif sayisi 
y_test = to_categorical(y_test, 10)       


# ANN modelinin oluşturulması ve derlenmesi
model = Sequential()   # Modeli oluşturdum ama şuan içi boş

# İlk katman 512 Cell yani nörondan oluşuyor, Relu aktivasyon fonksiyonu kullanacağım, input size 28*28 = 784 (mnsit veri setindekş fotoların boyutu böyle)
model.add(Dense(512, activation="relu", input_shape=(28*28,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# İkinci katman: 256 cell, activation: tanh  (relu, tanh a göre daha hızlı çalışır.)
model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Üçüncü katman
model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
# Dropout burada opsiyonel, istersen 0.1-0.2 eklersin

# output layer: 10 tane olmak zorunda çünkü 10 sınıfım var
# Activation fonksiyonu softmax olmak zorunda çünkü 2 den fazla sınıfım varsa bu fonksiyonu kullanmam lazım
model.add(Dense(10, activation="softmax"))


model.summary() # bu fonksiyon inşa ettiğimiz modeli gösteriyor özetliyor...

# NOT ÖNEMLİ
# 2 DEN FAZLA SINIF VARSA SOFTMAX FONKSİYONU KULLANMALIYIM.
# EĞER 2 TANE İSE SİGMOİD FONKSİYONUNU KULLANIRIM.
 

# Model derlemesi: optimizer (adam: buyuk veri ve kompleks aglar için idealdir.)
# Model derlemesi: loss (categorical_crossentropy)
# Model derlemesi: accuracy (Doğruluk, metrik)
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])


# Callbacks'lerin tanımlanması ve Training
# Erken durdurma: eğer validation_loss (val_loss) iyileşmiyorsa eğitimi durduralım.
# monitor: doğrulama setindeki (val) kaybı (loss) izler.
# Patience:5 ise ==> 5 epoch boyunca loss değeri iyileşmezse model eğitimi duracak.
# restore_best_weights=True ==> En iyi modelin ağırlıklarını geri yükler...
early_stopping = EarlyStopping(monitor = "val_loss", patience = 5, restore_best_weights=True) 


# Model checkpoint: En iyi modelin ağırlıklarını kaydeder.
# save_best_only=True: sadece en iyi performans gösteren modeli kaydeder.
checkpoint = ModelCheckpoint("ann_best_model.keras", monitor ="val_loss", save_best_only=True) 




# Model training: 10 epochs, batch size = 64, dogrulama seti orani = %20
history = model.fit(x_train, y_train, # train veri seti
          epochs=10,         # mdoel 10 kere veri setini görecek 10 devir eğitim yapılacak.
          batch_size=60,     # 60'erli parcalar ile egitim yapilacak.
          validation_split=0.2,  # Eğitim verilerinin %20 sini test için kullanılması
          callbacks=[early_stopping, checkpoint])

# model 60.000 veri setini her biri 60 parcadan olusan 60.000//60 = 1000 ama bunu %20 olarak test diye ayrıdık kalan 800 kerede train edecek ve biz buna 1 epoch diyeceğiz.
# yani düşünelim 60.000 değil de %20 yi ayırınca 48000 veri var 800 kerede train edecek...
# 60 * 800 = 48.000


# test verisi ile model performansi degerlendirme 
# evaluate: modelin test verisi uzerindeki loss (test_loss) ve accuracy (test_acc) hesaplar test_loss, test_acc model.evaluate(x_test, y_test) 
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test acc: {test_acc}, test Loss: {test_loss}") 
# training and validation accuracy gorsellestir 
plt.figure() 
plt.plot(history.history["accuracy"], marker = "o", label = "Training Accuracy") 
plt.plot(history.history["val_accuracy"], marker = "o", label = "Validation Accuracy") 
plt.title("ANN Accuracy on MNIST Data Set") 
plt.xlabel("Epochs") 
plt.ylabel("Acc") 
plt.legend() 
plt.grid(True) 
plt.show() 

#training and validation loss gorsellestirme 
plt.figure() 
plt.plot(history.history["loss"], marker = "o", label = "Training Loss") 
plt.plot(history.history["val_loss"], marker = "o", label = "Validation Loss") 
plt.title("ANN Loss on MNIST Data Set") 
plt.xlabel("Epochs") 
plt.ylabel("Loss") 
plt.legend() 
plt.grid(True) 
plt.show()


# Modeli yukarıdaki gibi (checkpoint = ModelCheckpoint("ann_best_model.keras")) şeklinde otomatik kaydettik zaten 
# Kaydetme yöntemlerinden biri;
#model.save("final_mnist_ann_model.keras")


# Modelleri geri yükleme 
#loaded_model = load_model("final_mnist_ann_model.keras")
#print(f"Loaded Model Result --> Test acc: {test_acc}, test Loss: {test_loss}") 








