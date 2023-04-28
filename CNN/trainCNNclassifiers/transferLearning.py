from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Conv2D, Input, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np 
import cv2 
import time

def get_transfer_model():
    # VGG 16 modelini imagenet ağırlıkları ile oku
    # Transfer learning için modelin son FC katmanını kaldır.
    baseModel = VGG16(weights='imagenet',include_top=False, input_shape=(224, 224, 3))
    
    baseModel.summary()
    # VGG16 ile belirlenen taban modele ek katmanlar oluşturup model özelleştirilir.
    headModel = baseModel.output
    # KODU YAZ

    # Örnek bazı katman eklemeleri. 
    headModel=Conv2D(filters=1024, kernel_size=(3,3), use_bias=True, strides=1,
               activation='relu', padding="same")(headModel)
    headModel = MaxPooling2D(pool_size=(2, 2))(headModel) 
    # headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(20, activation="softmax")(headModel)

    # Taban model ile ana model birleştirilir. 
    model = Model(inputs=baseModel.input, outputs=headModel)
    
    # Taban modelin katmanlarını dondurup ağırlıklarının eğitilmesini önle
    # VGG16 zaten ayırtedici özellikler öğrenmek üzere eğitilmiş bir ağdır.
    for layer in baseModel.layers:
        layer.trainable = False

    # Modeli özetle
    model.summary()

    return model

def preprocess_img(img):
    # apply opencv preprocessing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,  (224, 224)) 
    img = np.asarray(img, dtype=np.float)
    return img

def load_data():
    dataset = './lecture12/all'
    # Veri seti dizinindeki tüm görüntü dosyalarını yükle
    print("[INFO] veri yukleniyor...")
    # Görüntü dizinlerini oku
    imagePaths = list(paths.list_images(dataset))
    data = []
    labels = []
    # Tüm RAW görüntü türlerini oku
    for imagePath in imagePaths:
        # RAW dosya ismi ise görüntüyü okuyup data içerisine ekle
        img_type = imagePath.split('\\')[-2]
        # Dosya isminden etiketi belirle
        label = imagePath.split('\\')[-3]
        if img_type == 'RAW':
            # Görüntüyü oku ve önişle
            img = cv2.imread(imagePath)
            image = preprocess_img(img)       
            image = preprocess_input(image)
            # update the data and labels lists, respectively
            data.append(image)
            labels.append(label)

    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    
    # Kategorik etiketleri dönüştür
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    #labels = to_categorical(labels)
    
    # veriyi %75 eğitim %25 test olmak üzere ayır.
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)
   
    return trainX, testX, trainY, testY, lb

def train_model(model, trainX, trainY, testX, testY):
    # Modeli derle
    print("[INFO] model derleniyor...")
    INIT_LR = 1e-4
    EPOCHS = 20
    BS = 32
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    print("Derlendi !!")
    # Veriyi çeşitlendirmek (data augmentation) için aşağıdaki fonksiyon çağrılabilir
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    # Modelin sonradan eklenen ana kısmını eğit
    print("[INFO] Ana model egitiliyor...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)
    # Model eğitim kaybı ve başarısını çizdir
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('evaluation.png')

    return model

def evaluate_model(model, testX, testY, lb):
    BS = 32
    start = time.time()
    # Test kümesi üzerindeki tahminleri belirle
    print("[INFO] model degerlendiriliyor...")
    predIdxs = model.predict(testX, batch_size=BS)
    print(time.time() - start)

    # En yüksek olasılığa denk gelen etiketin indeksini 
    # tüm test görüntüleri için belirle
    predIdxs = np.argmax(predIdxs, axis=1)
    realIdxs = testY.argmax(axis=1)
    # Gerçek sınıf etiketleri ve tahmin edilen etiketleri karşılaştır
    # ve düzgün bir formatta raporla

    print(classification_report(realIdxs, predIdxs, target_names=lb.classes_))
    print(confusion_matrix(realIdxs,predIdxs))

# Veriyi yükle ve eğitim ve test olarak ayır.
trainX, testX, trainY, testY, lb = load_data()

# Modeli oluştur ve eğit
# A şıkkı için tamamlanması gereken model
cnn_model = get_transfer_model()
# B şıkkı için oluşturulması gereken model
# cnn_model = get_your_model()
cnn_model = train_model(cnn_model, trainX, trainY, testX, testY)
# Modeli değerlendir
# cnn_model = load_model("kitchen_equipments")
cnn_model.summary()
evaluate_model(cnn_model, testX, testY, lb)

# Modeli diske kaydet
print("[INFO] Mutfak araç ve gereçleri modelini diske kaydet...")
cnn_model.save('kitchen_equipments')
print("Bitti !")



