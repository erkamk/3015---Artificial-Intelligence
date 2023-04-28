import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

file_train = r"C:\Users\asus\Desktop\odev4\training-spectrograms"
file_test = r"C:\Users\asus\Desktop\odev4\testing-spectrograms"
labels_train_number = []
labels_train_person = []
training_data = []
person_dict = {'george': 0, 'jackson': 1, 'lucas': 2, 'nicolas': 3, 'theo': 4, 'yweweler':5}

def load_data():
    # veriler bilgisayarın dizinine göre yüklendi.
    for file_name in os.listdir(file_train):
        labels_train_number.append(file_name[0])
    
    for file_name in os.listdir(file_test):
        labels_train_number.append(file_name[0])
    
    for file_name in os.listdir(file_train):
        splt = file_name.split(sep = "_")
        labels_train_person.append(splt[1])
    
    for file_name in os.listdir(file_test):
        splt = file_name.split(sep = "_")
        labels_train_person.append(splt[1])
        
    for img in os.listdir(file_train):
        try:
            path = os.path.join(file_train + "\\",img)
            img_array = cv2.imread(path, cv2.COLOR_BGR2RGB)
            new_array = cv2.resize(img_array,(224,224)) 
            training_data.append(new_array)
        except Exception as e:
            pass
        
    for img in os.listdir(file_test):
        try:
            path = os.path.join(file_test + "\\",img)
            img_array = cv2.imread(path, cv2.COLOR_BGR2RGB)
            new_array = cv2.resize(img_array,(224,224)) 
            training_data.append(new_array)
        except Exception as e:
            pass
        
    for i in range(len(labels_train_person)):
        labels_train_person[i] = person_dict[labels_train_person[i]]
        

load_data()
#%%
from keras.utils.np_utils import to_categorical 
# categorical veriye cevrim.
import numpy as np
y_number = to_categorical(labels_train_number, num_classes = 10)
y_person = to_categorical(labels_train_person, num_classes = 6)
y_number = np.array(y_number)
y_person = np.array(y_person)
#%%
import numpy as np
#normalizasyon
training_data = np.array(training_data).reshape(-1,224,224)
training_data = training_data/255.0 
training_data= training_data.reshape(-1,224,224,3)


#%%
from keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
loss_train1 = 0
loss_val1 = 0

def get_transfer_model_number():
    
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False 
    
    flatten_layer = layers.Flatten()
    dense_layer_1 = layers.Dense(48, activation='relu')
    dense_layer_2 = layers.Dense(32, activation='relu')
    prediction_layer = layers.Dense(10, activation='softmax')
    
    
    model1 = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])
    model1.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )
    
    
    model1.summary()

    history1 = model1.fit(training_data, y_number, epochs=10, validation_split=0.2, batch_size=12,shuffle = True)
    
    global loss_train1,loss_val1 
    loss_train1 = history1.history['loss']
    loss_val1 = history1.history['val_loss']
    
get_transfer_model_number()
#%%
loss_train2 = 0
loss_val2 = 0
def get_transfer_model_person():

    
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False 
    flatten_layer = layers.Flatten()
    dense_layer_1 = layers.Dense(48, activation='relu')
    dense_layer_2 = layers.Dense(32, activation='relu')
    prediction_layer = layers.Dense(6, activation='softmax')
    
    
    model2 = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])
    model2.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )
    
    
    model2.summary()
    history2 = model2.fit(training_data, y_person, epochs=10, validation_split=0.2, batch_size=12,shuffle = True)
    
    global loss_train2,loss_val2 
    loss_train2 = history2.history['loss']
    loss_val2 = history2.history['val_loss']
    
    
get_transfer_model_person()
#%%
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
loss_train3 =0
loss_val3 = 0
def get_your_model_number():
    model3 = Sequential()
    #
    model3.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same', 
                     activation ='relu', input_shape = (224,224,3)))
    model3.add(MaxPool2D(pool_size=(2,2)))
    model3.add(Dropout(0.25))
    
    #
    model3.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model3.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model3.add(Dropout(0.3))
    
    #
    model3.add(Conv2D(filters = 64, kernel_size = (2,2),padding = 'Same', 
                     activation ='relu'))
    model3.add(Dropout(0.3))
    
    # 
    model3.add(Flatten())
    model3.add(Dense(48, activation = "relu"))
    model3.add(Dropout(0.25))
    #
    model3.add(Dense(32, activation = "relu"))
    
    
    model3.add(Dense(10, activation = "softmax"))
    
    model3.compile(optimizer = 'Adam' , loss = 'categorical_crossentropy', metrics=["accuracy"])
    model3.summary()

    history3 = model3.fit(training_data, y_number, epochs=30, validation_split=0.2, batch_size=12,shuffle = True)
    global loss_train3, loss_val3
    loss_train3 = history3.history['loss']
    loss_val3 = history3.history['val_loss']
        
get_your_model_number()


#%%
loss_train4 = 0
loss_val4 = 0
def get_your_model_person():
    model4 = Sequential()
    #
    model4.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same', 
                     activation ='relu', input_shape = (224,224,3)))
    model4.add(MaxPool2D(pool_size=(2,2)))
    model4.add(Dropout(0.25))
    
    #
    model4.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model4.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model4.add(Dropout(0.3))
    
    #
    model4.add(Conv2D(filters = 64, kernel_size = (2,2),padding = 'Same', 
                     activation ='relu'))
    model4.add(Dropout(0.3))
    
    # 
    model4.add(Flatten())
    model4.add(Dense(48, activation = "relu"))
    model4.add(Dropout(0.25))
    #
    model4.add(Dense(32, activation = "relu"))
    
    
    model4.add(Dense(6, activation = "softmax"))
    
    model4.compile(optimizer = 'Adam' , loss = 'categorical_crossentropy', metrics=["accuracy"])
    model4.summary()

    history4 = model4.fit(training_data, y_person, epochs=30, validation_split=0.2, batch_size=12,shuffle = True)
    global loss_train4, loss_val4
    loss_train4 = history4.history['loss']
    loss_val4 = history4.history['val_loss']    

get_your_model_person()

#%%


