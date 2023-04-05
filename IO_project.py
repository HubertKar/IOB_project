
import tensorflow as tf
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

def import_images(path, max=7000, color = "RGB"):
    images_set = []
    n = 0 
    for images in os.listdir(path):
       if n < max:
        image = cv.imread(path+'/'+images,1)

        if color == "RGB":
           image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        elif color == "HSV":
           image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        elif color == "YUV":
           image = cv.cvtColor(image, cv.COLOR_BGR2YUV)
        
        images_set.append(tf.convert_to_tensor(image))
        n += 1
    return images_set

def import_labels(path, max=7000):
    images_set = []
    n = 0 
    for images in os.listdir(path):
       if n < max:
        image = cv.imread(path+'/'+images, 0)
        images_set.append(tf.convert_to_tensor(image))
        n += 1
    return images_set

def proces_images(image):
    image = tf.cast(image, tf.float32) #konwersja i normalizacja
    image = image/255.0
    return image

def proces_labels(label):          
    label = tf.expand_dims(label, axis=-1) 
    label = tf.cast(label, tf.bool)
    return label
  
def crate_U_model(img_height, img_width,img_channels, num_classes = 1, show_summary = False): # Tworzenie modelu siecu konwolucyjnej (U-net)

    #Architektur sieci nazrazie zajebana z internertu bo robienie sieci U-net jest ciężkie
    inputs = tf.keras.layers.Input((img_height, img_width, img_channels)) # Input layer, piewrwsza wartswa przyjmuje obraz o wymiarach (img_height, img_width, img_channels)

    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs) #Pierwsza warstwa konwolucyjna
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    b1 = tf.keras.layers.BatchNormalization()(c1)
    r1 = tf.keras.layers.ReLU()(b1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(r1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    b2 = tf.keras.layers.BatchNormalization()(c2)
    r2 = tf.keras.layers.ReLU()(b2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(r2)
    
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    b3 = tf.keras.layers.BatchNormalization()(c3)
    r3 = tf.keras.layers.ReLU()(b3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(r3)
    
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    b4 = tf.keras.layers.BatchNormalization()(c4)
    r4 = tf.keras.layers.ReLU()(b4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(r4)
    
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    b5 = tf.keras.layers.BatchNormalization()(c5)
    r5 = tf.keras.layers.ReLU()(b5)
    c5 = tf.keras.layers.Dropout(0.3)(r5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.BatchNormalization()(u6)
    u6 = tf.keras.layers.ReLU()(u6)

    
    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.BatchNormalization()(u7)
    u7 = tf.keras.layers.ReLU()(u7)

    
    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.BatchNormalization()(u8)
    u8 = tf.keras.layers.ReLU()(u8)
    
    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(u8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    u9 = tf.keras.layers.BatchNormalization()(u9)
    u9 = tf.keras.layers.ReLU()(u9)
    
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(u9)

    u_net_model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    u_net_model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy']) # Konfiugracja modelu

    if show_summary == True: # Wyświetl podsumowanie inforamcji o sieci
        u_net_model.summary()

    return u_net_model

# Definicja zmiennych
img_width = 256
img_height = 256
img_channels = 3
batch_size = 32
EPOCHS = 20
model_name = "HSV"
color_model = "RGB"

# Import zdjeć i masek (labels)
with tf.device('/CPU:0'): # Wykonuj obliczenia na CPU i przechowuj zmienne W RAM nie w VRAM, domyślnie wszystkoz z tf robione jest na GPU
    train_images = import_images("images/train/img", color=color_model) # Dane do uczenia 
    train_labels = import_labels("labels/train/img")

    valid_images = import_images("images/valid/img",color=color_model) # Dane do tesotwania działania sieci
    valid_labels = import_labels("labels/valid/img")

    test_images = import_images("images/test/img",color=color_model) # Dane do tesotowania
    test_labels = import_labels("labels/test/img")

    # Skalowanie i konwersja
    train_images = [proces_images(i) for i in train_images] # Dla każdego elemntu z train_images wykonaj funckcje proces_image()
    train_labels = [proces_labels(l) for l in train_labels]

    valid_images = [proces_images(i) for i in valid_images]
    valid_labels = [proces_labels(l) for l in valid_labels]

    test_images = [proces_images(i) for i in test_images]
    test_labels = [proces_images(i) for i in test_labels]

    # Konwersja na tf.Dataset
    train_X = tf.data.Dataset.from_tensor_slices(train_images)
    train_Y = tf.data.Dataset.from_tensor_slices(train_labels)

    valid_X = tf.data.Dataset.from_tensor_slices(valid_images)
    valid_Y = tf.data.Dataset.from_tensor_slices(valid_labels)

    test_X = tf.data.Dataset.from_tensor_slices(test_images)
    test_Y = tf.data.Dataset.from_tensor_slices(test_labels)

train_set = tf.data.Dataset.zip((train_X, train_Y)) #Lączenie datasetów, połączenie zdjęć z maskami
valid_set = tf.data.Dataset.zip((valid_X, valid_Y))
test_set  = tf.data.Dataset.zip((test_X, test_Y))

#### Podzial na batche
AT = tf.data.AUTOTUNE ## sprwadzić
STEPS_PER_EPOCH = len(train_images)//batch_size # Kroki na epoke (liczba_zdjęć podzielona przez batch_size)
VALIDATION_STEPS = len(valid_images)//batch_size

Buffer = len(train_images) # Bufor do losowania danych
train_set = train_set.cache().shuffle(Buffer).batch(batch_size).repeat() # Podziel dane na batche i przemieszaj
train_set = train_set.prefetch(buffer_size=AT) # To zwikęsza wydajność kosztem zużycia pamięci

valid_set = valid_set.batch(batch_size) 
test_set = test_set.batch(batch_size)

input = input("Wybierz model do wczytania, 0 - stórz nowy model:  ")

u_net_model = crate_U_model(img_height, img_width, img_channels,num_classes=1, show_summary=True) # Tworzenie modelu

if input == '0':
    history = u_net_model.fit(train_set, validation_data = valid_set,steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, epochs = EPOCHS) # Uczenia modelu
    loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']

    plt.figure() # Wykres zmian "loss" (chyba jako koszt sie to tłumaczy) przy kolejnych epokach
    plt.plot(loss_history, 'b', label="Training loss")
    plt.plot(val_loss_history, 'ro', label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(model_name+".png")
    plt.show()

else:
   u_net_model.load_weights("saved_model/" + input + ".keras") # Załaduj wagi do warst modelu z pliku

u_net_model.save("saved_model/" + model_name + ".keras") # Zapisz model

print("Dokładność dla zbioru testowego")
u_net_model.evaluate(test_set)

while True: 
    i = np.random.randint(0,len(test_images)-1) # Losowa zdjęcie i masska ze zbioru testowego
    sample_image = test_images[i] 
    sample_mask = test_labels[i]

    prediction = u_net_model.predict(sample_image[tf.newaxis, ...])[0] # Przewidywanie maski, tf.newaxis powiększa wymiar macierzy o jeden, predict oczukuje zbioru danych  

    plt.figure(figsize=(10,5))
    plt.subplot(131)
    plt.imshow(sample_image)
    plt.title("Obraz wejściowy")

    plt.subplot(132)
    predicted_mask = (prediction > 0.5).astype(np.uint8) # 0.5 próg powyżej którego pikel trakowany jest jako drzewo
    plt.imshow(tf.keras.utils.array_to_img(predicted_mask))
    plt.axis('off')
    plt.title('Przewidywana maska')

    plt.subplot(133)
    plt.imshow(sample_mask)
    plt.axis('off')
    plt.title('Poprawna maska')
    plt.show()
