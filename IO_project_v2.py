
import tensorflow as tf
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time
import gc

def import_images(path, max=-1, color = "RGB"):
    images_set = []
    n = 0 
    for images in os.listdir(path):
       if n < max or max == -1:
        image = cv.imread(path+'/'+images,1)

        if color == "RGB":
           image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        elif color == "HSV":
           image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        elif color == "YUV":
           image = cv.cvtColor(image, cv.COLOR_BGR2YUV)
        elif color == "GRAY":
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        elif color == "RGB+H":
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image_r, image_g, image_b = cv.split(image)
            image_HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            image_H, image_S, image_V = cv.split(image_HSV)
            image = cv.merge((image_r, image_g, image_b, image_H))
        elif color == "RGB+HSV":
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image_r, image_g, image_b = cv.split(image)
            image_HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            image_H, image_S, image_V = cv.split(image_HSV)
            image = cv.merge((image_r, image_g, image_b, image_H, image_S, image_V))
        elif color == "RGB+EDGES":
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image_r, image_g, image_b = cv.split(image)
            image_canny = cv.Canny(image,100,200)
            image = cv.merge((image_r, image_g, image_b, image_canny))
        elif color == "H":
            image_HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            image_H, image_S, image_V = cv.split(image_HSV)
            image = image_H

        images_set.append(tf.convert_to_tensor(image))
        n += 1
    return images_set

def import_labels(path, max=-1):
    images_set = []
    n = 0 
    for images in os.listdir(path):
       if n < max or max == -1:
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
  
def create_U_model(img_height, img_width, img_channels, num_classes = 1, show_summary = False, node_num = 16): # Tworzenie modelu siecu konwolucyjnej (U-net)

    #Architektur sieci U_net
    inputs = tf.keras.layers.Input((img_height, img_width, img_channels)) # Input layer, piewrwsza wartswa przyjmuje obraz o wymiarach (img_height, img_width, img_channels)

    #Contraction path
    c1 = tf.keras.layers.Conv2D(node_num, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs) #Pierwsza warstwa konwolucyjna, node_num - filtrów (3x3), kernel_in - sposób inicjalizaji wag dla filtrów,
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(node_num, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    b1 = tf.keras.layers.BatchNormalization()(c1)
    r1 = tf.keras.layers.ReLU()(b1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(r1)

    c2 = tf.keras.layers.Conv2D(node_num*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(node_num*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    b2 = tf.keras.layers.BatchNormalization()(c2)
    r2 = tf.keras.layers.ReLU()(b2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(r2)
    
    c3 = tf.keras.layers.Conv2D(node_num*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(node_num*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    b3 = tf.keras.layers.BatchNormalization()(c3)
    r3 = tf.keras.layers.ReLU()(b3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(r3)
    
    c4 = tf.keras.layers.Conv2D(node_num*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(node_num*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    b4 = tf.keras.layers.BatchNormalization()(c4)
    r4 = tf.keras.layers.ReLU()(b4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(r4)
    
    c5 = tf.keras.layers.Conv2D(node_num*16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    b5 = tf.keras.layers.BatchNormalization()(c5)
    r5 = tf.keras.layers.ReLU()(b5)
    c5 = tf.keras.layers.Dropout(0.3)(r5)
    c5 = tf.keras.layers.Conv2D(node_num*16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(node_num*8, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.BatchNormalization()(u6)
    u6 = tf.keras.layers.ReLU()(u6)

    u7 = tf.keras.layers.Conv2DTranspose(node_num*4, (2, 2), strides=(2, 2), padding='same')(u6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.BatchNormalization()(u7)
    u7 = tf.keras.layers.ReLU()(u7)

    u8 = tf.keras.layers.Conv2DTranspose(node_num*2, (2, 2), strides=(2, 2), padding='same')(u7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.BatchNormalization()(u8)
    u8 = tf.keras.layers.ReLU()(u8)
    
    u9 = tf.keras.layers.Conv2DTranspose(node_num, (2, 2), strides=(2, 2), padding='same')(u8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    u9 = tf.keras.layers.BatchNormalization()(u9)
    u9 = tf.keras.layers.ReLU()(u9)
    
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(u9)

    u_net_model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    u_net_model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy']) # Konfiugracja modelu

    if show_summary == True: # Wyświetl podsumowanie inforamcji o sieci
        u_net_model.summary()

    return u_net_model

def create_U_model_T2(img_height, img_width, img_channels, num_classes = 1, show_summary = False, node_num = 16): # Inna wersja architekrury (prostsza i taka bardziej typowa)
    inputs = tf.keras.layers.Input((img_height, img_width, img_channels)) 

    c1 = tf.keras.layers.Conv2D(node_num, (3, 3), activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(node_num, (3, 3), activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = tf.keras.layers.Dropout(0.25)(p1)

    c2 = tf.keras.layers.Conv2D(node_num*2, (3, 3), activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(node_num*2, (3, 3), activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = tf.keras.layers.Dropout(0.5)(p2)
    
    c3 = tf.keras.layers.Conv2D(node_num*4, (3, 3), activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(node_num*4, (3, 3), activation='relu', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = tf.keras.layers.Dropout(0.5)(p3)
    
    c4 = tf.keras.layers.Conv2D(node_num*8, (3, 3), activation='relu', padding='same')(p3)
    c4 = tf.keras.layers.Conv2D(node_num*8, (3, 3), activation='relu', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
    p4 = tf.keras.layers.Dropout(0.5)(p4)
    
    c5 = tf.keras.layers.Conv2D(node_num*16, (3, 3), activation='relu', padding='same')(p4)
    c5 = tf.keras.layers.Conv2D(node_num*16, (3, 3), activation='relu', padding='same')(c5)

    #Expansive path 
    dc4 = tf.keras.layers.Conv2DTranspose(node_num*8, (2, 2), strides=(2, 2), padding='same')(c5)
    uc4 = tf.keras.layers.concatenate([dc4, c4])
    uc4 = tf.keras.layers.Dropout(0.5)(uc4)
    uc4 = tf.keras.layers.Conv2D(node_num * 8, (3, 3), activation="relu", padding="same")(uc4)
    uc4 = tf.keras.layers.Conv2D(node_num * 8, (3, 3), activation="relu", padding="same")(uc4)

    dc3 = tf.keras.layers.Conv2DTranspose(node_num*4, (2, 2), strides=(2, 2), padding='same')(uc4)
    uc3 = tf.keras.layers.concatenate([dc3, c3])
    uc3 = tf.keras.layers.Dropout(0.5)(uc3)
    uc3 = tf.keras.layers.Conv2D(node_num * 4, (3, 3), activation="relu", padding="same")(uc3)
    uc3 = tf.keras.layers.Conv2D(node_num * 4, (3, 3), activation="relu", padding="same")(uc3)

    dc2 = tf.keras.layers.Conv2DTranspose(node_num*2, (2, 2), strides=(2, 2), padding='same')(uc3)
    uc2 = tf.keras.layers.concatenate([dc2, c2])
    uc2 = tf.keras.layers.Dropout(0.5)(uc2)
    uc2 = tf.keras.layers.Conv2D(node_num * 2, (3, 3), activation="relu", padding="same")(uc2)
    uc2 = tf.keras.layers.Conv2D(node_num * 2, (3, 3), activation="relu", padding="same")(uc2)
    
    dc1 = tf.keras.layers.Conv2DTranspose(node_num, (2, 2), strides=(2, 2), padding='same')(uc2)
    uc1 = tf.keras.layers.concatenate([dc1, c1])
    uc1 = tf.keras.layers.Dropout(0.5)(uc1)
    uc1 = tf.keras.layers.Conv2D(node_num, (3, 3), activation="relu", padding="same")(uc1)
    uc1 = tf.keras.layers.Conv2D(node_num, (3, 3), activation="relu", padding="same")(uc1)
    
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), padding="same" ,activation='sigmoid')(uc1)

    u_net_model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    u_net_model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy']) # Konfiugracja modelu

    if show_summary == True: # Wyświetl podsumowanie inforamcji o sieci
        u_net_model.summary()

    return u_net_model
   
class Next_button(object):  # Klasa do przycisku zarządzania przyciskiem w funkcji test_model
    def __init__(self, axes, label, test_images, test_labels, u_net_model):
        self.button = Button(axes, label)
        self.button.on_clicked(self.clicked)
        self.test_images = test_images
        self.test_labels = test_labels
        self.u_net_model = u_net_model

    def clicked(self, event):
        i = np.random.randint(0,len(self.test_images)-1) # Losowa zdjęcie i masska ze zbioru testowego
        sample_image = self.test_images[i] 
        sample_mask = self.test_labels[i]

        prediction = self.u_net_model.predict(sample_image[tf.newaxis, ...])[0]


        plt.subplot(131)
        if len(sample_image.shape) < 3: # Jeśli tylko jeden kanał
            if COLOR_NAME == 'H':
                plt.imshow(sample_image, cmap='hsv')
            if COLOR_NAME == 'GRAY':
                plt.imshow(sample_image, cmap='gray')
            else:
                plt.imshow(sample_image)
        else:
            if sample_image.shape[2] > 3: # Jeśli jest więcej niż 3 kanały to weź pierwsze 3
                plt.imshow(sample_image[:,:,0:3])
            else:
                plt.imshow(sample_image)

        plt.subplot(132)
        predicted_mask = (prediction > 0.5).astype(np.uint8) # 0.5 próg powyżej którego pikel trakowany jest jako drzewo
        plt.imshow(tf.keras.utils.array_to_img(predicted_mask))
        plt.axis('off')
        plt.title('Przewidywana maska')

        plt.subplot(133)
        plt.imshow(sample_mask)
        plt.axis('off')
        plt.title('Poprawna maska')

        plt.draw()

def test_model(u_net_model, test_images, test_labels): # Funckja do testowania działania sieci 

    i = np.random.randint(0,len(test_images)-1) # Losowa zdjęcie i masska ze zbioru testowego
    sample_image = test_images[i] 
    sample_mask = test_labels[i]

    prediction = u_net_model.predict(sample_image[tf.newaxis, ...])[0] # Przewidywanie maski, tf.newaxis powiększa wymiar macierzy o jeden, predict oczukuje zbioru danych  

    fig = plt.figure(figsize=(10,5))

    plt.subplot(131)

    if len(sample_image.shape) < 3: # Jeśli tylko jeden kanał
        if COLOR_NAME == 'H':
            plt.imshow(sample_image, cmap='hsv')
        if COLOR_NAME == 'GRAY':
            plt.imshow(sample_image, cmap='gray')
        else:
            plt.imshow(sample_image)
    else:
        if sample_image.shape[2] > 3: # Jeśli jest więcej niż 3 kanały to weź pierwsze 3
            plt.imshow(sample_image[:,:,0:3])
        else:
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

    axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    bnext = Next_button(axnext, "NEXT", test_images, test_labels, u_net_model)

    plt.show()


# Definicja stałych
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANELS = 4
BATCH_SIZE = 32
EPOCHS = 20
UNITS_NUM = 16
MODEL_NAME= "RGB+H_E20_T2"
COLOR_NAME= "RGB+H"
MODEL_TYPE = 2
FINAL_TEST = False

######################## MAIN #################
def main():
   
    if MODEL_TYPE == 1:
        u_net_model = create_U_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANELS, num_classes=1, show_summary=True, node_num=UNITS_NUM) # Tworzenie modelu
    elif MODEL_TYPE == 2:
        u_net_model = create_U_model_T2(IMG_HEIGHT, IMG_WIDTH, IMG_CHANELS, num_classes=1, show_summary=True, node_num=UNITS_NUM)

    model_choice = input("[0] - stórz nowy model,  [1] - wczytaj model: ") # Wybór czy wczytać model czy uczyć nowy


    if model_choice == '0':
        print("Wczytywanie zbioru trenującego i walidacyjnego...")
        with tf.device('/CPU:0'): # Wykonuj obliczenia na CPU i przechowuj zmienne W RAM nie w VRAM, domyślnie wszystko z tf robione jest na GPU
            train_images = import_images("images/train/img", color=COLOR_NAME) # Dane do uczenia 
            train_labels = import_labels("labels/train/img")

            valid_images = import_images("images/valid/img",color=COLOR_NAME) # Dane do tesotwania działania sieci
            valid_labels = import_labels("labels/valid/img")

            # Skalowanie i konwersja
            train_images = [proces_images(i) for i in train_images] # Dla każdego elementu z train_images wykonaj funckcje proces_image()
            train_labels = [proces_labels(l) for l in train_labels]

            valid_images = [proces_images(i) for i in valid_images]
            valid_labels = [proces_labels(l) for l in valid_labels]

            # Konwersja na tf.Dataset
            train_X = tf.data.Dataset.from_tensor_slices(train_images)
            train_Y = tf.data.Dataset.from_tensor_slices(train_labels)

            valid_X = tf.data.Dataset.from_tensor_slices(valid_images)
            valid_Y = tf.data.Dataset.from_tensor_slices(valid_labels)


        train_set = tf.data.Dataset.zip((train_X, train_Y)) #Lączenie datasetów, połączenie zdjęć z maskami
        valid_set = tf.data.Dataset.zip((valid_X, valid_Y))

        #### Podzial na batche
        AT = tf.data.AUTOTUNE ## sprwadzić
        STEPS_PER_EPOCH = len(train_images)//BATCH_SIZE # Kroki na epoke (liczba_zdjęć podzielona przez batch_size)
        VALIDATION_STEPS = len(valid_images)//BATCH_SIZE

        Buffer = len(train_images) # Bufor do losowania danych
        train_set = train_set.cache().shuffle(Buffer).batch(BATCH_SIZE).repeat() # Podziel dane na batche i przemieszaj
        train_set = train_set.prefetch(buffer_size=AT) # To zwiększa wydajność kosztem zużycia pamięci
        valid_set = valid_set.batch(BATCH_SIZE) 
        
        start = time.time() # Pomiar czasu uczenia
        history = u_net_model.fit(train_set, validation_data = valid_set, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, epochs = EPOCHS) # Uczenia modelu
        end = time.time()
        loss_history = history.history['loss']
        acc_history = history.history['accuracy']
        val_loss_history = history.history['val_loss']
        val_acc_history = history.history['val_accuracy']

        plt.figure(figsize=[12.8, 4.8]) # Wykres zmian "loss" (chyba jako koszt sie to tłumaczy) przy kolejnych epokach
        plt.subplot(1, 2, 1)
        plt.plot(loss_history, 'b', label="Training loss")
        plt.plot(val_loss_history, 'ro', label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(acc_history, 'b', label="Training accuracy")
        plt.plot(val_acc_history, 'ro', label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("figures/" + MODEL_NAME + ".png")
        plt.show()

        # Zapisanie informacji o modelu do pliku txt
        file = open("models_info/" + MODEL_NAME + ".txt", "w")
        file.write("Training time: " + f'{end - start:.3f}' + " s" + '\n')
        file.write("Model type: " + str(MODEL_TYPE) + " Epoch: " + str(EPOCHS) + " COLOR: " + COLOR_NAME + " Batch size: " + str(BATCH_SIZE) +  " UNITS_NUM: " + str(UNITS_NUM) + '\n')
        file.write("Trainig set: Loss = " + f'{loss_history[-1]:.3f}' + "  Accuracy: " + f'{acc_history[-1]:.3f}' + '\n')
        file.write("Validation set: Loss = " + f'{val_loss_history[-1]:.3f}' + "  Accuracy: " + f'{val_acc_history[-1]:.3f}' + '\n')
        file.close()
        u_net_model.save("saved_model/" + MODEL_NAME+ ".keras") # Zapisz model
    
    if model_choice == "1":
        print("Dostępne modele: ")
        print(os.listdir("saved_model"))
        model_choice = input("Wybierz model (wpisz bez .keras): ")
        u_net_model.load_weights("saved_model/" + model_choice + ".keras") # Wczytanie wag z pliku

        print("Wczytywanie zbioru walidacyjnego...")

        with tf.device('/CPU:0'):

            valid_images = import_images("images/valid/img",color=COLOR_NAME) # Dane do tesotwania działania sieci
            valid_labels = import_labels("labels/valid/img")

            valid_images = [proces_images(i) for i in valid_images]
            valid_labels = [proces_labels(l) for l in valid_labels]

            valid_X = tf.data.Dataset.from_tensor_slices(valid_images)
            valid_Y = tf.data.Dataset.from_tensor_slices(valid_labels)

        valid_set = tf.data.Dataset.zip((valid_X, valid_Y))
        VALIDATION_STEPS = len(valid_images)//BATCH_SIZE
        valid_set = valid_set.batch(BATCH_SIZE) 

    print("Dokładność dla zbioru validacyjnego")

    u_net_model.evaluate(valid_set) # Sprawdzenie sieci na zbiorze validacyjnym

    test_model(u_net_model, valid_images, valid_labels) # Testowanie sieci na przykładowych zdjęciach ze zbioru validacyjnego


    if FINAL_TEST == True:
        with tf.device('/CPU:0'): 
            print("Wczytywanie zbioru testowego...")
            test_images = import_images("images/test/img",color=COLOR_NAME) # Dane do tesotowania
            test_labels = import_labels("labels/test/img")

            test_images = [proces_images(i) for i in test_images]
            test_labels = [proces_images(i) for i in test_labels]

            test_X = tf.data.Dataset.from_tensor_slices(test_images)
            test_Y = tf.data.Dataset.from_tensor_slices(test_labels)

        test_set  = tf.data.Dataset.zip((test_X, test_Y))
        test_set = test_set.batch(BATCH_SIZE)

        u_net_model.evaluate(test_set) #Sprawdzenie sieci na zbiorze testowym 

        test_model(u_net_model, test_images, test_labels)

if __name__ == "__main__":
   main()