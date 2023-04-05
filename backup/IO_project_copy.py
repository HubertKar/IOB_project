
import tensorflow as tf
import cv2 as cv
import os
import numpy as np
from tensorflow.keras import datasets
from tensorflow.python.keras import layers, models
import matplotlib.pyplot as plt


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

def import_images(path):
    images_set = []
    for images in os.listdir(path):
       image = cv.imread(path+'/'+images,1)
       image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
       image = tf.cast(image, tf.float32)
       image = image/255.0
       images_set.append(tf.convert_to_tensor(image))
    return images_set

def import_labels(path):
    images_set = []
    for images in os.listdir(path):
       image = cv.imread(path+'/'+images, 0)
       image = tf.expand_dims(image, axis=-1) # ????? sprawdzić czy to potrzebne
       image = tf.cast(image, tf.bool)
       images_set.append(tf.convert_to_tensor(image))
       
    return images_set

def proces_images(image):
    image = tf.cast(image, tf.float32)
    image = image/255.0
    return image


def proces_labels(label):          
    label = tf.expand_dims(label, axis=-1) # ????? sprawdzić czy to potrzebne
    label = tf.cast(label, tf.bool)
    return label

# Variables
img_width = 256
img_height = 256
img_channels = 3
batch_size = 2
EPOCHS = 20
Buffer = 1000
# 

# Import images and labels (masks)
train_images = import_images("images/train/img")
train_labels = import_labels("labels/train/img")

valid_images = import_images("images/valid/img")
valid_labels = import_labels("labels/valid/img")

#test_images = import_images("images/test/img")
#test_labels = import_labels("labels/test/img")

#scalling 
# train_images = [proces_images(i) for i in train_images]
# train_labels = [proces_labels(l) for l in train_labels]

# valid_images = [proces_images(i) for i in valid_images]
# valid_labels = [proces_labels(l) for l in valid_labels]

#test_images = proces_images(test_images)
#test_labels = proces_images(test_labels, 0)

# conert to tf Dataset
train_X = tf.data.Dataset.from_tensor_slices(train_images)
train_Y = tf.data.Dataset.from_tensor_slices(train_labels)

valid_X = tf.data.Dataset.from_tensor_slices(valid_images)
valid_Y = tf.data.Dataset.from_tensor_slices(valid_labels)

train_set = tf.data.Dataset.zip((train_X, train_Y))
valid_set = tf.data.Dataset.zip((valid_X, valid_Y))

############################# 
AT = tf.data.AUTOTUNE ## sprwadzić
STEPS_PER_EPOCH = 800//batch_size
VALIDATION_STEPS = 200//batch_size

train_set = train_set.cache().shuffle(Buffer).batch(batch_size).repeat()
train_set = train_set.prefetch(buffer_size=AT)

valid_set = valid_set.batch(batch_size) 

######## Architektur sieci nazrazie zajebana z internertu po robienie sieci U-net jest ciężkie

num_classes = 1

inputs = tf.keras.layers.Input((img_height, img_width, img_channels))

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
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

#################################

u_net_model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
u_net_model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

u_net_model.summary()

u_net_model.fit(train_set, validation_data = valid_set,steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, epochs = EPOCHS)