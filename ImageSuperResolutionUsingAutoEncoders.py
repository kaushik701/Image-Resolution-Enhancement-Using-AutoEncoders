# %%
import numpy as np
import cv2
import glob
import tensorflow
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
# %%
face_images = glob.glob(r'C:\Users\Kaushik\Desktop\MLProjects\Image Compression and Generation using Variational Autoencoders\Completed_Notebook_Data_Autoencoders\lfw-deepfunneled\C*\*.jpg')
print(len(face_images))
# %%
print(face_images[:2])
# %%
from tqdm import tqdm
# %%
img_array = []
for i in tqdm(face_images):
    img = image.load_img(i, target_size=(80, 80, 3))
    img = image.img_to_array(img)
    img = img / 255.
    img_array.append(img)
# %%
with open('img_array.pickle', 'wb') as f:
    pickle.dump(img_array, f)
print(len(img_array))
# %%
all_images = np.array(img_array)
# Split test and train data. all_images will be our output images
train_x, val_x = train_test_split(all_images, random_state=32, test_size=0.4)

# %%
def pixalate_image(image, scale_percent=40):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    small_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # scale back to original size
    width = int(small_image.shape[1] * 100 / scale_percent)
    height = int(small_image.shape[0] * 100 / scale_percent)
    dim = (width, height)
    low_res_image = cv2.resize(small_image, dim, interpolation=cv2.INTER_AREA)
    return low_res_image

# %%
train_x_px = []
for i in range(train_x.shape[0]):
    temp = pixalate_image(train_x[i, :, :, :])
    train_x_px.append(temp)
train_x_px = np.array(train_x_px)  # Distorted images
# get low resolution images for the validation set
val_x_px = []
for i in range(val_x.shape[0]):
    temp = pixalate_image(val_x[i, :, :, :])
    val_x_px.append(temp)
val_x_px = np.array(val_x_px)

# %%
Input_img = Input(shape=(80, 80, 3))

# encoding architecture
x1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(Input_img)
x2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x1)
x3 = MaxPool2D(padding='same')(x2)
x4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x3)
x5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x4)
x6 = MaxPool2D(padding='same')(x5)
encoded = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x6)
# encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
# decoding architecture
x7 = UpSampling2D()(encoded)
x8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x7)
x9 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x8)
x10 = Add()([x5, x9])
x11 = UpSampling2D()(x10)
x12 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x11)
x13 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x12)
x14 = Add()([x2, x13])
# x3 = UpSampling2D((2, 2))(x3)
# x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
# x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
decoded = Conv2D(3, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l1(10e-10))(x14)
autoencoder = Model(Input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
# %%
autoencoder.summary()
# %%
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint('superResolution_checkpoint3.h5', save_best_only=True)
# %%
history = autoencoder.fit(train_x_px, train_x,
                          epochs=5,
                          validation_data=(val_x_px, val_x),
                          callbacks=[early_stopper,model_checkpoint])
# %%
val_loss,val_accuracy = autoencoder.evaluate(val_x_px, val_x)
print('val_loss, val_accuracy', (val_loss*100),(val_accuracy*100))
# %%
predictions = autoencoder.predict(val_x_px)
n = 5
plt.figure(figsize=(20, 10))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(val_x_px[i + 20])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(predictions[i + 20])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
