import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.models import load_model


def build_colorization_model(input_shape):
    inputs = Input(shape=input_shape)
    condition = Input(shape=(160, 160, 3))

    concatenated_input = concatenate([inputs, condition], axis=3)

    # Енкодер
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(concatenated_input)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Декодер
    up6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(pool4)
    up6 = concatenate([up6, conv4], axis=3)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)

    outputs = Conv2D(3, (1, 1), activation='tanh')(conv9)

    model = Model(inputs=[inputs, condition], outputs=outputs)

    return model

# Відновлення масивів з файлів
with open('color_img.pkl', 'rb') as f:
    color_img = pickle.load(f)

with open('gray_img.pkl', 'rb') as f:
    gray_img = pickle.load(f)

# Перетворення масивів у numpy масиви
color_img = np.array(color_img)
gray_img = np.array(gray_img)

# Розбиття на тренувальну та тестову вибірки
train_gray_img = gray_img[:7000]
train_color_img = color_img[:7000]
test_gray_img = gray_img[:1000]
test_color_img = color_img[:1000]

# reshaping
train_gray_img = np.reshape(train_gray_img, (len(train_gray_img), 160, 160, 3))
train_color_img = np.reshape(train_color_img, (len(train_color_img), 160, 160, 3))

test_gray_img = np.reshape(test_gray_img, (len(test_gray_img), 160, 160, 3))
test_color_img = np.reshape(test_color_img, (len(test_color_img), 160, 160, 3))

print(train_gray_img.shape, train_color_img.shape)

# Створення моделі
model = build_colorization_model(input_shape=(160, 160, 3))
# model = load_model('model(3)')

# Компіляція моделі
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error', metrics=['accuracy'], loss_weights=[1.0, 100.0])

# Навчання моделі
model.fit([train_gray_img, train_gray_img], train_color_img, batch_size=32, epochs=10, validation_data=([test_gray_img, test_gray_img], test_color_img))

# Збереження моделі
model.save('model')
