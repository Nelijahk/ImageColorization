import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
import pickle

with open('color_img.pkl', 'rb') as f:
    color_img = pickle.load(f)

with open('gray_img.pkl', 'rb') as f:
    gray_img = pickle.load(f)

# Перетворення масивів у numpy масиви
color_img = np.array(color_img)
gray_img = np.array(gray_img)

# Розбиття на тренувальну та тестову вибірки
train_color_img, test_color_img, train_gray_img, test_gray_img = train_test_split(color_img, gray_img, test_size=0.2, random_state=42)

# reshaping
train_gray_img = np.reshape(train_gray_img, (len(train_gray_img), 160, 160, 3))
train_color_img = np.reshape(train_color_img, (len(train_color_img), 160, 160, 3))

test_gray_img = np.reshape(test_gray_img, (len(test_gray_img), 160, 160, 3))
test_color_img = np.reshape(test_color_img, (len(test_color_img), 160, 160, 3))

model = load_model('model(5)')

# Отримання прогнозів моделі для тестових зображень
predictions = model.predict([test_gray_img, test_gray_img])

# Вибір трьох випадкових індексів
random_indices = np.random.choice(len(test_gray_img), 10, replace=False)

# Виведення колоризованих зображень та їх відповідних чорно-білих зображень
for index in random_indices:
    # Отримання колоризованого зображення
    predicted_img = predictions[index].reshape(160, 160, 3)

    # Отримання чорно-білого зображення
    gray = test_gray_img[index].reshape(160, 160, 3)

    # Відображення зображень
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_img)
    plt.title('Colorized Image')
    plt.axis('off')

    plt.show()

loss, acc = model.evaluate([test_gray_img, test_gray_img], test_color_img, verbose=0)
print('Loss:', loss)
print('Accuracy:', acc)