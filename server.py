import cv2
import numpy as np
import os
from PIL import Image
from flask import Flask, render_template, request, send_file, jsonify
from keras.models import load_model

# Завантаження моделі для колоризації
model = load_model('model(5)')

save_dir = 'colorized_images'
os.makedirs(save_dir, exist_ok=True)

# Ініціалізація Flask додатку
app = Flask(__name__)

# Головна сторінка
@app.route('/')
def index():
    return render_template('index.html')


# Маршрут для обробки POST-запиту зі зображенням для колоризації
@app.route('/colorize', methods=['POST'])
def colorize():
    # Отримуємо чорнобіле зображення, надіслане у вигляді файлу
    file = request.files['image']

    # Завантажуємо чорнобіле зображення і перетворюємо його у RGB-режим
    img = Image.open(file)
    img = img.convert('RGB')
    #
    img = np.array(img)
    img = cv2.resize(img, (160, 160))
    img = img.astype('float32') / 255.0

    # Виконуємо операції для кольоризації зображення
    img = np.expand_dims(img, axis=0)
    pred_img = model.predict([img, img])

    pred_img = pred_img.reshape(160, 160, 3)
    pred_img = cv2.resize(pred_img, (320, 320), interpolation=cv2.INTER_LANCZOS4)

    # Масштабування значень пікселів до діапазону 0-255
    predicted_img = cv2.convertScaleAbs(pred_img, alpha=(255.0 / pred_img.max()))

    # Конвертація зображення в об'єкт Image
    pil_image = Image.fromarray(predicted_img)

    # Збереження колоризованого зображення
    save_path = os.path.join(save_dir, f'colorized_image.png')
    pil_image.save(save_path)

    return jsonify({'image_url':  f'{request.host_url}colorized_image.jpg'})

# Маршрут для отримання кольоризованого зображення
@app.route('/colorized_image.jpg')
def get_colorized_image():
    # Отримуємо шлях до кольорового зображення
    colorized_image_path = 'colorized_images/colorized_image.png'

    # Перевіряємо, чи існує файл
    if os.path.exists(colorized_image_path):
        return send_file(colorized_image_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Colorized image not found'})


if __name__ == '__main__':
    app.run()
