import base64
import io

import cv2
import numpy as np
import os
from PIL import Image
from flask import Flask, render_template, request, send_file, jsonify
from keras.models import load_model

# Завантаження моделі для колоризації
model = load_model('model')

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

    img = np.array(img)
    img = img.astype(np.uint8)
    img = cv2.resize(img, (160, 160))

    # Виконуємо операції для кольоризації зображення
    img = np.expand_dims(img, axis=0)
    pred_img = model.predict(img)

    # Відновлюємо масштаб та перетворюємо на правильний діапазон значень
    pred_img = (pred_img * 127.5) + 127.5
    pred_img = np.clip(pred_img[0], 0, 255).astype(np.uint8)

    # Зберігаємо кольорове зображення у файл
    colorized_image_path = 'colorized_image.jpg'
    pred_img_rgb = cv2.cvtColor(pred_img[0], cv2.COLOR_BGR2RGB)
    colorized_img = Image.fromarray(pred_img_rgb)
    colorized_img.save(colorized_image_path, format='JPEG')

    # Зберігаємо кольорове зображення у форматі BytesIO
    output = io.BytesIO()
    colorized_img = Image.fromarray(pred_img[0].astype(np.uint8))
    colorized_img.save(output, format='JPEG')
    output.seek(0)

    # Перетворюємо BytesIO на рядок base64
    base64_image = base64.b64encode(output.getvalue()).decode('utf-8')

    # Повертаємо посилання на кольорове зображення та саме зображення у відповіді
    colorized_image_url = f'{request.host_url}colorized_image.jpg'
    # return jsonify({'image_url': colorized_image_url, 'image': base64_image})
    return render_template('index.html')


# Маршрут для отримання кольоризованого зображення
@app.route('/colorized_image.jpg')
def get_colorized_image():
    # Отримуємо шлях до кольорового зображення
    colorized_image_path = 'colorized_image.jpg'

    # Перевіряємо, чи існує файл
    if os.path.exists(colorized_image_path):
        return send_file(colorized_image_path, mimetype='image/jpeg')
    else:
        return jsonify({'error': 'Colorized image not found'})


if __name__ == '__main__':
    app.run()
