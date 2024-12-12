import os
import random
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)

# Пути к папкам с изображениями
folder1_path = '/content/drive/My Drive/datasets/impressionist/validation/validation'
folder2_path = '/content/drive/My Drive/datasets/impressionist/training/training'

# Функция для смешивания двух изображений с весом w
def blend_images(img1, img2, w):
    img1 = np.array(img1)
    img2 = np.array(img2)
    blended_img = (1 - w) * img1 + w * img2
    return Image.fromarray(np.uint8(blended_img))

# Функция для случайного выбора изображения из папки
def random_image_from_folder(folder_path):
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    random_image = random.choice(image_files)
    return Image.open(os.path.join(folder_path, random_image))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/blend', methods=['GET', 'POST'])
def blend():
    if request.method == 'POST':
        # Случайный выбор изображений из двух папок
        image1 = random_image_from_folder(folder1_path)
        image2 = random_image_from_folder(folder2_path)

        # Убедитесь, что изображения одного размера
        image2 = image2.resize(image1.size)

        # Генерация значений w от 0.0 до 1.0
        w_values = np.linspace(0, 1, 11)

        # Создание временных изображений для отображения
        output_images = []
        for i, w in enumerate(w_values):
            blended_image = blend_images(image1, image2, w)
            img_path = f"static/images/blended_{i}.png"
            blended_image.save(img_path)
            output_images.append(img_path)

        return render_template('blend.html', images=output_images)

    return render_template('index.html')

@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory('static/images', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
