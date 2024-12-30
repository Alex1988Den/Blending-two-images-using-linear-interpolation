import os
import random
import numpy as np
from flask import Flask, render_template, request, send_from_directory, url_for
from PIL import Image
import logging

# Инициализация приложения Flask
app = Flask(__name__, static_folder="static", template_folder="templates")

# Пути к папкам с изображениями
folder1_path = '/content/drive/My Drive/datasets/impressionist/validation/validation'
folder2_path = '/content/drive/My Drive/datasets/impressionist/training/training'

# Папка для сохранения изображений
static_images_path = os.path.join(app.static_folder, "images")
os.makedirs(static_images_path, exist_ok=True)

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)

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
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {folder_path}")
    
    random_image = random.choice(image_files)
    try:
        return Image.open(os.path.join(folder_path, random_image))
    except Exception as e:
        raise IOError(f"Error loading image {random_image}: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/blend', methods=['POST'])
def blend():
    try:
        # Логирование начала выполнения
        app.logger.debug("Starting to blend images.")
        
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
            img_path = os.path.join(static_images_path, f"blended_{i}.png")

            # Логирование пути сохранения
            app.logger.debug(f"Saving blended image to {img_path}")

            # Попытка сохранить изображение
            try:
                blended_image.save(img_path)
                output_images.append(f"images/blended_{i}.png")  # Путь относительно static
                app.logger.debug(f"Saved blended image {img_path}")
            except Exception as e:
                app.logger.error(f"Failed to save image {img_path}: {e}")

        # Логирование успешного завершения
        app.logger.debug("Blending images successful.")

        # Передаем список изображений в шаблон
        return render_template('blend.html', images=output_images)

    except Exception as e:
        app.logger.error(f"Error occurred: {e}")
        return "Internal Server Error", 500

@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory(static_images_path, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
