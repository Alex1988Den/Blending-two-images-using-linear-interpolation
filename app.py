from flask import Flask, render_template, request, redirect, url_for
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG19
from tensorflow.keras import backend as K
from werkzeug.utils import secure_filename

# Создание Flask-приложения
app = Flask(__name__)

# Пути к данным (сделайте папки для сохранения изображений)
train_dir = "C:/Users/user/Downloads/datasets/impressionist/training/training"
val_dir = "C:/Users/user/Downloads/datasets/impressionist/validation/validation"
upload_folder = 'static/uploads'
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Функция для проверки формата файла
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Загрузка модели VGG19
vgg = VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Слои для извлечения контента и стиля
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

def get_model():
    model = tf.keras.models.Model(inputs=vgg.input, 
                                  outputs=[vgg.get_layer(name).output for name in (style_layers + content_layers)])
    return model

model = get_model()

# Функция для загрузки и обработки изображения
def load_and_process_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

# Функция для вычисления Gram-матрицы
def gram_matrix(tensor):
    shape = tf.shape(tensor)
    channels = shape[-1]
    matrix = tf.reshape(tensor, [-1, channels])
    gram = tf.matmul(matrix, matrix, transpose_a=True)
    return gram

# Потеря для контента
def content_loss(content, generated):
    return K.sum(K.square(generated - content))

# Потеря для стиля
def style_loss(style, generated):
    style_gram = gram_matrix(style)
    generated_gram = gram_matrix(generated)
    return K.sum(K.square(style_gram - generated_gram))

# Роут для главной страницы
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        content_image = request.files["content_image"]
        style_image = request.files["style_image"]

        if content_image and allowed_file(content_image.filename) and style_image and allowed_file(style_image.filename):
            # Сохранение изображений
            content_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(content_image.filename))
            style_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(style_image.filename))

            content_image.save(content_image_path)
            style_image.save(style_image_path)

            # Загружаем и обрабатываем изображения
            content_img = load_and_process_image(content_image_path)
            style_img = load_and_process_image(style_image_path)

            # Создаем случайное изображение для генерации
            generated_image = tf.Variable(np.random.randn(1, 224, 224, 3).astype(np.float32))

            # Оптимизатор
            opt = tf.optimizers.Adam(learning_rate=0.02)

            # Функция для обучения
            def train_step(model, content_image, style_image, generated_image):
                with tf.GradientTape() as tape:
                    outputs = model(generated_image)
                    style_outputs, content_outputs = outputs[:len(style_layers)], outputs[len(style_layers):]
                    
                    # Потери
                    c_loss = content_loss(content_outputs[-1], content_image)
                    s_loss = sum([style_loss(style, gen) for style, gen in zip(style_outputs, outputs[:len(style_layers)])])
                    total_loss = c_loss + s_loss
                
                grads = tape.gradient(total_loss, generated_image)
                opt.apply_gradients([(grads, generated_image)])
                return total_loss

            # Проводим обучение
            for epoch in range(10):  # 10 эпох для примера
                train_step(model, content_img, style_img, generated_image)

            # Сохраняем результат
            result_image = generated_image.numpy()[0]
            result_path = "static/result_image.png"
            plt.imshow(result_image)
            plt.axis('off')
            plt.savefig(result_path)

            result = result_path

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
