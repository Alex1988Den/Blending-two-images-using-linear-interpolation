from flask import Flask, render_template, request
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        content_image = request.files["content_image"]
        style_image = request.files["style_image"]

        # Тут добавьте вашу логику обработки изображений
        # Например, сохранение файлов, передача в модель и получение результата
        
        result = "/static/result_image.jpg"  # Путь к изображению после обработки

        # Сохраните обработанное изображение в папку static
        content_image.save(os.path.join("static", "content_image.jpg"))
        style_image.save(os.path.join("static", "style_image.jpg"))
        # Обработанное изображение сохраните в папке static

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
