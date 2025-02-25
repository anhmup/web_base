import numpy as np 

from flask import Flask, render_template, request, redirect, url_for
import os
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

RESULT_IMAGE_NAME = "processed_image.jpg" 
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, RESULT_IMAGE_NAME)
            file.save(file_path)
            return render_template("index_2.html", filename= RESULT_IMAGE_NAME, original_image_path=RESULT_IMAGE_NAME)
    return render_template("index_2.html", filename=None)

@app.route("/draw", methods=["POST"])
def draw_rectangles():
    original_image_path = request.form["original_image_path"]
    input_path = os.path.join(UPLOAD_FOLDER, original_image_path)
    output_path = os.path.join(RESULT_FOLDER, "processed_" + original_image_path)
    if not os.path.exists(input_path):
        return "No uploaded image found!"
    image = cv2.imread(input_path)
    height, width, _ = image.shape

    # Vẽ một hình chữ nhật đơn giản (ví dụ ngẫu nhiên)
    start_point = (width // 4, height // 4)
    end_point = (3 * width // 4, 3 * height // 4)
    color = (0, 255, 0)  # Màu xanh lá cây (BGR)
    thickness = 3
    cv2.rectangle(image, start_point, end_point, color, thickness)

    cv2.imwrite(output_path, image)
    
    return render_template("result.html",  original_image_path=input_path, drawn_image_path=output_path)
    # return redirect(url_for("result_page"))
# @app.route("/result")
# def result_page():
#     return render_template("result.html" , original_image_path=input_path, drawn_image_path=output_path)


if __name__ == "__main__":
    app.run(debug=True)