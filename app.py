import os
from flask import Flask, render_template, request, redirect
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model(
    "model.h5",
    custom_objects={
        "preprocess_input": preprocess_input,
        "Lambda": Lambda
    }
)

img_height = 32
img_width = 32

app.config["UPLOAD_FOLDER"] = "static/uploads"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None
    
    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            img_path = filepath

            img = image.load_img(filepath, target_size=(img_height, img_width))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)[0][0]
            label = "REAL" if pred >= 0.5 else "FAKE"
            prediction = f"{label} (sigmoid={pred:.4f})"

    return render_template("index.jinja", prediction=prediction, img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)