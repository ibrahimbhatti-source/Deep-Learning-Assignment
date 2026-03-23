import os
import json
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Make upload folder if not exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load trained model
model = load_model("/content/drive/MyDrive/CNN_Flask_Project/face_recognition_cnn.h5")

# Load class names
with open("class_names.json", "r") as f:
    class_indices = json.load(f)

# Convert {"person1":0,"person2":1} into {0:"person1",1:"person2"}
class_names = {v: k for k, v in class_indices.items()}

# Image settings
img_height = 128
img_width = 128

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Load image
            img = image.load_img(filepath, target_size=(img_height, img_width))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            pred = model.predict(img_array)
            pred_index = np.argmax(pred, axis=1)[0]
            prediction = class_names[pred_index]

            image_path = filepath

    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)