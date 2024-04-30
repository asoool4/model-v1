from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import io
import os
import kagglehub

app = Flask(__name__)
CORS(app)

SAVED_MODEL_PATH = kagglehub.model_download("kaggle/esrgan-tf2/tensorFlow2/esrgan-tf2")

model = hub.load(SAVED_MODEL_PATH)
def preprocess_image(image):
    hr_image = tf.image.decode_image(tf.io.read_file(image))

    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

@app.route('/superres', methods=['POST'])
def super_res():
    file = request.files['image']
    image = Image.open(file.stream)  # PIL image
    image.save("temp.png")
    hr_image = preprocess_image("temp.png")
    fake_image = model(hr_image)
    fake_image = tf.squeeze(fake_image)
    fake_image = tf.clip_by_value(fake_image, 0, 255)
    fake_image = Image.fromarray(tf.cast(fake_image, tf.uint8).numpy())
    fake_image.save("temp_super_res.png")
    return send_file("temp_super_res.png", mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
