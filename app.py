import gradio as gr
import numpy as np
import tensorflow as tf
import os
from PIL import Image

directory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
model_path = os.path.join(directory, 'model.keras')
model = tf.keras.models.load_model(model_path)


def predict_image(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    image_size = 100
    img = img.resize((image_size, image_size))

    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = img_arr / 255.0

    predictions = model.predict(img_arr)
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_names = ['cat', 'dog']

    return {class_names[predicted_class]: float(predictions[0][predicted_class])}

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=2),
    title="Bildklassifikator",
    description="Laden Sie ein Bild hoch, um eine Klassifikation durchzuführen."
)

demo.launch(share=True,
            server_name="0.0.0.0", server_port=7860)