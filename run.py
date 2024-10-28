import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
import os
model_path = os.path.join(os.path.dirname(__file__), "best_trained_save.h5")
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(model_path)
    return model

def predict_class(img, model):
    # Resize the image
    img = img.resize((224, 224))  # Resize to 224x224
    processed_img = image.img_to_array(img)  # Convert the image to array
    processed_img = np.expand_dims(processed_img, axis=0)  # Add batch dimension
    processed_img = inception_preprocess_input(processed_img)  # Preprocess image

    # Get prediction using the network
    predictions = model.predict(processed_img)[0]
    return predictions

model = load_model()
st.title('BushFire Classifier')

file = st.file_uploader("Upload an image", type=["jpg", "png"])

if file is None:
    st.text('Waiting for upload....')
else:
    slot = st.empty()
    slot.text('Running inference....')

    test_image = Image.open(file)
    st.image(test_image, caption="Input Image", width=400)

    pred = predict_class(test_image, model)

    classes = ['fire', 'no_fire', 'start_fire']
    result = classes[np.argmax(pred)]

    output = 'The image is a ' + result
    slot.text('Done')
    st.success(output)
