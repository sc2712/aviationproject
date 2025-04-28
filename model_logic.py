import numpy as np
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing import image
import streamlit as st
import os

class_labels = [
    'fake commercial aircraft (AI)',
    'fake military aircraft (AI)',
    'fake private aircraft (AI)',
    'real commercial aircraft',
    'real military aircraft',
    'real private aircraft'
]

def load_model_from_path(model_path):
    #Loads the Keras model from the specified path.
    try:
        # Ensure the file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model = keras_load_model(model_path)
        st.success("Model loaded successfully.")
        return model
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image_file, img_size=(224, 224)):
    img = image.load_img(image_file, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def classify_image(model, img_tensor):
    prediction = model.predict(img_tensor)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    class_name = class_labels[class_index]
    return class_name, confidence