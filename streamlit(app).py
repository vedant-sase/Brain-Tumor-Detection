import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

import os

# Directory path where the file is located
directory_path = 'D:/Brain Tumor Detector'

# File name to be appended to the directory path
file_name = 'my_model_categorical.keras'

# Join the directory path and file name to get the absolute path
absolute_path = os.path.join(directory_path, file_name)

print("Absolute Path:", absolute_path)



# Function to load and preprocess image
def preprocess_image(image):
    resized_image = image.resize((128, 128))
    normalized_image = np.array(resized_image) / 255.0
    return normalized_image

# Function to perform tumor detection
def detect_tumor(model, image):
    processed_image = preprocess_image(image)
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    return prediction[0][0]

# Streamlit UI
def main():
    st.title("Brain Tumor Detection")
    st.sidebar.title("Upload Image")

    uploaded_file = st.sidebar.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Load the trained model using an absolute path
            model_path = 'D:\Brain Tumor Detector\my_model_categorical'
            model = tf.keras.models.load_model(model_path)
            
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded MRI Image', use_column_width=True)

            # Perform tumor detection on button click
            if st.button("Detect Tumor"):
                with st.spinner('Detecting...'):
                    # Call detect_tumor function with the loaded model and image
                    tumor_probability = detect_tumor(model, image)

                    # Display the prediction result
                    if tumor_probability > 0.5:
                        st.error(f"Brain tumor detected with {tumor_probability:.2%} probability!")
                    else:
                        st.success(f"No brain tumor detected (Probability: {tumor_probability:.2%})")

        except Exception as e:
            st.error(f"Error: {e}")

# Run the app
if __name__ == '__main__':
    main()
