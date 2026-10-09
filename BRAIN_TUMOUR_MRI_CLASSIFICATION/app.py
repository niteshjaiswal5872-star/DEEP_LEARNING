import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Brain MRI Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

MODEL_PATH = "brain_mri_cnn.keras"
IMG_SIZE = (320, 320)

classm = ["glioma", "meningioma", "no_tumor", "pituitary"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

st.title("🧠 Brain MRI Tumor Classification")
st.write("Upload a Brain MRI image to predict its class using a trained CNN model.")

try:
    model = load_model()
except Exception as e:
    st.error("Model could not be loaded.")
    st.code(str(e))
    st.stop()

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    if st.button("🔍 Predict"):
        img = image.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0]

        predicted_index = int(np.argmax(prediction))
        predicted_class = classm[predicted_index]
        confidence = float(prediction[predicted_index]) * 100

        st.subheader("Prediction")

        if predicted_class == "notumor":
            st.success(f"Prediction: {predicted_class.upper()}")
        else:
            st.warning(f"Prediction: {predicted_class.upper()}")

        st.write(f"Confidence: **{confidence:.2f}%**")

        st.subheader("Class Probabilities")

        for i, name in enumerate(classm):
            st.write(f"{name}: {prediction[i] * 100:.2f}%")
            st.progress(float(prediction[i]))

st.info(
    "This application is an educational AI project and should not be used "
    "for medical diagnosis."
)


