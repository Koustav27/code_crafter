import streamlit as st
import tensorflow as tf
import numpy as np

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_plant_model():
    return tf.keras.models.load_model(
        "plant_disease_model_fixed.h5",
        compile=False
    )

model = load_plant_model()


# =========================
# CLASS NAMES
# =========================

class_name = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


# =========================
# PREDICTION
# =========================

def model_prediction(test_image):

    image = tf.keras.preprocessing.image.load_img(
        test_image,
        target_size=(64, 64)
    )

    input_arr = tf.keras.preprocessing.image.img_to_array(image)

    input_arr = np.expand_dims(input_arr, axis=0)

    prediction = model.predict(
        input_arr,
        verbose=0
    )

    result_index = np.argmax(prediction[0])

    confidence = float(np.max(prediction[0])) * 100

    return result_index, confidence


# =========================
# SIDEBAR
# =========================

st.sidebar.title("🌿 Dashboard")

app_mode = st.sidebar.selectbox(
    "Select Page",
    ["Home", "About", "Disease Recognition"]
)


# =========================
# HOME
# =========================

if app_mode == "Home":

    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")

    st.write("""
    Welcome to the Plant Disease Recognition System!

    This application uses a trained CNN model to identify
    plant diseases from leaf images.
    """)

    st.subheader("How It Works")

    st.write("""
    1. Go to Disease Recognition.
    2. Upload a plant leaf image.
    3. Click Predict Disease.
    4. The AI model will identify the disease.
    5. The confidence score will be displayed.
    """)

    st.subheader("Features")

    st.write("""
    - AI-based plant disease detection
    - 38 plant classes
    - Fast prediction
    - Confidence score
    - Simple interface
    """)


# =========================
# ABOUT
# =========================

elif app_mode == "About":

    st.header("📖 About")

    st.write("""
    This Plant Disease Recognition System uses a
    Convolutional Neural Network (CNN).

    The dataset contains approximately 87,000 RGB images
    covering 38 different plant disease and healthy classes.

    Dataset:

    - Training: 70,295 images
    - Validation: 17,572 images
    - Test: 33 images
    """)


# =========================
# DISEASE RECOGNITION
# =========================

elif app_mode == "Disease Recognition":

    st.header("🌱 Disease Recognition")

    test_image = st.file_uploader(
        "Choose a plant leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if test_image is not None:

        st.subheader("Uploaded Image")

        st.image(
            test_image,
            caption="Selected Plant Leaf",
            use_column_width=True
        )

        if st.button("🔍 Predict Disease"):

            with st.spinner("Analyzing image..."):

                try:

                    result_index, confidence = model_prediction(
                        test_image
                    )

                    st.success(
                        "🌿 Prediction: "
                        + class_name[result_index]
                    )

                    st.info(
                        f"🎯 Confidence: {confidence:.2f}%"
                    )

                except Exception as e:

                    st.error("Prediction failed.")

                    st.exception(e)

    else:

        st.info(
            "👆 Please upload a plant leaf image first."
        )