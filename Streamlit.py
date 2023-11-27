import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load word index for Sentiment Classification
word_to_index = imdb.get_word_index()


# Function to perform sentiment classification
def sentiment_classification(new_review_text, model):
    max_review_length = 500
    new_review_tokens = [word_to_index.get(word, 0) for word in new_review_text.split()]
    new_review_tokens = pad_sequences([new_review_tokens], maxlen=max_review_length)
    prediction = model.predict(new_review_tokens)
    if type(prediction) == list:
        prediction = prediction[0]
    return "Positive" if prediction > 0.5 else "Negative"

# Function to perform tumor detection
def tumor_detection(img, model):
    img = Image.open(img)
    img=img.resize((128,128))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img)
    return "Tumour Found" if res else "No Tumor"

# Streamlit App
st.title("Deep Learning Predictions App")

# Choose between tasks
task = st.radio("Choose a project", ("Sentiment Classification", "Identification of Tumours"))

if task == "Sentiment Classification":
    # Input box for new review
    new_review_text = st.text_area("Add a New Status:", value="")
    if st.button("Submit") and not new_review_text.strip():
        st.warning("Kindly add a Status")

    if new_review_text.strip():
        st.subheader("Select a Model for Sentiment classification")
        model_option = st.selectbox("Choose Model", ("Perceptron", "Backpropagation", "DNN", "RNN", "LSTM"))

        # Load models dynamically based on the selected option
        if model_option == "Perceptron":
            with open(r'F:\Study\DUK\3rd Semester\Deep Learning\Code\AKS Streamlit\imdbP.pkl', 'rb') as file:
                model = pickle.load(file)
        elif model_option == "Backpropagation":
            with open(r'F:\Study\DUK\3rd Semester\Deep Learning\Code\AKS Streamlit\imdbBP.pkl', 'rb') as file:
                model = pickle.load(file)
        elif model_option == "DNN":
            model = load_model(r'F:\Study\DUK\3rd Semester\Deep Learning\Code\AKS Streamlit\AKS.keras')
        elif model_option == "RNN":
            model = load_model(r'F:\Study\DUK\3rd Semester\Deep Learning\Code\AKS Streamlit\AKSRNN.keras')
        elif model_option == "LSTM":
            model = load_model(r'F:\Study\DUK\3rd Semester\Deep Learning\Code\AKS Streamlit\AKSLSTM.keras')

        if st.button("Classify Sentiment"):
            result = sentiment_classification(new_review_text, model)
            st.subheader("Sentiment Classification Result")
            st.write(f"**{result}**")

elif task == "Identification of Tumours":
    st.subheader("Tumor Detection")
    uploaded_file = st.file_uploader("Select a picture of the tumour", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the tumor detection model
        model = load_model(r'F:\Study\DUK\3rd Semester\Deep Learning\Code\AKS Streamlit\CNN Tumor.keras')
        st.image(uploaded_file, caption="Uploaded the Image.", use_column_width=False, width=200)
        st.write("")

        if st.button("Detect Tumor"):
            result = tumor_detection(uploaded_file, model)
            st.subheader("Result of Tumor Detection")
            st.write(f"**{result}**")
