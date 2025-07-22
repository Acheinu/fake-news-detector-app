import streamlit as st
import joblib
import spacy # Import spacy
import pandas as pd # Import pandas (if needed for future expansions)

# Load spaCy model
# You might need to install 'en_core_web_sm' on the deployment environment
# !python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Define preprocessing function (same as in your notebook)
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Load the model and vectorizer
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    st.error("Model or Vectorizer files not found. Please ensure 'model.pkl' and 'vectorizer.pkl' are in the same directory.")
    st.stop() # Stop the app if files are missing


st.title("Fake News Detector")

user_input = st.text_area("Enter a news article:")

if st.button("Check if it's fake"):
    if user_input.strip():
        # Preprocess the input using the defined function
        cleaned_input = preprocess(user_input)
        # Vectorize the cleaned input
        vector = vectorizer.transform([cleaned_input])
        # Make prediction
        prediction = model.predict(vector)[0]

        # Display prediction and confidence score (optional, but good practice)
        # To get confidence, you'd need predict_proba and handle model.classes_
        # For simplicity, we'll just show the label for now.
        st.success(f"This news is: **{prediction}**")

    else:
        st.warning("Please enter some text to analyze.")
