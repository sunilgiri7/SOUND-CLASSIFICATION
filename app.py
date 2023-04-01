import streamlit as st
import librosa
import librosa.display
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.models import load_model

"""
Created on sat April 1 12:53:04 2023
@author: Sunil.Giri
"""

# Load the model from a file
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)
st.title('SOUND CLASSIFICATION APP')
file_path = st.text_input(label="Enter path to audio file")
# Load audio file and extract features when submit button is clicked
if st.button("Submit"):
    if file_path:
        try:
            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
            mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
            mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

            model = load_model('audio_classification.hdf5')
            predicted_label = model.predict(mfccs_scaled_features)
            new_predicted = np.argmax(predicted_label, axis=1)
            labelencoder = LabelEncoder()
            labelencoder = np.load('label_encoder.pkl', allow_pickle=True)
            prediction_class = labelencoder.inverse_transform(new_predicted)

            # Display the predicted label
            st.write("Predicted label:", prediction_class[0])

        except Exception as e:
            st.write("Error loading audio file:", e)
