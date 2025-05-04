import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="remote-work-productivity.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul
st.title("Prediksi Tipe Pekerjaan")
st.write("Prediksi apakah karyawan cocok untuk bekerja secara Remote atau In-Office berdasarkan data input.")

# Input pengguna
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
hours_worked = st.slider("Jam Kerja per Minggu", 0, 80, 40)
productivity_score = st.slider("Skor Produktivitas", 0, 100, 70)
well_being = st.slider("Skor Kesejahteraan", 0, 100, 70)

# Encoding gender
gender_encoded = 1 if gender == "Male" else 0

# Susun input array
input_data = np.array([[gender_encoded, hours_worked, productivity_score, well_being]])
input_scaled = scaler.transform(input_data).astype(np.float32)

# Prediksi
if st.button("Prediksi Tipe Pekerjaan"):
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    st.success(f"Karyawan ini diprediksi cocok bekerja secara: **{predicted_label.upper()}**")
