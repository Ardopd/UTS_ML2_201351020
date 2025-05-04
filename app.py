import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder jika ada
scaler = joblib.load('scaler.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="remote-work-productivity.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("Productivity Score Prediction")
st.write("Masukkan data karyawan untuk memprediksi skor produktivitas.")

# Form input
employment_type = st.selectbox("Tipe Pekerjaan", ["Remote", "In-Office"])
hours_worked = st.slider("Jam Kerja per Minggu", 0, 80, 40)
well_being = st.slider("Skor Kesejahteraan", 0, 100, 70)

# Encoding employment_type (pastikan sama dengan pelatihan model)
employment_encoded = 1 if employment_type == "Remote" else 0

if st.button("Prediksi Skor Produktivitas"):
    input_data = np.array([[employment_encoded, hours_worked, well_being]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_score = int(prediction[0][0])
    st.success(f"Prediksi Skor Produktivitas: **{predicted_score}**")
