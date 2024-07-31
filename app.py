import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('gradient_boosting_classifier.pkl')

# Streamlit app title
st.title('Heart Disease Prediction')

# Function to get user input
def get_user_input():
    st.header('Enter the patient data:')
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Sex', options=['Male', 'Female'])
    chest_pain_type = st.selectbox('Chest Pain Type', options=['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    resting_bp = st.number_input('Resting Blood Pressure', min_value=50, max_value=250, value=120)
    cholesterol = st.number_input('Serum Cholesterol in mg/dl', min_value=100, max_value=600, value=200)
    fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
    resting_ecg = st.selectbox('Resting Electrocardiographic Results', options=['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
    max_hr = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=250, value=150)
    exercise_angina = st.selectbox('Exercise Induced Angina', options=['Yes', 'No'])
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
    st_slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=['Upsloping', 'Flat', 'Downsloping'])

    # Map categorical inputs to numerical values
    sex = 1 if sex == 'Male' else 0
    chest_pain_type_mapping = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    chest_pain_type = chest_pain_type_mapping[chest_pain_type]
    resting_ecg_mapping = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
    resting_ecg = resting_ecg_mapping[resting_ecg]
    exercise_angina = 1 if exercise_angina == 'Yes' else 0
    st_slope_mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    st_slope = st_slope_mapping[st_slope]

    # Create a dataframe with the user input
    user_data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()

st.subheader('User Input:')
st.write(user_input)

prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

st.subheader('Prediction:')
heart_disease = 'Yes' if prediction[0] == 1 else 'No'
st.write(f'Heart Disease: {heart_disease}')

st.subheader('Prediction Probability:')
st.write(prediction_proba)
