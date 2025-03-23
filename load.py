import streamlit as st
import pickle
import numpy as np

with open('Employee_Attrition_prediction.pkl', 'rb') as file:
    model = pickle.load(file)


st.title('Employee Attrition Prediction Model')
col1, col2, col3, col4 = st.columns(4)
with col1:
    gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    age = st.number_input('Age', min_value=20.0, max_value=100.0)

with col2:
    department = st.selectbox('Department', ['Sales', 'HR', 'Marketing', 'Engineering'])

with col3:
    years_at_company = st.number_input('Years at Company', min_value=0, max_value=50)
    satisfaction_level = st.slider('Satisfaction Level', min_value=0.0, max_value=1.0, step=0.01)

with col4:
    avg_monthly_hours = st.number_input('Average Monthly Hours', min_value=0, max_value=744)
    promotion_last_5_years = st.selectbox('Promotion Last 5 Years', ['Yes', 'No'])
    salary = st.selectbox('Salary Level', ['Low', 'Medium', 'High'])

gender_map = {'Male': 0, 'Female': 1}
promotion_map = {'No': 0, 'Yes': 1}
salary_map = {'Low': 0, 'Medium': 1, 'High': 2}
department_map = {
    'Sales': [1, 0, 0, 0],
    'HR': [0, 1, 0, 0],
    'Marketing': [0, 0, 1, 0],
    'Engineering': [0, 0, 0, 1],
}

gender_encoded = gender_map[gender]
promotion_encoded = promotion_map[promotion_last_5_years]
salary_encoded = salary_map[salary]
department_encoded = department_map[department]  # One-hot encoding for department

input_data = np.array([[age, gender_encoded, years_at_company, satisfaction_level, avg_monthly_hours, 
                        promotion_encoded, salary_encoded] + department_encoded])

if st.button('Predict Attrition Rate'):
    predicted_attrition_prob = model.predict_proba(input_data)[0][1] 

    st.write(f'Predicted Employee Attrition Probability: {predicted_attrition_prob:.2f}')