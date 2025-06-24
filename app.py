import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from tensorflow.keras.models import load_model
import json

# Load saved model and preprocessor
model = load_model("heart_disease_model.keras")
scaler = joblib.load("scaler.pkl")
shap_background = joblib.load("shap_background.pkl")
with open("feature_columns.json") as f:
    feature_cols = json.load(f)

# Set page config and refined CSS
dark_bg_url = "https://c4.wallpaperflare.com/wallpaper/615/484/27/artistic-abstract-black-ecg-wallpaper-preview.jpg"
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

with st.sidebar:
    st.image("https://github.com/im-Paulsive/AI-Health-Predictor/blob/main/ChatGPT%20Image%20Jun%2024%2C%202025%2C%2012_28_22%20PM.png", width=200)
    st.markdown("""
    ## 🩺 About this Model
    This AI model predicts the **risk of heart disease** based on:
    - Age, Gender, Ethnicity
    - BMI and Blood Pressure
    - Sleep quality and medication use
    - Dietary intake and physical activity

    > **Note**: This tool is for educational use only. Always consult medical professionals for health advice.
    """)

st.markdown(
    f"""
    <style>
    html, body, .stApp {{
        background-image: url('{dark_bg_url}');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
        color: #f0f0f0;
    }}
    .stApp {{
        background-color: rgba(0, 0, 0, 0.75);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 0 30px rgba(0,0,0,0.5);
    }}
    h1, h2, h3, label, .stTextInput>div>div>input, .stMarkdown, .stMetricValue, .stFormLabel {{
        color: #f0f0f0 !important;
        font-family: 'Segoe UI', sans-serif;
    }}
    .stSlider>div>div>div>div {{
        background-color: #f63366 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🫀 AI-Powered Heart Disease Risk Predictor")
st.write("Enter patient data to estimate the risk of heart disease.")

# Form for inputs
with st.form("prediction_form"):
    age = st.slider("Age (years)", 0, 100, 50, help="Age in completed years")
    gender = st.selectbox("Gender", ["Male", "Female"], help="Select biological sex")
    age_group = st.selectbox("Age Group", ["infant", "child", "young_adult", "adult", "senior"], help="Automatically derived group based on age")
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, help="Body Mass Index")
    bmi_category = st.selectbox("BMI Category", ["underweight", "normal", "overweight", "obese"], help="BMI classification")
    systolic = st.number_input("Systolic BP", min_value=80, max_value=200, value=120, help="Upper value in BP reading")
    diastolic = st.number_input("Diastolic BP", min_value=50, max_value=140, value=80, help="Lower value in BP reading")
    hypertension = st.selectbox("Hypertension", ["No", "Yes"], help="Do you have hypertension")
    smoker = st.selectbox("Smoker", ["No", "Yes"], help="Current or past smoking status")
    alcoholic = st.selectbox("Alcohol Use", ["No", "Yes"], help="Regular alcohol consumption")
    sleep = st.selectbox("Poor Sleep", ["No", "Yes"], help="Reported poor sleep quality")
    medication_count = st.slider("No. of Medications", 0, 20, 2, help="Number of prescribed medications")
    calories = st.slider("Calories Intake (kcal)", 500, 4000, 2000, help="Total calorie intake per day")
    protein = st.slider("Protein (g)", 0, 300, 50, help="Total protein intake in grams")
    carbs = st.slider("Carbohydrate (g)", 0, 500, 200, help="Total carbohydrate intake")
    fiber = st.slider("Fiber (g)", 0, 100, 20, help="Total dietary fiber")
    fat = st.slider("Saturated Fat (g)", 0, 100, 20, help="Total saturated fat")
    cholesterol = st.slider("Cholesterol (mg)", 0, 500, 200, help="Total cholesterol intake")
    sugar = st.slider("Sugar (g)", 0, 200, 50, help="Total sugar intake")
    sodium = st.slider("Sodium (mg)", 0, 8000, 2500, help="Sodium intake in milligrams")
    paq620 = st.selectbox("Vigorous Activity (PAQ620)", [0, 1], help="1 = Yes, 0 = No")
    pad615 = st.slider("Minutes Vigorous Activity (PAD615)", 0, 200, 30, help="Time spent in vigorous activity")
    pad630 = st.slider("Minutes Moderate Activity (PAD630)", 0, 300, 60, help="Time spent in moderate activity")
    submit = st.form_submit_button("Predict")

if submit:
    gender_val = 1 if gender == "Male" else 2
    smoker_val = 1 if smoker == "Yes" else 0
    alcoholic_val = 1 if alcoholic == "Yes" else 0
    sleep_val = 1 if sleep == "Yes" else 0
    hypertension_val = 1 if hypertension == "Yes" else 0
    input_dict = {
        'RIDAGEYR': age,
        'age_group': f"age_group_{age_group}",
        'RIAGENDR': gender_val,
        'BMXBMI': bmi,
        'bmi_category': f"bmi_category_{bmi_category}",
        'BPXSY1': systolic,
        'BPXDI1': diastolic,
        'hypertension': hypertension_val,
        'Smoker': smoker_val,
        'Alcoholic': alcoholic_val,
        'Poor_Sleep': sleep_val,
        'Medication_Count': medication_count,
        'DR1TKCAL': calories,
        'DR1TPROT': protein,
        'DR1TCARB': carbs,
        'DR1TFIBE': fiber,
        'DR1TSFAT': fat,
        'DR1TCHOL': cholesterol,
        'DR1TSUGR': sugar,
        'DR1TSODI': sodium,
        'PAQ620': paq620,
        'PAD615': pad615,
        'PAD630': pad630
    }

    input_df = pd.DataFrame([input_dict])

    # Fill missing feature columns with 0
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns
    input_df = input_df[feature_cols]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prob = model.predict(input_scaled)[0][0]
    pred = "🔴 High Risk" if prob >= 0.5 else "✅ Low Risk"

    st.subheader("Prediction Result")
    st.metric(label="Heart Disease Risk", value=f"{prob:.2%}")

    if prob < 0.5:
        st.success(pred)
        st.balloons()
    else:
        st.error(pred)

        # SHAP Explanation
    # SHAP Explanation
    with st.expander("🔍 Top 10 contributing features (SHAP explanation)"):
        if X_train_scaled is not None:
            idx = np.random.choice(len(X_train_scaled), min(100, len(X_train_scaled)), replace=False)
            background = X_train_scaled[idx]

            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(input_scaled)
            shap_values_array = np.array(shap_values)[0]  # shape: (1, n_features)

            st.subheader("Top 10 contributing features")
            mean_abs_shap = np.abs(shap_values_array).flatten()
            top_indices = np.argsort(mean_abs_shap)[-10:][::-1]
            top_features = [feature_cols[i] for i in top_indices]
            top_values = mean_abs_shap[top_indices]

            fig, ax = plt.subplots()
            ax.barh(top_features[::-1], top_values[::-1], color="#f63366")
            ax.set_xlabel("SHAP value (impact)")
            ax.set_title("Top 10 Important Features")
            st.pyplot(fig)
        else:
            st.warning("SHAP explanation skipped: Background training data not available.")


    st.download_button(
        label="Download Prediction Result",
        data=input_df.to_csv(index=False),
        file_name="patient_prediction.csv",
        mime="text/csv"
    )

    with st.expander("🔎 See explanation"):
        st.markdown("""
        - The model uses inputs like age, blood pressure, BMI, diet, and activity levels to estimate risk.
        - A score above 50% is classified as **High Risk**.
        - This is not a medical diagnosis — consult a doctor for real evaluation.
        """)

# Footer credits
st.markdown("""
---
imPaulsiveX™ | Powered by Streamlit + Tensorflow | https://github.com/im-Paulsive
""")


   


