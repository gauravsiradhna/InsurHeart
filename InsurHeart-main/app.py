import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor # Specifically imported for regression in insurance model
from sklearn.preprocessing import StandardScaler
import os

# --- Page Configuration ---
st.set_page_config(page_title="Health & Insurance Predictor", page_icon="❤️‍🩹💰", layout="centered")

# --- Custom background and main content styling ---
st.markdown("""
    <style>
    body {
        background-color: #cce7f0; /* Slightly more distinct light blue background color */
        background-image: url('https://static.vecteezy.com/system/resources/thumbnails/008/137/021/small/insurance-concept-insurance-agent-and-clients-of_the_insurance_company_wide_banner_composition_with_bokeh_in_the_background_photo.jpeg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.85); /* Slightly more transparent white for content area */
        padding: 2rem;
        border-radius: 0.5rem;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #003366;
    }
    .stButton > button {
        background-color: #003366;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #002244;
    }
    /* Adjusted styles to be applied via st.markdown for content within alerts */
    .styled-error-box {
        padding: 1.5rem;
        background-color:#ffe6e6;
        border-left: 6px solid crimson;
        border-radius: 10px;
    }
    .styled-success-box {
        padding: 1.5rem;
        background-color:#e6ffed;
        border-left: 6px solid green;
        border-radius: 10px;
    }
    .styled-info-box {
        padding: 1.5rem;
        background-color:#f0f8ff;
        border-left: 6px solid #6cb2eb;
        border-radius: 10px;
    }
    h4.crimson-text {
        color: crimson;
    }
    h4.green-text {
        color: green;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load and Train Heart Disease Model ---
@st.cache_resource # Use st.cache_resource for models/scalers
def load_heart_disease_model():
    file_path = "heart_disease_data.csv"
    if not os.path.exists(file_path):
        st.error(f"Error: The file '{file_path}' for heart disease model was not found.")
        return None, None, None
    try:
        df = pd.read_csv(file_path)
        X = df.drop('target', axis=1)
        y = df['target']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Heart disease model remains RandomForestClassifier for binary prediction
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        return model, scaler, X.columns.tolist()
    except Exception as e:
        st.error(f"Error loading or training heart disease model: {e}. Please check 'heart_disease_data.csv'.")
        return None, None, None

# --- Load and Train Insurance Cost Model ---
@st.cache_resource # Use st.cache_resource for models/scalers
def load_insurance_model():
    file_path = "insurance_with_heart_features_updated.csv"
    if not os.path.exists(file_path):
        st.error(f"Error: The file '{file_path}' for insurance model was not found.")
        return None, None, None, None
    try:
        df = pd.read_csv(file_path)
        median_charge = df["charges"].median()
        # Target for insurance cost is now the continuous 'charges' for regression
        X = df.drop(["charges", "charges_binary"], axis=1, errors='ignore') # Ensure charges_binary is dropped if it exists
        y = df["charges"] # Target is the continuous 'charges'

        X_encoded = pd.get_dummies(X, drop_first=True)

        scaler = StandardScaler()
        numerical_cols_to_scale = ['age', 'bmi', 'children', 'probablity_of_heart_disease']
        numerical_cols_in_x_encoded = [col for col in numerical_cols_to_scale if col in X_encoded.columns]
        X_encoded[numerical_cols_in_x_encoded] = scaler.fit_transform(X_encoded[numerical_cols_in_x_encoded])

        # Changed to RandomForestRegressor for predicting continuous values
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_encoded, y)
        return model, scaler, X_encoded.columns.tolist(), median_charge
    except Exception as e:
        st.error(f"Error loading or training insurance model: {e}. Please check 'insurance_with_heart_features_updated.csv'.")
        return None, None, None, None

# Load models and their components
heart_model, heart_scaler, heart_features = load_heart_disease_model()
insurance_model, insurance_scaler, insurance_features, median_charge = load_insurance_model()

# --- Main Application Logic ---
st.markdown("<h1 style='text-align: center;'>Integrated Health & Insurance Predictor 🩺💰</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>First, predict your heart disease risk. Then, use that risk to estimate your insurance cost category.</p>", unsafe_allow_html=True)
st.markdown("---")

# Initialize session state for page control and heart_prob
if 'page' not in st.session_state:
    st.session_state.page = 'heart_disease' # Default to heart disease page

if 'heart_prob_predicted' not in st.session_state:
    st.session_state.heart_prob_predicted = None

# --- Heart Disease Prediction Page ---
if st.session_state.page == 'heart_disease':
    st.subheader("1. Heart Disease Risk Prediction")

    if heart_model is not None and heart_scaler is not None and heart_features is not None:
        with st.form("heart_disease_form"):
            hd_col1, hd_col2 = st.columns(2) # Inner columns for heart disease form

            with hd_col1:
                hd_age = st.slider("Age", 20, 100, 50, key="hd_age")
                hd_sex = st.radio("Sex", ["Male", "Female"], key="hd_sex")
                hd_cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3], key="hd_cp")
                hd_trestbps = st.slider("Resting Blood Pressure", 80, 200, 120, key="hd_trestbps")
                hd_chol = st.slider("Cholesterol", 100, 600, 200, key="hd_chol")
                hd_fbs = st.radio("Fasting Blood Sugar > 120 mg/dl?", ["Yes", "No"], key="hd_fbs")

            with hd_col2:
                hd_restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", [0, 1, 2], key="hd_restecg")
                hd_thalach = st.slider("Maximum Heart Rate Achieved", 60, 250, 150, key="hd_thalach")
                hd_exang = st.radio("Exercise Induced Angina?", ["Yes", "No"], key="hd_exang")
                hd_oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, step=0.1, key="hd_oldpeak")
                hd_slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", [0, 1, 2], key="hd_slope")
                hd_ca = st.selectbox("Number of Major Vessels Colored by Flourosopy (0-4)", [0, 1, 2, 3, 4], key="hd_ca")
                hd_thal = st.selectbox("Thal (0-3)", [0, 1, 2, 3], key="hd_thal")

            heart_submitted = st.form_submit_button("🚨 Predict Heart Disease Risk")

        if heart_submitted:
            # Prepare input for heart disease model
            input_data_hd = pd.DataFrame([[
                hd_age,
                1 if hd_sex == "Male" else 0,
                hd_cp,
                hd_trestbps,
                hd_chol,
                1 if hd_fbs == "Yes" else 0,
                hd_restecg,
                hd_thalach,
                1 if hd_exang == "Yes" else 0,
                hd_oldpeak,
                hd_slope,
                hd_ca,
                hd_thal
            ]], columns=heart_features)

            scaled_input_hd = heart_scaler.transform(input_data_hd)
            hd_pred = heart_model.predict(scaled_input_hd)[0]
            hd_prob = heart_model.predict_proba(scaled_input_hd)[0][1] # Probability of heart disease (class 1)

            st.session_state.heart_prob_predicted = hd_prob # Store the predicted probability

            st.markdown("### 🔍 Heart Disease Prediction Result")
            if hd_pred == 1:
                st.markdown(f"""
                <div class="styled-error-box">
                <h4 class="crimson-text">⚠️ High Risk of Heart Disease</h4>
                <p><strong>Probability:</strong> {hd_prob:.2%}</p>
                <p>Please consult a cardiologist immediately and consider a full health check-up.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="styled-success-box">
                <h4 class="green-text">✅ Low Risk of Heart Disease</h4>
                <p><strong>Probability:</strong> {hd_prob:.2%}</p>
                <p>Keep maintaining a healthy lifestyle! 😊</p>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("---")
    else:
        st.error("Heart disease prediction model could not be loaded. Please check file paths and data integrity.")

    # Button to navigate to the insurance page
    if st.button("Proceed to Insurance Prediction ➡️"):
        st.session_state.page = 'insurance_prediction'
        st.rerun() # Rerun to switch page

# --- Insurance Cost Prediction Page ---
elif st.session_state.page == 'insurance_prediction':
    st.subheader("2. Insurance Cost Prediction (using Heart Disease Probability)")

    # Button to navigate back to heart disease page
    if st.button("⬅️ Back to Heart Disease Prediction"):
        st.session_state.page = 'heart_disease'
        st.rerun() # Rerun to switch page
    st.markdown("---") # Separator after back button

    if insurance_model is not None and insurance_scaler is not None and insurance_features is not None and median_charge is not None:
        # Use the predicted heart disease probability, or a default if not yet predicted
        current_heart_prob = st.session_state.heart_prob_predicted if st.session_state.heart_prob_predicted is not None else 0.3 # Default for demonstration

        st.markdown(f"""
            <div class="styled-info-box">
                Using Heart Disease Probability: <strong>{current_heart_prob:.2%}</strong> (from previous prediction or default)
            </div>
        """, unsafe_allow_html=True)

        with st.form("insurance_cost_form"):
            ins_col1, ins_col2 = st.columns(2) # Inner columns for insurance form
            with ins_col1:
                ins_age = st.slider("Age", 18, 100, 30, key="ins_age")
                ins_sex = st.radio("Sex", ["male", "female"], key="ins_sex")
                ins_bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0, step=0.1, key="ins_bmi")
            with ins_col2:
                ins_children = st.slider("Number of Children", 0, 5, 1, key="ins_children")
                ins_smoker = st.radio("Do you smoke?", ["yes", "no"], key="ins_smoker")
                ins_region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"], key="ins_region")
            
            # Display heart_prob_predicted but don't allow direct user input here as it comes from step 1
            st.write(f"Heart Disease Probability (from Step 1): **{current_heart_prob:.2%}**")

            insurance_submitted = st.form_submit_button("Predict Insurance Cost") # Changed button text

        if insurance_submitted:
            # Prepare input for insurance model
            input_data_raw_ins = {
                "age": ins_age,
                "sex": ins_sex,
                "bmi": ins_bmi,
                "children": ins_children,
                "smoker": ins_smoker,
                "region": ins_region,
                "probablity_of_heart_disease": current_heart_prob
            }

            # Convert to DataFrame
            input_df_ins = pd.DataFrame([input_data_raw_ins])

            # Apply one-hot encoding consistent with training data
            input_encoded_temp = pd.get_dummies(input_df_ins, drop_first=True)

            # Reindex to align columns with the training data (insurance_features)
            input_encoded_ins = input_encoded_temp.reindex(columns=insurance_features, fill_value=0)

            # Scale numerical input features
            numerical_cols_to_scale_ins = [col for col in ['age', 'bmi', 'children', 'probablity_of_heart_disease'] if col in input_encoded_ins.columns]
            input_encoded_ins[numerical_cols_to_scale_ins] = insurance_scaler.transform(input_encoded_ins[numerical_cols_to_scale_ins])

            # Predict insurance cost (now a continuous value from RandomForestRegressor)
            predicted_charge = insurance_model.predict(input_encoded_ins)[0]
            
            st.markdown("### 🔍 Insurance Prediction Result")
            st.markdown(f"""
                <div class="styled-info-box">
                    Predicted Insurance Cost: <strong>₹{predicted_charge:,.2f}</strong>
                </div>
            """, unsafe_allow_html=True)
            
            # Determine if above/below median for the visual indicator
            if predicted_charge >= median_charge:
                st.markdown(f"""
                    <div class="styled-success-box">
                        <h4 class="green-text">Result: Likely Above Average 💹</h4>
                    </div>
                """, unsafe_allow_html=True)
            else: # Below Average
                st.markdown(f"""
                    <div class="styled-success-box">
                        <h4 class="green-text">Result: Likely Below Average ✅</h4>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.error("Insurance cost prediction model could not be loaded. Please ensure file paths and data integrity.")
