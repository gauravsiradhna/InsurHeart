## 📂 Project Structure

```text
Integrated-Health-Insurance-Predictor/
│
├── app.py
├── heart_disease_data.csv
├── insurance_with_heart_features_updated.csv
└── README.md
```

> Make sure both CSV files are present in the same directory as `app.py`.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/integrated-health-insurance-predictor.git
```

### 2. Navigate to the Project

```bash
cd integrated-health-insurance-predictor
```

### 3. Install Required Libraries

```bash
pip install streamlit pandas numpy scikit-learn
```

### 4. Run the Application

```bash
streamlit run app.py
```

Streamlit will start the application and provide a local URL such as:

```text
http://localhost:8501
```

Open the URL in your browser.

---

## 🖥️ How to Use

### Step 1 — Heart Disease Prediction

Enter the required health information and click:

```text
🚨 Predict Heart Disease Risk
```

The application will display the predicted heart disease risk and probability.

### Step 2 — Insurance Prediction

Click:

```text
Proceed to Insurance Prediction ➡️
```

Enter:

* Age
* Sex
* BMI
* Number of children
* Smoking status
* Region

The heart disease probability from Step 1 is automatically passed to the insurance model.

Click:

```text
Predict Insurance Cost
```

The application will display the estimated insurance cost.

---

## ⚠️ Disclaimer

This application is developed for **educational and demonstration purposes only**.

The heart disease prediction should **not be considered a medical diagnosis**, and the insurance prediction should not be considered an official insurance quotation.

Users should consult qualified healthcare professionals for medical decisions and insurance providers for actual insurance pricing.
