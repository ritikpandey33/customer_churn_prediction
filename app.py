import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load('models/model.pkl')
        return model_data
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please ensure 'models/model.pkl' exists.")
        st.stop()

model_artifacts = load_model()

# Extract model components (adjust based on how your model was saved)
if isinstance(model_artifacts, dict):
    model = model_artifacts.get('model')
    encoders = model_artifacts.get('encoders', {})
    scaler = model_artifacts.get('scaler')
else:
    model = model_artifacts
    encoders = {}
    scaler = None

# Header
st.markdown('<h1 class="main-header">📊 Telco Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Customer Retention Analytics | 83% Accuracy</p>', unsafe_allow_html=True)

# Sidebar - Customer Information Input
st.sidebar.header("👤 Customer Information")
st.sidebar.markdown("---")

# Demographics
st.sidebar.subheader("📋 Demographics")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

# Account Information
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Account Info")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox("Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Services
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Services")
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", 
    ["No phone service", "No", "Yes"] if phone_service == "Yes" else ["No phone service"])

internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

if internet_service != "No":
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
else:
    online_security = online_backup = device_protection = "No internet service"
    tech_support = streaming_tv = streaming_movies = "No internet service"

# Charges
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Billing")
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=5.0)
total_charges = tenure * monthly_charges

# Prediction button
predict_button = st.sidebar.button("🔮 Predict Churn")

# Main content
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.metric("📅 Tenure", f"{tenure} months")
with col2:
    st.metric("💵 Monthly Charges", f"${monthly_charges:.2f}")
with col3:
    st.metric("💰 Total Charges", f"${total_charges:.2f}")

st.markdown("---")

if predict_button:
    # Prepare input data
    input_data = {
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure columns are in the exact order the model expects
    expected_cols = [
        'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ]
    input_df = input_df[expected_cols]
    
    # Encode categorical features
    # Note: The notebook used LabelEncoder for these. We need to match that.
    # Since we don't have the original encoders, we'll recreate them or use a mapping.
    # Based on standard LabelEncoder behavior (alphabetical order):
    
    mappings = {
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'OnlineBackup': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'DeviceProtection': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'PaperlessBilling': {'No': 0, 'Yes': 1},
        'PaymentMethod': {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 'Electronic check': 2, 'Mailed check': 3}
    }

    for col, mapping in mappings.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].map(mapping)
            
    # Scale numerical features if scaler exists
    if scaler:
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Make prediction
    try:
        # LightGBM might complain about column names if they don't match exactly what it was trained with
        # (Column_0, etc.). We might need to rename them or pass as numpy array.
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        churn_prob = prediction_proba[1] * 100
        no_churn_prob = prediction_proba[0] * 100
        
        # Display results
        st.markdown("## 🎯 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = churn_prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability", 'font': {'size': 24}},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkred" if churn_prob > 50 else "darkgreen"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': 'lightgreen'},
                        {'range': [30, 70], 'color': 'yellow'},
                        {'range': [70, 100], 'color': 'lightcoral'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Prediction Breakdown")
            
            if prediction == 1:
                st.error("### ⚠️ HIGH RISK - Customer Likely to Churn")
                st.markdown(f"**Churn Probability:** {churn_prob:.1f}%")
                st.markdown(f"**Retention Probability:** {no_churn_prob:.1f}%")
            else:
                st.success("### ✅ LOW RISK - Customer Likely to Stay")
                st.markdown(f"**Retention Probability:** {no_churn_prob:.1f}%")
                st.markdown(f"**Churn Probability:** {churn_prob:.1f}%")
            
            # Risk level
            if churn_prob > 70:
                risk_level = "🔴 CRITICAL"
            elif churn_prob > 50:
                risk_level = "🟡 MODERATE"
            else:
                risk_level = "🟢 LOW"
            
            st.markdown(f"**Risk Level:** {risk_level}")
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("## 💡 Retention Recommendations")
        
        if prediction == 1:
            recommendations = []
            
            if contract == "Month-to-month":
                recommendations.append("📝 **Upgrade to Long-term Contract:** Offer incentives for 1 or 2-year contracts")
            
            if payment_method == "Electronic check":
                recommendations.append("💳 **Switch Payment Method:** Encourage automatic payment methods")
            
            if internet_service == "Fiber optic" and monthly_charges > 80:
                recommendations.append("💰 **Review Pricing:** Consider loyalty discounts or bundled services")
            
            if tech_support == "No":
                recommendations.append("🛠️ **Offer Tech Support:** Provide complimentary tech support period")
            
            if online_security == "No":
                recommendations.append("🔒 **Security Package:** Offer online security add-on at discounted rate")
            
            if not recommendations:
                recommendations.append("📞 **Personal Outreach:** Schedule call with retention specialist")
            
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("✨ **Customer is Stable!** Continue providing excellent service and monitor for changes in behavior.")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.exception(e)

else:
    # Show model info when no prediction
    st.markdown("## 🤖 About the Model")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Model Type**\n\nLightGBM Classifier")
    with col2:
        st.info("**Accuracy**\n\n83%")
    with col3:
        st.info("**Dataset**\n\n7,043 customers")
    
    st.markdown("---")
    st.markdown("""
    ### 📈 How It Works
    
    This AI-powered system analyzes customer data to predict churn probability in real-time:
    
    1. **Enter Customer Data** - Fill in the customer information in the sidebar
    2. **Click Predict** - The model processes the data using advanced machine learning
    3. **Get Insights** - Receive churn prediction with actionable retention strategies
    
    ### 🎯 Key Features
    
    - ✅ 83% prediction accuracy
    - ⚡ Real-time analysis
    - 💡 Personalized retention recommendations
    - 📊 Visual probability breakdown
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Powered by LightGBM | Built with Streamlit | © 2025 Telco Churn Analytics</p>
</div>
""", unsafe_allow_html=True)
