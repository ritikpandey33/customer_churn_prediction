# 📊 Retention Command Center

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![Accuracy](https://img.shields.io/badge/Accuracy-83%25-brightgreen)

## 🚀 Overview

The **Retention Command Center** is an AI-powered analytics dashboard designed to predict customer churn in the telecommunications industry. Built with **Streamlit** and powered by a high-performance **LightGBM** machine learning model, this application helps businesses identify at-risk customers and take proactive retention actions.

The model achieves an impressive **83% accuracy** on the Telco Customer Churn dataset, outperforming traditional baseline models.

## ✨ Key Features

- **🔮 Real-Time Prediction**: Instantly calculate churn probability for any customer.
- **📊 Interactive Dashboard**: User-friendly interface for data input and visualization.
- **📈 Visual Insights**: Gauge charts and probability breakdowns for clear risk assessment.
- **💡 Smart Recommendations**: Actionable retention strategies based on customer profiles (e.g., suggesting contract upgrades or tech support).
- **⚡ High Performance**: Optimized LightGBM model with 13 key features.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: LightGBM, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Deployment**: Ready for local or cloud deployment

## 📂 Project Structure

```
customer-churn-predictionn/
├── models/
│   └── model.pkl          # Trained LightGBM model (83% accuracy)
├── notebooks/
│   └── Telco_Customer_Churn_Prediction.ipynb  # Model training & EDA
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Pip (Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd customer-churn-predictionn
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

Launch the dashboard with a single command:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## 🧠 Model Details

The core of this project is a **LightGBM Classifier** trained on the IBM Telco Customer Churn dataset.

- **Preprocessing**:
  - Handling missing values (TotalCharges)
  - Label Encoding for categorical variables
  - SMOTE for handling class imbalance
  - Feature Selection (Top 13 features selected via Chi-Squared & ANOVA)
- **Performance**:
  - **Accuracy**: 83%
  - **ROC-AUC**: ~82%


---
*Built with ❤️ for Data Science*
