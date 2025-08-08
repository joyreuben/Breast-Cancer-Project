# 🩺 Breast Cancer Prediction (Lightweight Demo)

This is a lightweight web app built with **Streamlit** that predicts whether a breast tumor is **benign** or **malignant**, based on a few selected numerical features extracted from the Breast Cancer Wisconsin dataset.

## 💡 About the Project

This demo app uses a **RandomForestClassifier** trained on just **6 key features** to keep the model fast and lightweight, while still maintaining reasonable predictive power. The model and scaler are saved using `joblib` and loaded for inference in the Streamlit app.

## 🚀 How It Works

1. Users input the following 6 features:
   - Mean Concave Points
   - Worst Perimeter
   - Worst Radius
   - Mean Perimeter
   - Worst Concavity
   - Mean Radius

2. Input values are scaled using a pre-fitted `StandardScaler`.

3. The scaled inputs are passed into the trained model to generate predictions.

4. The result — **Benign** or **Malignant** — is displayed instantly.

## 🧪 Technologies Used

- Python
- Scikit-learn
- Streamlit
- NumPy & Pandas
- Joblib

## 📦 Files Included

- `breast_cancer_model.pkl`: Trained RandomForestClassifier
- `scaler.pkl`: Scaler used to normalize feature input
- `app.py`: Streamlit frontend logic
- `README.md`: Project documentation

## 🖥️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/joyreuben/Breast-Cancer-Project.git
   cd breast-cancer-lightweight-demo