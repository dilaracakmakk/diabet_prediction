# 🩺 Diabetes Prediction Web App using Streamlit

This project is a **Streamlit-based web application** that uses machine learning to predict whether a person is diabetic or not based on health-related inputs. Users can enter their own data and get an instant prediction result.

## 🎯 Project Goal

- Predict the presence of diabetes at an early stage
- Build a simple yet effective prediction model using machine learning
- Provide a user-friendly web interface for real-time predictions

---

## 🔍 Dataset Used

- **Pima Indians Diabetes Dataset**
- [Source (Kaggle)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

### Features:

- Pregnancies  
- Glucose  
- BloodPressure  
- SkinThickness  
- Insulin  
- BMI  
- DiabetesPedigreeFunction  
- Age  
- Outcome (0 = Non-diabetic, 1 = Diabetic)

---

## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- **Streamlit** (for the web interface)  
- PIL (optional image handling)

---

## 🚀 How to Run the App

1. Install the required libraries:
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn pillow
Run the application:

bash
streamlit run streamlit_app.py

It will automatically open in your browser. If not, use the local URL provided in the terminal (e.g., http://localhost:8501).

🧠 Model Information
Model: RandomForestClassifier
Train/Test split: 80/20
Evaluation Metric: Accuracy
Accuracy: Displayed in real time on the app

🧪 Application Features
Training Data Summary: Displayed using df.describe()

Bar Chart: Overview of all feature distributions

Sidebar User Inputs:
Pregnancies
Glucose
Blood Pressure
Skin Thickness
Insulin
BMI
Diabetes Pedigree Function
Age

Results:
Predicts if the person is diabetic
Visualizes insulin levels vs. age (user vs dataset)
Shows model accuracy
Displays a result message



👩‍💻 Developer
Dilara Cakmak
