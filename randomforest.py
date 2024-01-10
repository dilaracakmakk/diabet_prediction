import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv('csv/diabetes.csv')


st.title('Diabetic or not ?')

st.subheader('Training Data')
st.write(df.describe())

st.subheader('Visualization')
st.bar_chart(df)

x= df.drop(['Outcome'], axis =1)
y=df.iloc[:,-1]

x_train, x_test, y_train, y_test, = train_test_split(x,y, test_size = 0.2,random_state = 0)

def user_report():
  Pregnancies =st.sidebar.slider('Pregnancies', 0,17, 3)
  Glucose = st.sidebar.slider('Glucose',0,200, 120)
  BloodPressure=st.sidebar.slider('Blood Pressure',0,122, 7)
  SkinThickness=st.sidebar.slider('Skin Thickness',0,100, 20)
  Insulin=st.sidebar.slider('Insulin',0,846, 79)
  BMI=st.sidebar.slider('BMI',0,67, 20)
  DiabetesPedigreeFunction=st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47)
  Age=st.sidebar.slider('Age', 21,88, 33)

  user_report ={
    'Pregnancies' :Pregnancies,
    'Glucose' :Glucose,
    'BloodPressure' :BloodPressure,
    'SkinThickness' :SkinThickness,
    'Insulin' :Insulin,
    'BMI' :BMI,
    'DiabetesPedigreeFunction' :DiabetesPedigreeFunction,
    'Age' :Age,

  }

  report_data = pd.DataFrame(user_report,index=[0])
  return report_data

user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

rf= RandomForestClassifier()
rf.fit(x_train,y_train)
user_result = rf.predict(user_data)

st.header('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['Age'], y = user_data['Insulin'], s = 150)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)

st.subheader('Accuracy:')
st.write(str(accuracy_score(y_test,rf.predict(x_test))*100)+'%')




st.subheader('YOUR RESULT: ')
output=''
if user_result[0]==0:
  output = 'YOU ARE NOT DIABETIC :)'
else:
  output = 'YOU ARE DIABETIC :('
st.title(output)
