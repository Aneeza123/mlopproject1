import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st


model =joblib.load("Models/liveModelV1.pk1")

data = pd.read_csv('mobile_price_range_data.csv')
X = data.iloc[:, :-1] #excluding the target column assuming the last column is the target
y =data.iloc[:, -1] #target column

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#make prediction for X_test set
y_pred = model.predict(X_test)

#calculate accuracy
accuracy = accuracy_score(y_test ,y_pred)
#page title
st.title("Model Accuracy and Real-Time Prediction")

#display accuracy
st.write(f'Model Accuracy:[accuracy:.2f]")

#Real-time predivtion based on user inputs
st.header('Real-Time Prediction')
input_data =[]
for col in X_test.columns:
    input_value = st.number_input(f'Input for feature {col}', value=0)
    input_data.append(input_value
    #convert input data to dataframe
    input_df = pd.DataFrame([input_data], columns=X_test.columns)
    #make prediction
    if st.button("predict"):
        prediction = model.predict(input_df)
        st.write(f'Prediction"{prediction[0]}')

