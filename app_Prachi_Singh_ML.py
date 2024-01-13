import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model=load_model(r'C:\Users\PRACHI SINGH\Desktop\MCA_BD\Stock Prediction Model Prachi Singh ML.keras')
st.header("Stock Market Predictor")
stock = st.text_input("Enter the Stock Symbol",'GOOG')

start='2012-01-01'
end='2022-12-31'
data=yf.download(stock,start,end)


st.subheader('Raw Data')
st.write(data)
data_train=pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test=pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

from  sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

past_100_days=data_train.tail(100)
data_test=pd.concat([past_100_days,data_test],ignore_index=True)
data_test_scale=scaler.fit_transform(data_test)

st.subheader('Price V/S Moving Average-50 Days')
ma_50_days=data.Close.rolling(50).mean()
fig1=plt.figure(figsize=(10,8))
plt.plot(ma_50_days,'r')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price V/S  Moving Average-50 Days V/S Moving Average-100 Days')
ma_100_days=data.Close.rolling(50).mean()
fig1=plt.figure(figsize=(10,8))
plt.plot(ma_50_days,'r')
plt.plot(ma_100_days,'r')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig1)


st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

#st.write(data)

x=[]
y=[]
for i in range(100,data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x,y=np.array(x),np.array(y)

predict= model.predict(x)
scale=1/scaler.scale_
predict=predict*scale
y=y*scale
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(6, 4))
plt.plot(predict, 'lightgreen', label='Original Price')
plt.plot(y, 'lightblue', label='Predicted Price')
plt.xlabel('TIME')
plt.ylabel('PRICE')
plt.show()
st.pyplot(fig4)

