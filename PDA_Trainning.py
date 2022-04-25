#import libraries

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn import utils
from PIL import Image
import streamlit as st
from sklearn.metrics import r2_score

#Create a title and a sub-title
st.write("""
# PDA trainning


""")

#Open and display an image
image = Image.open('C:/Users/ricky/OneDrive/바탕 화면/PDA data trainning/pda.png')
st.image(image, caption='PDA Training Model', use_column_width=True)


#Get the data
df = pd.read_csv('C:/Users/ricky/OneDrive/바탕 화면/PDA data trainning/PDA.csv')
df = df.dropna()
input = df.drop(['Pixel_Number', 'Light_Intensity(uW/cm2)', 'Dark(mV)', 'Light(mV)', 'L-D(mV)'], axis = 1)
output_dark = df['Dark(mV)']
output_Light = df['Light(mV)']
output_L_D = df['L-D(mV)']



#Set a subheader
st.subheader('Data Information')
#Show the data as a table
st.dataframe(df)
#Show Statistics on the data
st.write(df.describe())
#show the data as a chart
#char = st.bar_chart(df)


#Split the data into independent "X" and dependent "Y" variables
X = input.values
Y1 = output_dark.values
Y2 = output_Light.values
Y3 = output_L_D.values

#Split the data set into 75% Training and 25% Testing
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X,Y1, test_size= 0.25, random_state=0)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X,Y2, test_size= 0.25, random_state=0)
X3_train, X3_test, Y3_train, Y3_test = train_test_split(X,Y3, test_size= 0.25, random_state=0)


#Get the feature input from the user
def get_user_input():

    Layout = st.sidebar.number_input('Layout: 1: normal, 2: dark, 3: V_dark', min_value= 1)
    PVDD = st.sidebar.number_input('PVDD(mV): ', min_value= 2300)
    VG = st.sidebar.number_input('VG(mV): ',min_value = 1800 )
    NWBIAS = st.sidebar.number_input('NWBIAS(mV): ', min_value= 2500)
    Tint = st.sidebar.number_input('Tint(us): ', min_value=70)
    W = st.sidebar.number_input('W(um): ', min_value= 0.25)
    L = st.sidebar.number_input('L(um): ', min_value=0.35)
    Wavelength = st.sidebar.number_input('Wavelength(nm): ', min_value=640)
    Edge = st.sidebar.number_input('Edge: 1: True, 0: False')

    #Store a dictionary into a variable
    user_data = {'Layout': Layout,
                 'PVDD(mV)': PVDD,
                 'VG(mV)': VG,
                 'NWBIAS(mV)': NWBIAS,
                 'Tint(us)': Tint,
                 'W(um)': W,
                 'L(um)': L,
                 'Wavelength(nm)': Wavelength,
                 'Edge': Edge
    }

    #Transform the data into a data Frame
    features = pd.DataFrame(user_data, index = [0])
    return features

#Stroe the user input into a variabel
user_input = get_user_input()


#Set a subheader and display the users input
st.subheader('User Input:')
st.write(user_input)


#Create and train the model
dark = LinearRegression()
dark.fit(X1_train, Y1_train)
light = LinearRegression()
light.fit(X2_train, Y2_train)
L_D = LinearRegression()
L_D.fit(X3_train, Y3_train)


predicttion1 = dark.predict(X1_test)
predicttion2 = light.predict(X2_test)
predicttion3 = L_D.predict(X3_test)


#Show the models metrics
st.subheader('Dark Level Accuracy Score:')
score1 = r2_score(Y1_test, predicttion1)*100
st.write(str("{:.2f}".format(score1)) + "%")

st.subheader('Light Level Accuracy Score:')
score2 = r2_score(Y2_test, predicttion2)*100
st.write(str("{:.2f}".format(score2)) + "%")

st.subheader('L-D Level Accuracy Score:')
score3 = r2_score(Y3_test, predicttion3)*100
st.write(str("{:.2f}".format(score3)) + "%")


#Store the models prediction in a variabel
dark_prediction = dark.predict(user_input)
light_prediction = light.predict(user_input)
L_D_prediction = L_D.predict(user_input)





#Set a subheader and display the classification
st.subheader('Dark(mV): ')
st.write(dark_prediction)
st.subheader('Light(mV): ')
st.write(light_prediction)
st.subheader('L-D(mV): ')
st.write(L_D_prediction)



