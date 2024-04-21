import tkinter as tk
    
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import Sequential
import cv2
from PIL import Image, ImageTk
from tkinter import *      
import tkinter

from tkinter import *
from tkinter import filedialog
import os


from PIL import Image
from numpy import asarray

import numpy as np
from skimage import color
from skimage import io
from tkinter.filedialog import askopenfilename
import cv2

import os
import matplotlib.pyplot as plt 
import cv2
from PIL import ImageTk, Image
from cv2 import *
import random
from PIL import Image as im


root = tk.Tk()

root.geometry("900x470")

root.resizable(width=True, height=True)

root.title(" Water Quality Prediction ")

root['bg']='bisque'


img = None
resized_image = None

canvas = Canvas(root, width=1000, height=1000)

canvas.pack()  
img = ImageTk.PhotoImage(Image.open("C:\\Projects\\Data_Projects\\Water-Quality\\Images\\1.png"))  
canvas.create_image(2, 20, anchor=NW, image=img)

def startt():
    #============================= IMPORT LIBRARIES =============================

    import pandas as pd
    from sklearn import preprocessing
    import warnings
    warnings.filterwarnings("ignore")
    
    #============================= DATA SELECTION ==============================
    dataframe=pd.read_csv("C:\\Projects\\Data_Projects\\Water-Quality\\Dataset\\water_potability.csv")
    
    print("----------------------------------------------------")
    print("Input Data          ")
    print("----------------------------------------------------")
    print()
    print(dataframe.head(20))
    
    #============================= PREPROCESSING ==============================
    
    #==== checking missing values ====
    
    print("----------------------------------------------------")
    print("Before checking Missing Values          ")
    print("----------------------------------------------------")
    print()
    print(dataframe.isnull().sum())
    
    
    print("----------------------------------------------------")
    print("After checking Missing Values          ")
    print("----------------------------------------------------")
    print()
    dataframe=dataframe.fillna(dataframe.mean())
    print(dataframe.isnull().sum())
    
    #========================= DATA SPLITTING ==============================
    
    print("----------------------------------------------------")
    print("Data Splitting          ")
    print("----------------------------------------------------")
    print()
    
    from sklearn.model_selection import train_test_split
    
    X = dataframe.drop('Potability', axis=1)
    y = dataframe['Potability']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, test_size=0.3, random_state=100)
    
    print("Total no of data's       :",dataframe.shape[0])
    print()
    print("Total no of Train data's :",X_train.shape[0])
    print()
    print("Total no of Test data's  :",X_test.shape[0])
    
    #1-- safe and 0-- not safe
    
    
    #========================= CLASSIFICATION ==============================
    
    # ==== CNN WITH LSTM ====
    
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Conv1D,MaxPooling1D
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import  Dropout
    from tensorflow.keras.models import Sequential
    import matplotlib.pyplot as plt
    from tensorflow.keras.layers import Activation
    
    import numpy as np
    Xx=np.expand_dims(X_train, axis=2)
    Yy=np.expand_dims(y_train,axis=1)
    
    nb_out = 1
    
    
    print("----------------------------------------------------")
    print("Hybrid CNN With LSTM          ")
    print("----------------------------------------------------")
    print()
    
    
    # Initialize the layer 
    model = Sequential()
    
    # LSTM layer
    model.add(LSTM(input_shape=(9,1), units=100, return_sequences=True))
    model.add(Dropout(0.2))
    
    # CNN layer
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # LSTM layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # CNN layer
    model.add(Dense(units=nb_out))
    model.add(Activation("linear"))
    
    # Compile layer
    model.compile(loss="mae", optimizer="adam", metrics=['Accuracy','mae','mse'])
    print(model.summary())
    
    
    # Fit the layer
    his_lstm=model.fit(Xx, Yy, epochs=3, batch_size=32, validation_split=0.1, verbose=1)
    
    
    mae_lstm =model.evaluate(Xx, Yy, verbose=2)[2]
    
    mse_lstm =model.evaluate(Xx, Yy, verbose=2)[3]
    
    
    y_pred1 = model.predict(Xx)
    
    y_pred11 = (y_pred1 > 0.3)
    
    y_pred11=y_pred11.astype('uint8')
    
    # from sklearn import metrics
    
    # mae=metrics.mean_absolute_error(y_pred11,Yy)
    
    # mse=metrics.mean_squared_error(y_pred11,Yy)
    
    import math
    rsme_lstm = math.sqrt(mse_lstm)
    
    # mape=metrics.mean_absolute_percentage_error(y_pred11,Yy)
    
    print("----------------------------------------------------")
    print("Performance Analysis ----> CNN with LSTM         ")
    print("----------------------------------------------------")
    print()
    print()
    print("1. Mean Absolute Error :", mae_lstm)
    print()
    print("2. Mean Squared Error :", mse_lstm)
    print()
    print("3. Root Mean Squared Error :", rsme_lstm)
    
    
    # ==== CNN WITH GRU ====
    
    from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU
    from keras.optimizers import SGD
    from tensorflow.keras.models import Sequential
    
    
    print("----------------------------------------------------")
    print("Hybrid CNN With GRU          ")
    print("----------------------------------------------------")
    print()
    
    from tensorflow.keras.layers import Flatten
    
    model2 = Sequential()
    
    model2.add(Conv1D(filters=256, kernel_size=1, activation='relu', input_shape = (Xx.shape[1], Xx.shape[2])))
    
    model2.add(GRU(units = 256, activation = 'relu',return_sequences=True, input_shape = (Xx.shape[1], Xx.shape[2])))
    
    model2.add(GRU(units = 256, activation = 'relu',return_sequences=True, input_shape = (Xx.shape[1], Xx.shape[2])))
    
    model2.add(Dense(units = 1))
    
    model2.add(Flatten())
    
    model2.compile(optimizer = 'adam', loss = 'mae',metrics=['mae','mse'])
    
    model2.summary()
    
    his_gru=model2.fit(Xx,Yy,epochs=20,batch_size=16,verbose=1)
    
    
    mae_gru =model2.evaluate(Xx, Yy, verbose=2)[1]
    
    mse_gru =model2.evaluate(Xx, Yy, verbose=2)[2]
    
    
    y_pred1_gru = model2.predict(Xx)
    
    y_pred11_gru = (y_pred1_gru > 0.3)
    
    y_pred11_gru=y_pred11.astype('uint8')
    
    
    # mae_gru=metrics.mean_absolute_error(y_pred11_gru,Yy)
    
    # mse_gru=metrics.mean_squared_error(y_pred11_gru,Yy)
    
    import math
    rsme_gru = math.sqrt(mse_gru)
    
    # mape=metrics.mean_absolute_percentage_error(y_pred11,Yy)
    
    print("----------------------------------------------------")
    print("Performance Analysis ----> CNN with GRU         ")
    print("----------------------------------------------------")
    print()
    print()
    print("1. Mean Absolute Error :", mae_gru)
    print()
    print("2. Mean Squared Error :", mse_gru)
    print()
    print("3. Root Mean Squared Error :", rsme_gru)
    
    
    
    # ==== LOGISTIC REGRESSION ====
    
    
    from sklearn import linear_model
    
    lr=linear_model.LogisticRegression(random_state = 50)
    
    # lr.fit(X,y)
    
    lr.fit(X_train,y_train)
    
    pred_lr=lr.predict(X_train)
    
    from sklearn import metrics
    # acc=metrics.accuracy_score(pred_lr,y)
    
    mae_lr=metrics.mean_absolute_error(pred_lr,y_train)
    
    
    mse_lr=metrics.mean_squared_error(pred_lr,y_train)
    
    
    rsme_lr = math.sqrt(mse_lr)
    
    
    print("----------------------------------------------------")
    print("Performance Analysis ----> Logistic Regression      ")
    print("----------------------------------------------------")
    print()
    print()
    print("1. Mean Absolute Error :", mae_lr)
    print()
    print("2. Mean Squared Error :", mse_lr)
    print()
    print("3. Root Mean Squared Error :", rsme_lr)
    
    
    #=========================== PREDICTION =================================
    
    
    print("----------------------------------------------------")
    print("Prediction ----> Water Quality      ")
    print("----------------------------------------------------")
    print()
    print()
#    import cv2
    for i in range(0,10):
        if pred_lr[i]==1:
            print("-------------------------------------")
            print()
            print([i],"Safe ( Water Quality is Good)")
            img = cv2.imread("safe.jpg")
            plt.imshow(img)
            plt.title('Safe')
            plt.axis ('off')
            plt.show()
        else:
            print("-------------------------------------")
            print()
            print([i],"Not Safe ( Water Quality is Not Good)")        
            img = cv2.imread("unsafe.jpg")
            plt.imshow(img)
            plt.title('Unsafe')
            plt.axis ('off')
            plt.show()            
    
    
    print()
    print("-------------------------------------------------------------------")
    print()
    
    #=========================== PREDICTION =================================
    
    # ==== COMPARISON GRAPH =====
    
    print()
    print("Error Value Must Be Low")
    print()
    
    
    objects = ('CNN with LSTM', 'CNN with GRU', 'Logistic Regression')
    y_pos = np.arange(len(objects))
    performance = [mae_lstm,mae_gru,mae_lr]
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Performance ')
    plt.title('Comparison Graph -- Error Values')
    plt.show()

#========= FUNCTION FOR QUIT =============

    
def Close():
    root.destroy()

#============ USERNAME ===========

lbl = Label(root, text=" Water Quality Prediction Using Deep Learning:",font=('Century 20'))
lbl.pack(side=tk.TOP)
lbl.place(x=100, y=50)




#============ SET INPUT ===========


btn = tk.Button(root, text='CLICK HERE ( GET START)',width=35, command=startt,fg='black', bg='light blue')
# .pack()
btn.pack(side=tk.TOP)
btn.place(x=350, y=150)


root.mainloop()
