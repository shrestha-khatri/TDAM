import numpy as np
import time
import csv

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU, TimeDistributed,Lambda,Multiply,GlobalAveragePooling1D,Input,Reshape,Softmax,RepeatVector
from tensorflow.keras.layers import Multiply,Permute
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
from IPython.core.pylabtools import figsize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
import random as rn

np.random.seed(2017)
rn.seed(12345)
tf.random.set_seed(1234)

def get_data(gen = "train", speed_list=None, area_list=None, density_list=None, temp_list=None,sigma_B_list = None,
             pressure_list=None,ICME_list=None,sequence_length=None, predict_length=None, area_length=None, area_length_two=None):
    if gen == "train":
        X_train_speed = []
        X_train_area = []
        X_train_density = []
        X_train_temp = []
        X_train_sigma_B = []
        X_train_pressure = []
        X_train_ICME = []
        y_train_speed = []
        print(len(speed_list),len(area_list),len(density_list),len(sigma_B_list),len(pressure_list),len(ICME_list))
        for index in range(len(speed_list) - sequence_length - predict_length + 1):
            X_train_speed.append(speed_list[index: index + sequence_length]) 
            y_train_speed.append(speed_list[index + sequence_length + predict_length - 1]) 
            X_train_area.append(area_list[index + area_length - area_length_two: index + area_length + area_length_two])  
            X_train_density.append(density_list[index: index + sequence_length])
            X_train_temp.append(temp_list[index: index + sequence_length])
            X_train_sigma_B.append(sigma_B_list[index: index + sequence_length])
            X_train_pressure.append(pressure_list[index: index + sequence_length])
            X_train_ICME.append(ICME_list[index: index + sequence_length])

        print(len(X_train_speed[0]), len(X_train_area[0]), type(X_train_area), type(X_train_area[0]))
        print(len(X_train_area))
        tmp = np.zeros(shape=[len(X_train_area), sequence_length - len(X_train_area[0]), 1])
        print(tmp.shape, type(tmp))
        X_train_area = np.hstack((X_train_area, tmp))
        print("%", X_train_area)

        X_train_speed = np.array(X_train_speed)
        print(X_train_speed.shape)  
        X_train_speed = np.reshape(X_train_speed, (X_train_speed.shape[0], X_train_speed.shape[1]))
        print(X_train_speed.shape)   
        y_train_speed = np.array(y_train_speed) 
        print(y_train_speed.shape)  
        
      
        print(X_train_speed, y_train_speed)

        X_train_area = np.array(X_train_area)
        print(X_train_area.shape) 
        X_train_area = np.reshape(X_train_area, (X_train_area.shape[0], X_train_area.shape[1]))
        print(X_train_area.shape)
        print(X_train_area)

        X_train_density = np.array(X_train_density)
        print(X_train_density.shape)
        X_train_density = np.reshape(X_train_density, (X_train_density.shape[0], X_train_density.shape[1]))
        print(X_train_density.shape)

        X_train_temp = np.array(X_train_temp)
        print(X_train_temp.shape)
        X_train_temp = np.reshape(X_train_temp, (X_train_temp.shape[0],X_train_temp.shape[1]))
        print(X_train_temp.shape)

        X_train_sigma_B = np.array(X_train_sigma_B)
        X_train_sigma_B = np.reshape(X_train_sigma_B,(X_train_sigma_B.shape[0], X_train_sigma_B.shape[1]))
        print(X_train_sigma_B.shape)

        X_train_pressure = np.array(X_train_pressure)
        X_train_pressure = np.reshape(X_train_pressure,(X_train_pressure.shape[0],X_train_pressure.shape[1]))
        print(X_train_pressure.shape)

        X_train_ICME = np.array(X_train_ICME)
        X_train_ICME = np.reshape(X_train_ICME,(X_train_ICME.shape[0],X_train_ICME.shape[1]))
        print(X_train_ICME.shape)
        return [X_train_speed, X_train_area, X_train_density, X_train_temp, X_train_sigma_B, X_train_pressure, X_train_ICME,y_train_speed]
    if gen == "val":
        print("val_data!!!!!")
        X_val_speed = []
        X_val_area = []
        X_val_density = []
        X_val_temp = []
        X_val_sigma_B = []
        X_val_pressure = []
        X_val_ICME = []
        y_val_speed = []
        for index in range(len(speed_list) - sequence_length - predict_length + 1):
            X_val_speed.append(speed_list[index: index + sequence_length])  
            y_val_speed.append(speed_list[index + sequence_length + predict_length - 1])  
            X_val_area.append(area_list[index + area_length - area_length_two : index + area_length + area_length_two]) 
            X_val_temp.append(temp_list[index: index + sequence_length])
            X_val_sigma_B.append(sigma_B_list[index: index + sequence_length])
            X_val_pressure.append(pressure_list[index: index + sequence_length])
            X_val_ICME.append(ICME_list[index: index + sequence_length])

        print(len(X_val_speed[0]), len(X_val_area[0]), type(X_val_area), type(X_val_area[0]))
        print(len(X_val_area))
        tmp = np.zeros(shape=[len(X_val_area), sequence_length - len(X_val_area[0]), 1])
        print(tmp.shape, type(tmp))
        X_val_area = np.hstack((X_val_area, tmp))
        print("%", X_val_area)

        X_val_speed = np.array(X_val_speed)
        print(X_val_speed.shape)  
        X_val_speed = np.reshape(X_val_speed, (X_val_speed.shape[0], X_val_speed.shape[1]))
        print(X_val_speed.shape) 
        y_val_speed = np.array(y_val_speed) 
        print(y_val_speed.shape) 
      
        print(X_val_speed, y_val_speed)

        X_val_area = np.array(X_val_area)
        print(X_val_area.shape)  
        X_val_area = np.reshape(X_val_area, (X_val_area.shape[0], X_val_area.shape[1]))
        print(X_val_area.shape)
        print(X_val_area)

        X_val_density = np.array(X_val_density)
        print(X_val_density.shape)
        X_val_density = np.reshape(X_val_density, (X_val_density.shape[0], X_val_density.shape[1]))
        print(X_val_density.shape)

        X_val_temp = np.array(X_val_temp)
        print(X_val_temp.shape)
        X_val_temp = np.reshape(X_val_temp, (X_val_temp.shape[0], X_val_temp.shape[1]))
        print(X_val_temp.shape)

        X_val_sigma_B = np.array(X_val_sigma_B)
        X_val_sigma_B = np.reshape(X_val_sigma_B, (X_val_sigma_B.shape[0], X_val_sigma_B.shape[1]))
        print(X_val_sigma_B.shape)

        X_val_pressure = np.array(X_val_pressure)
        X_val_pressure = np.reshape(X_val_pressure, (X_val_pressure.shape[0],X_val_pressure.shape[1]))
        print(X_val_pressure.shape)

        X_val_ICME = np.array(X_val_ICME)
        X_val_ICME = np.reshape(X_val_ICME,(X_val_ICME.shape[0],X_val_ICME.shape[1]))
        print(X_val_ICME.shape)
        return [X_val_speed, X_val_area, X_val_density, X_val_temp, X_val_sigma_B, X_val_pressure, X_val_ICME,y_val_speed]
def plotmse(epoch,hist,file_dir,i):
	epochss=np.linspace(1,epoch,epoch)
	plt.figure(i)
	plt.plot(epochss,hist.history['loss'],label='loss')
	plt.plot(epochss,hist.history['mean_squared_error'],label='mean_squared_error')
	plt.plot(epochss,hist.history['mean_absolute_error'],label='mean_absolute_error')
	plt.plot(epochss,hist.history['val_loss'],label='val_loss')
	plt.plot(epochss,hist.history['val_mean_squared_error'],label='val_mean_squared_error')
	plt.plot(epochss,hist.history['val_mean_absolute_error'],label='val_mean_absolute_error')
	plt.legend(loc='upper right',fontsize=10)
	plt.savefig(os.path.join(file_dir, 'mse_mae.jpg'))
def plottruepredict(true,predict,time,file_dir,i):
    figsize(20,5)
    true_list = np.array(true, dtype='float64')
    predict_list = np.array(predict, dtype='float64')
    print(predict_list.shape[0], true_list.shape[0])
    epochss = np.linspace(1, predict_list.shape[0], predict_list.shape[0])
    plt.figure(i+1)
    print(predict_list[0:predict_list.shape[0]])
    print(true_list[0:true_list.shape[0]])
    plt.plot(epochss, predict_list[0:predict_list.shape[0]], color='magenta', label='predict', linewidth=1)
    plt.plot(epochss, true_list[0:true_list.shape[0]], color='cyan', label='true', linewidth=1)
    plt.legend(loc='upper right', fontsize=10)
    plt.title('True_Predict')
    plt.savefig(os.path.join(file_dir, 'true_predict.jpg'))

class LossHistory(Callback):
 
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

def computecc(targets, outputs):
   
    print("*",targets.shape, outputs.shape)
    xBar = targets.mean()
    yBar = outputs.mean()
    print(xBar,yBar)
    SSR = 0
    varX = 0  
    varY = 0  
    for i in range(0, targets.shape[0]):
        diffXXBar = targets[i] - xBar
        diffYYBar = outputs[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2
  
    SST = np.sqrt(varX * varY)
  
    xxx = SSR / SST
   
    return xxx
def build_model(X_train):
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    print(inputs.shape)
    dense = Dense(3,activation='tanh')(inputs)
    print(dense.shape)

    x = keras.layers.GlobalAveragePooling1D()(dense)
    print(x.shape)
    x=Softmax()(x)
    print(x.shape)
    feature=RepeatVector(X_train.shape[1])(x)
    print(feature.shape)

    L=Multiply()([dense,feature])
    print(L.shape)

    data = Permute((2,1))(dense)
    print(data.shape)
    data = GlobalAveragePooling1D()(data)
    print(data.shape)
    data = Softmax()(data)
    print(data.shape)
    data = RepeatVector(L.shape[2])(data)
    print(data.shape) 
    data = Permute((2,1))(data)
    print(data.shape)

    F = Multiply()([L,data])
    print(F.shape)

    final = GRU(32, activation='tanh', return_sequences=False)(F)
    final = Dense(1)(final)
    model = Model(inputs = inputs,outputs= final)
    return model
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.learning_rate
    return lr

def run_network(model=None, data=None, batch_size = None, epoch=None,sequence_length=None,predict_length=None,
                path=None,i=None,checkpoint_dir=None, area_length=None,area_length_two=None):
    print("@")
    print(path)
    if data is None:
        print('Loading data... ')
        speed_train = open(r"C:\Users\DELL\my tdam\C\speed_train.txt")
        area_train = open(r"C:\Users\DELL\my tdam\C\area_train_15.txt")
        density_train = open(r"C:\Users\DELL\my tdam\C\density_train.txt")
        temp_train = open(r"C:\Users\DELL\my tdam\C\temp_train.txt")
        sigma_B_train = open(r"C:\Users\DELL\my tdam\C\sigma_B_train.txt")
        pressure_train = open(r"C:\Users\DELL\my tdam\C\pressure_train.txt")
        ICME_train = open(r"C:\Users\DELL\my tdam\C\ICME_train.txt")
        speed_line = speed_train.readline()
        area_line = area_train.readline()
        density_line = density_train.readline()
        temp_line = temp_train.readline()
        sigma_B_line = sigma_B_train.readline()
        pressure_line = pressure_train.readline()
        ICME_line = ICME_train.readline()
        train_speed_list = []
        train_area_list = []
        train_density_list = []
        train_temp_list = []
        train_sigma_B_list = []
        train_pressure_list = []
        train_ICME_list = []
        while speed_line:
            speed = list(map(float, speed_line.split()))
            train_speed_list.append(speed)
            speed_line = speed_train.readline()
        while area_line:
            area = list(map(float, area_line.split()))
            train_area_list.append(area)
            area_line = area_train.readline()
        while density_line:
            density = list(map(float, density_line.split()))
            train_density_list.append(density)
            density_line = density_train.readline()
        while temp_line:
            temp = list(map(float, temp_line.split()))
            train_temp_list.append(temp)
            temp_line = temp_train.readline()
        while sigma_B_line:
            sigma_B = list(map(float, sigma_B_line.split()))
            train_sigma_B_list.append(sigma_B)
            sigma_B_line = sigma_B_train.readline()
        while pressure_line:
            pressure = list(map(float, pressure_line.split()))
            train_pressure_list.append(pressure)
            pressure_line = pressure_train.readline()
        while ICME_line:
            ICME = list(map(float,ICME_line.split()))
            train_ICME_list.append(ICME)
            ICME_line = ICME_train.readline()
        train_speed_list = np.array(train_speed_list, dtype='float64')
        train_area_list = np.array(train_area_list, dtype='float64')
        train_density_list = np.array(train_density_list, dtype="float64")
        train_temp_list = np.array(train_temp_list, dtype='float64')
        train_sigma_B_list = np.array(train_sigma_B_list,dtype='float64')
        train_pressure_list = np.array(train_pressure_list,dtype='float64')
        train_ICME_list = np.array(train_ICME_list,dtype="float64")
        speed_train.close()
        area_train.close()
        density_train.close()
        temp_train.close()
        sigma_B_train.close()
        pressure_train.close()
        ICME_train.close()
        speed_val = open(r"C:\Users\DELL\my tdam\C\speed_val(2016).txt")
        area_val = open(r"C:\Users\DELL\my tdam\C\area_val(2016)_15.txt")
        density_val = open(r"C:\Users\DELL\my tdam\C\density_val(2016).txt")
        temp_val = open(r"C:\Users\DELL\my tdam\C\temp_val(2016).txt")
        sigma_B_val = open(r"C:\Users\DELL\my tdam\C\sigma_B_val(2016).txt")
        pressure_val = open(r"C:\Users\DELL\my tdam\C\pressure_val(2016).txt")
        ICME_val = open(r"C:\Users\DELL\my tdam\C\ICME_val(2016).txt")
        speed_line_val = speed_val.readline()
        area_line_val = area_val.readline()
        density_line_val = density_val.readline()
        temp_line_val = temp_val.readline()
        sigma_B_line_val = sigma_B_val.readline()
        pressure_line_val = pressure_val.readline()
        ICME_line_val = ICME_val.readline()
        val_speed_list = []
        val_area_list = []
        val_density_list = []
        val_temp_list = []
        val_sigma_B_list = []
        val_pressure_list = []
        val_ICME_list = []
        while speed_line_val:
            speed = list(map(float, speed_line_val.split()))
            val_speed_list.append(speed)
            speed_line_val = speed_val.readline()
        while area_line_val:
            area = list(map(float, area_line_val.split()))
            val_area_list.append(area)
            area_line_val = area_val.readline()
        while density_line_val:
            density = list(map(float,density_line_val.split()))
            val_density_list.append(density)
            density_line_val = density_val.readline()
        while temp_line_val:
            temp = list(map(float, temp_line_val.split()))
            val_temp_list.append(temp)
            temp_line_val = temp_val.readline()
        while sigma_B_line_val:
            sigma_B = list(map(float, sigma_B_line_val.split()))
            val_sigma_B_list.append(sigma_B)
            sigma_B_line_val = sigma_B_val.readline()
        while pressure_line_val:
            pressure = list(map(float,pressure_line_val.split()))
            val_pressure_list.append(pressure)
            pressure_line_val = pressure_val.readline()
        while ICME_line_val:
            ICME = list(map(float,ICME_line_val.split()))
            val_ICME_list.append(ICME)
            ICME_line_val = ICME_val.readline()
        val_speed_list = np.array(val_speed_list, dtype='float64')
        val_area_list = np.array(val_area_list, dtype='float64')
        val_density_list = np.array(val_density_list, dtype="float64")
        val_temp_list = np.array(val_temp_list,dtype='float64')
        val_sigma_B_list = np.array(val_sigma_B_list,dtype='float64')
        val_pressure_list = np.array(val_pressure_list,dtype='float64')
        val_ICME_list = np.array(val_ICME_list,dtype='float64')
        speed_val.close()
        area_val.close()
        density_val.close()
        temp_val.close()
        sigma_B_val.close()
        pressure_val.close()
        ICME_val.close()
        print(train_speed_list, train_area_list, train_density_list, train_temp_list,train_sigma_B_list,train_pressure_list,
              val_speed_list, val_area_list, val_density_list, val_temp_list,val_sigma_B_list, val_pressure_list,
              train_speed_list.shape, train_area_list.shape, train_density_list.shape, train_temp_list.shape,
              train_sigma_B_list.shape ,train_pressure_list.shape,train_ICME_list.shape,
              val_speed_list.shape, val_area_list.shape, val_density_list.shape, val_temp_list.shape,
              val_sigma_B_list.shape, val_pressure_list.shape,val_ICME_list.shape)  
        X_train_speed, X_train_area, X_train_density, X_train_temp, X_train_sigma_B, X_train_pressure, X_train_ICME,\
        y_train_speed = \
            get_data(
            gen='train', speed_list=train_speed_list, area_list=train_area_list, density_list=train_density_list,
            temp_list=train_temp_list, sigma_B_list = train_sigma_B_list, pressure_list = train_pressure_list,
            ICME_list=train_ICME_list,sequence_length=sequence_length, predict_length=predict_length,
            area_length=area_length, area_length_two=area_length_two)  

        X_val_speed, X_val_area, X_val_density, X_val_temp, X_val_sigma_B, X_val_pressure,X_val_ICME,y_val_speed = \
            get_data(
            gen='val', speed_list=val_speed_list, area_list=val_area_list, density_list=val_density_list,
            temp_list=val_temp_list, sigma_B_list = val_sigma_B_list, pressure_list = val_pressure_list,
            ICME_list=val_ICME_list,sequence_length=sequence_length, predict_length=predict_length,
            area_length=area_length, area_length_two=area_length_two)

        print("train", X_train_speed, X_train_area, X_train_density, X_train_temp,
              X_train_sigma_B, X_train_pressure, X_train_ICME,y_train_speed)  
        print("val", X_val_speed, X_val_area, X_val_density, X_val_temp,
              X_val_sigma_B, X_val_pressure,X_val_ICME,y_val_speed) 


        s1 = StandardScaler()
        s2 = StandardScaler()
        s3 = StandardScaler()
        s4 = StandardScaler()
        s10 = StandardScaler()
        s11 = StandardScaler()
        s12 = StandardScaler()
        s15 = StandardScaler()
        X_train_speed = s1.fit_transform(X_train_speed)
        X_train_area = s2.fit_transform(X_train_area)
        y_train_speed = s3.fit_transform(y_train_speed)
        X_train_temp = s4.fit_transform(X_train_temp)
        X_train_density = s10.fit_transform(X_train_density)
        X_train_sigma_B = s11.fit_transform(X_train_sigma_B)
        X_train_pressure = s12.fit_transform(X_train_pressure)
        X_train_ICME = s15.fit_transform(X_train_ICME)

        s5 = StandardScaler()
        s6 = StandardScaler()
        s7 = StandardScaler()
        s8 = StandardScaler()
        s9 = StandardScaler()
        s13 = StandardScaler()
        s14 = StandardScaler()
        s16 = StandardScaler()
        X_val_speed = s5.fit_transform(X_val_speed)
        X_val_area = s6.fit_transform(X_val_area)
        y_val_speed = s7.fit_transform(y_val_speed)
        X_val_temp = s8.fit_transform(X_val_temp)
        X_val_density = s9.fit_transform(X_val_density)
        X_val_sigma_B = s13.fit_transform(X_val_sigma_B)
        X_val_pressure = s14.fit_transform(X_val_pressure)
        X_val_ICME = s16.fit_transform(X_val_ICME)

        print("#")
        print(X_train_speed.shape, X_train_area.shape, X_train_density.shape, X_train_temp.shape,
              X_train_sigma_B.shape, X_train_pressure.shape, X_train_ICME.shape, y_train_speed.shape)  
        print(X_val_speed.shape, X_val_area.shape, X_val_density.shape, X_val_temp.shape,
              X_val_sigma_B.shape, X_val_pressure.shape, X_val_ICME.shape, y_val_speed.shape)    
        print(type(X_train_speed), type(X_train_area), type(X_train_density), type(X_train_temp), type(X_val_sigma_B),
              type(X_train_pressure),type(X_train_ICME), type(y_train_speed))
        print(type(X_val_speed), type(X_val_area), type(X_val_density), type(X_val_temp), type(X_val_sigma_B),
              type(X_val_pressure),type(X_val_ICME),type(y_val_speed))
    else:
        X_train_speed, X_train_area, X_train_density, X_train_temp, X_train_sigma_B, X_train_pressure,X_train_ICME,\
        y_train_speed,
        X_val_speed, X_val_area, X_val_density, X_val_temp, X_val_sigma_B, X_val_pressure,X_val_ICME, y_val_speed = data
    print('\nData Loaded. Compiling...\n')
    if model is None:
        X_train_speed = np.reshape(X_train_speed,(X_train_speed.shape[0],X_train_speed.shape[1],1))
        X_train_area = np.reshape(X_train_area,(X_train_area.shape[0],X_train_area.shape[1],1))
        X_train_density = np.reshape(X_train_density, (X_train_density.shape[0], X_train_density.shape[1],1))
        X_train_temp = np.reshape(X_train_temp, (X_train_temp.shape[0], X_train_temp.shape[1],1))
        X_train_sigma_B = np.reshape(X_train_sigma_B, (X_train_sigma_B.shape[0], X_train_sigma_B.shape[1],1))
        X_train_pressure = np.reshape(X_train_pressure, (X_train_pressure.shape[0], X_train_pressure.shape[1],1))
        X_train_ICME = np.reshape(X_train_ICME,(X_train_ICME.shape[0],X_train_ICME.shape[1],1))
        X_train_speed = K.variable(np.array(X_train_speed))
        print("*",type(X_train_speed),X_train_speed.shape)
        X_train_area = K.variable(np.array(X_train_area))
        print("*",type(X_train_area),X_train_area.shape)
        X_train_density = K.variable(np.array(X_train_density))
        print("*",type(X_train_density),X_train_density.shape)
        X_train_temp = K.variable(np.array(X_train_temp))
        print("*",type(X_train_temp),X_train_temp.shape)
        X_train_sigma_B = K.variable(np.array(X_train_sigma_B))
        print("*",type(X_train_sigma_B),X_train_sigma_B.shape)
        X_train_pressure = K.variable(np.array(X_train_pressure))
        print("*",type(X_train_pressure),X_train_pressure.shape)
        X_train_ICME = K.variable(np.array(X_train_ICME))
        print("*",type(X_train_ICME),X_train_ICME.shape)
        X_train = K.concatenate((X_train_speed, X_train_area,X_train_density,X_train_temp,X_train_sigma_B,X_train_pressure,X_train_ICME),axis=-1)
        print("*",type(X_train),X_train.shape)
        y_train = y_train_speed

        X_val_speed = np.reshape(X_val_speed, (X_val_speed.shape[0], X_val_speed.shape[1], 1))
        X_val_area = np.reshape(X_val_area, (X_val_area.shape[0], X_val_area.shape[1], 1))
        X_val_density = np.reshape(X_val_density, (X_val_density.shape[0], X_val_density.shape[1],1))
        X_val_temp = np.reshape(X_val_temp,(X_val_temp.shape[0], X_val_temp.shape[1],1))
        X_val_sigma_B = np.reshape(X_val_sigma_B,(X_val_sigma_B.shape[0],X_val_sigma_B.shape[1],1))
        X_val_pressure = np.reshape(X_val_pressure, (X_val_pressure.shape[0],X_val_pressure.shape[1],1))
        X_val_ICME = np.reshape(X_val_ICME,(X_val_ICME.shape[0],X_val_ICME.shape[1],1))
        X_val_speed = K.variable(np.array(X_val_speed))
        X_val_area = K.variable(np.array(X_val_area))
        X_val_density = K.variable(np.array(X_val_density))
        X_val_temp = K.variable(np.array(X_val_temp))
        X_val_sigma_B = K.variable(np.array(X_val_sigma_B))
        X_val_pressure = K.variable(X_val_pressure)
        X_val_ICME = K.variable(X_val_ICME)
        X_val = K.concatenate([X_val_speed, X_val_area, X_val_density, X_val_temp, X_val_sigma_B, X_val_pressure,X_val_ICME],axis=-1)
        y_val = y_val_speed

        X_train = K.eval(X_train)
        X_val = K.eval(X_val)
        print(type(X_train),type(y_train))
        print("#", X_train.shape, y_train.shape, X_val.shape, y_val.shape)  
        print(X_train,y_train,X_val,y_val)

        model = build_model(X_train)
        if os.path.exists(checkpoint_dir):
            print('INFO:checkpoint exists, Load weights from %s\n' % checkpoint_dir)
            model.load_weights(checkpoint_dir)
            weight_Dense_2,bias_Dense_2 = model.get_layer('dense_1').get_weights()
            print(weight_Dense_2) 
            print(bias_Dense_2)
            print(weight_Dense_2.shape)    
            print(bias_Dense_2.shape)   
            print("checkpoint_loaded")
        print(model.summary())
        weight_gru_1, bias_gru_1 = model.get_layer('dense_1').get_weights()
        print("@")
        print(weight_gru_1)
        print(bias_gru_1)
        print(weight_gru_1.shape)  
        print(bias_gru_1.shape)
        print("model_par ok")
        print(model.summary())
        adam = optimizers.Adam(learning_rate=0.01, clipvalue=5.0)
        lr_metric = get_lr_metric(adam)
        model.compile(loss='mean_squared_error', optimizer=adam,
                      metrics=['mean_squared_error', 'mean_absolute_error', lr_metric])
        historyss = LossHistory()
        tb = TensorBoard(log_dir='./logs/model_3', histogram_freq=0, write_graph=True, write_images=False)

        checkpointer = ModelCheckpoint(os.path.join(path, 'lstm.{epoch:02d}-{loss:.4f}-{val_loss:.4f}.h5'), save_best_only=False)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto')
        print("$",X_train.shape,y_train.shape,X_val.shape,y_val.shape)
        hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch, validation_data=[X_val, y_val],
                         verbose=1,  callbacks=[reduce_lr, checkpointer, historyss])

        print(hist.history)
        print(hist.epoch)
        print(hist.history.keys())

        train_Y = s3.inverse_transform(y_train)
        test_Y = s7.inverse_transform(y_val)
       
        print("#")
        print(train_Y,test_Y) 

        train_predict = model.predict(X_train, batch_size=1)     
        val_predict = model.predict(X_val, batch_size=1)   

        train_predict = s3.inverse_transform(train_predict)
        test_predict = s7.inverse_transform(val_predict)
       
        print(test_predict, test_Y) 
        train_cc = computecc(train_Y, train_predict)
        print("train_CC:", train_cc)
        test_cc = computecc(test_Y, test_predict)
        print("val_CC:",test_cc)

        print("mse,mae:", mean_squared_error(test_Y, test_predict), mean_absolute_error(test_Y, test_predict))
        file_handle = open(os.path.join(path, 'mse_mae.txt'), mode='a+')
        file_handle.write(str(mean_squared_error(test_Y, test_predict)))    
        file_handle.write('\n')
        file_handle.write(str(mean_absolute_error(test_Y, test_predict)))   
        file_handle.close()

        n = len(test_Y)
        mse = sum(np.square(test_Y - test_predict)) / n
        mae = sum(np.abs(test_Y - test_predict)) / n
        print("mse",mse)
        print("mae",mae)

        np.savetxt(os.path.join(path, 'predict_train_predict.txt'), train_predict)
        np.savetxt(os.path.join(path, 'predict_val_predict.txt'), test_predict)
        np.savetxt(os.path.join(path, 'predict_traincc.txt'), train_cc)
        np.savetxt(os.path.join(path, 'predict_testcc.txt'), test_cc)
        kk = model.evaluate(X_val, y_val, batch_size=1, sample_weight=None, verbose=1)
        print(kk)
        np.savetxt(os.path.join(path, 'model_evaluate_kk.txt'), kk)
        plotmse(epoch=epoch, hist=hist, file_dir=path, i=i)
        plottruepredict(test_Y,test_predict,sequence_length+predict_length,file_dir=path, i=i)
def test(data = None,batch_size=None,model=None, sequence_length=None,predict_length=None,path = None):
    if data is None:
        print('Loading data... ')
        train = open(r"C:\Users\DELL\my tdam\C\speed_train.txt")
        line = train.readline()
        train_list = []
        while line:
            num = list(map(float, line.split()))
            train_list.append(num)
            line = train.readline()
        train_list = np.array(train_list, dtype = 'float64')
        train.close()
        val = open(r"C:\Users\DELL\my tdam\C\speed_val(2016).txt")
        line = val.readline()
        val_list = []
        while line:
            num = list(map(float, line.split()))
            val_list.append(num)
            line = val.readline()
        val_list = np.array(val_list, dtype = 'float64')
        val.close()

        train_n, train_low, train_high = Normalize(train_list)
        
        val_n = Normalize2(val_list, train_low, train_high)
        print(train_n, val_n)

        X_train, y_train = get_data(gen='train', data_list=train_n, sequence_length=sequence_length, predict_length=predict_length)  
        X_val, y_val = get_data(gen='val', data_list=val_n, sequence_length=sequence_length, predict_length=predict_length)
        print("#")
        print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)  
    else:
        X_train, y_train, X_val, y_val = data
    print('\nData Loaded. Compiling...\n')

    print("loading model")
    model = load_model(model)
    print("model ok!")
    print(model.metrics_names)
    train_Y = FNoramlize(y_train, train_low, train_high)
    test_Y = FNoramlize(y_val, train_low, train_high)
    
    print(train_Y, test_Y) 

    train_predict = model.predict(X_train, batch_size=batch_size)  
    val_predict = model.predict(X_val, batch_size=batch_size) 

    train_predict = FNoramlize(train_predict, train_low, train_high)  
    test_predict = FNoramlize(val_predict, train_low, train_high)  
    print(test_predict, test_Y)  

    train_cc = computecc(train_Y, train_predict)
    print("train_CC:",train_cc)
    test_cc = computecc(test_Y, test_predict)
    print("val_CC:",test_cc)

    diff = np.mean((test_predict - test_Y) ** 2, axis=0)
    print("diff:")
    print(np.sqrt(diff))  

    np.savetxt(os.path.join(path, 'model_predict_train_predict.txt'), train_predict)
    np.savetxt(os.path.join(path,'model_predict_val_predict.txt'), test_predict)
    np.savetxt(os.path.join(path,'model_predict_traincc.txt'), train_cc)
    np.savetxt(os.path.join(path,'model_predict_testcc.txt'), test_cc)
    kk = model.evaluate(X_val, y_val, batch_size=batch_size, sample_weight=None, verbose=1)
    print(model.metrics_names)
    print(kk)
    np.savetxt(os.path.join(path, 'model_evaluate_kk.txt'), kk)
    plottruepredict(y_val,val_predict,sequence_length+predict_length,file_dir=path)

if __name__ == '__main__':
    i = 132  
    predict_length = 24
    run_network(batch_size=128, epoch=50, sequence_length=i, predict_length=predict_length,
                path='Speed_area_density_tempure_sigma_B_pressure_ICME_'+str(i)+'_24', i=1,
                checkpoint_dir='', area_length=i + predict_length - 96, area_length_two=10)
