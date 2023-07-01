# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29  1 19:23:56 2023

@author: Long Luo
"""

################################################################################
## Keras implementation of day-ahead prediction of the ISO-NE hourly demand data.

# -----------------------------------------------------------------------------
# load original data file
# modification of the data loading procedure is needed if you have your own dataset

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
start=0
start=time.perf_counter()
parse_dates = ['date']
df = pd.read_csv('selected_data_ISONE.csv', parse_dates=parse_dates, index_col='date')

# 1. get the maximum and minimum demands in 0-24 clock intervals
# 2. get the daily demand and temperature values
MaximumLoadDays = df.groupby('date').demand.max().values
MinimumLoadDays = df.groupby('date').demand.min().values
Load = df.demand.values
Temperature = df.temperature.values

# duplicate max and min daily demand values for 24 hours in a day
MaximumLoad = np.zeros(len(Load))
MinimumLoad = np.zeros(len(Load))

for i in range(len(Load)):
    n_day = int(i/24)
    MaximumLoad[i] = MaximumLoadDays[n_day]
    MinimumLoad[i] = MinimumLoadDays[n_day]

# normalization based on peak values
MaximumLoad = MaximumLoad / 24000.
MinimumLoad = MinimumLoad / 24000.
Load = Load / 24000.
Temperature = Temperature / 100.

# add weekday info to the dataset
# the initial value for iter_weekday corresponds to the first day of the dataset
InitialWeek = 6
weekday = np.zeros((24*4324,))
for i in range(4324):
    mod = np.mod(InitialWeek, 7)
    for j in range(24):
        if (mod == 6) or (mod == 0):
            weekday[i*24 + j] = 0
        else:
            weekday[i*24 + j] = 1
    InitialWeek += 1

# add season and festival info to the dataset
import datetime
iter_date = datetime.date(2003, 3, 1)
season = np.zeros((24*4324,))
festival = np.zeros((24*4324,))
for i in range(4324):
    month = iter_date.month
    day = iter_date.day
    for j in range(24):
        if (month==4) | (month==5) | ((month==3) and (day>7)) | ((month==6) and (day<8)):
            season[i*24 + j] = 0
        elif (month==7) | (month==8) | ((month==6) and (day>7)) | ((month==9) and (day<8)):
            season[i*24 + j] = 1
        elif (month==10) | (month==11) | ((month==9) and (day>7)) | ((month==12) and (day<8)):
            season[i*24 + j] = 2
        elif (month==1) | (month==2) | ((month==12) and (day>7)) | ((month==3) and (day<8)):
            season[i*24 + j] = 3

        if (month == 7) and (day == 4):
            festival[i*24 + j] = 1
        if (month == 11) and (iter_date.weekday() == 4) and (day + 7 > 30):
            festival[i*24 + j] = 1
        if (month == 12) and (day == 25):
            festival[i*24 + j] = 1
    iter_date = iter_date + datetime.timedelta(1)
    
def data_split(D, T, D_max, D_min, season, weekday, festival, num_train_days, validation_split = 0.1):
    '''
    prepare the dataset used for training and testing of the model.
    '''

    input_1 = []
    input_21_D = []
    input_21_T = []
    input_22_D = []
    input_22_T = []
    input_23_D = []
    input_23_T = []
    input_3 = []
    input_4 = []
    input_5 = []
    input_season = []
    input_weekday = []
    input_festival = []
    output = []
    
    len_dataset = D.shape[0]
    num_sample = len_dataset-2016
    # 2016 hours (28*3 days) is needed so that we can formulate the first datapoint
    
    for i in range(2016,len_dataset):   
        # the demand values of the most recent 24 hours
        input_1.append(D[i-24:i])
        
        # multiple demand values every 24 hours within a week
        index_input_21 = [i-24, i-48, i-72, i-96, i-120, i-144, i-168]
        input_21_D.append(D[index_input_21])
        input_21_T.append(T[index_input_21])
        
        # multiple demand values every week within two months
        index_input_22 = [i-168, i-336, i-504, i-672, i-840, i-1008, i-1176, i-1344]
        input_22_D.append(D[index_input_22])
        input_22_T.append(T[index_input_22])
        
        # multiple demand values every month within several months
        index_input_23 = [i-672, i-1344, i-2016]
        input_23_D.append(D[index_input_23])
        input_23_T.append(T[index_input_23])
        
        input_3.append(T[i])
        input_4.append(D_max[i])
        input_5.append(D_min[i])
        
        output.append(D[i])
        
        # get one-hot representations of the additional information
        season_onehot = np.zeros(4)
        season_onehot[int(season[i])] = 1 
        input_season.append(season_onehot)

        weekday_onehot = np.zeros(2)
        weekday_onehot[int(weekday[i])] = 1 
        input_weekday.append(weekday_onehot)

        festival_onehot = np.zeros(2)
        festival_onehot[int(festival[i])] = 1 
        input_festival.append(festival_onehot)

        
    Input_1 = np.array(input_1)
    Input_21_D = np.array(input_21_D)
    Input_21_T = np.array(input_21_T)
    Input_22_D = np.array(input_22_D)
    Input_22_T = np.array(input_22_T)
    Input_23_D = np.array(input_23_D)
    Input_23_T = np.array(input_23_T)
    Input_3 = np.array(input_3)
    Input_4 = np.array(input_4)
    Input_5 = np.array(input_5)
    Input_season = np.array(input_season)
    Input_weekday = np.array(input_weekday)
    Input_festival = np.array(input_festival)
    Output_1 = np.array(output)

    
    num_train = num_train_days * 24
    num_val = int(num_train * validation_split)
    
    Input_train = []
    Input_val = []
    Input_test = []
    Output_train = []
    Output_val = []
    Output_test = []
    
    # we prepare 24 sets of data for the 24 sub-networks, each sub-network is aimed at forecasting the load of one hour of the next day
    for i in range(24):
        #               0                          1                         2                         3                         4                         5                         6                         7                    8                    9                    10                          11                           12                                              
        Input_train.append([Input_1[i:num_train:24,:24-i], Input_21_D[i:num_train:24,:], Input_21_T[i:num_train:24,:], Input_22_D[i:num_train:24,:], Input_22_T[i:num_train:24,:], Input_23_D[i:num_train:24,:], Input_23_T[i:num_train:24,:], Input_3[i:num_train:24], Input_4[i:num_train:24], Input_5[i:num_train:24], Input_season[i:num_train:24,:], Input_weekday[i:num_train:24,:], Input_festival[i:num_train:24,:]])
        Input_val.append([Input_1[num_train-num_val+i:num_train:24,:24-i], Input_21_D[num_train-num_val+i:num_train:24,:], Input_21_T[num_train-num_val+i:num_train:24,:], Input_22_D[num_train-num_val+i:num_train:24,:], Input_22_T[num_train-num_val+i:num_train:24,:], Input_23_D[num_train-num_val+i:num_train:24,:], Input_23_T[num_train-num_val+i:num_train:24,:], Input_3[num_train-num_val+i:num_train:24], Input_4[num_train-num_val+i:num_train:24], Input_5[num_train-num_val+i:num_train:24], Input_season[num_train-num_val+i:num_train:24,:], Input_weekday[num_train-num_val+i:num_train:24,:], Input_festival[num_train-num_val+i:num_train:24,:]])
        Input_test.append([Input_1[num_train+i:num_sample:24,:24-i], Input_21_D[num_train+i:num_sample:24,:], Input_21_T[num_train+i:num_sample:24,:], Input_22_D[num_train+i:num_sample:24,:], Input_22_T[num_train+i:num_sample:24,:], Input_23_D[num_train+i:num_sample:24,:], Input_23_T[num_train+i:num_sample:24,:], Input_3[num_train+i:num_sample:24], Input_4[num_train+i:num_sample:24], Input_5[num_train+i:num_sample:24], Input_season[num_train+i:num_sample:24,:], Input_weekday[num_train+i:num_sample:24,:], Input_festival[num_train+i:num_sample:24,:]])
        Output_train.append(Output_1[i:num_train:24])
        Output_val.append(Output_1[num_train-num_val+i:num_train:24])
        Output_test.append(Output_1[num_train+i:num_sample:24])

    return (Input_train, Input_val, Input_test, Output_train, Output_val, Output_test)

# num_pre_days: the number of days we need before we can get the first sample, in this case: 3*28 days 
number_of_pre_days = 84
number_of_days = 3227
number_of_test_days = 730
number_of_training_days = 2107
number_of_data_points = number_of_days * 24
number_of_days_start = number_of_days - number_of_pre_days - number_of_test_days - number_of_training_days
start_data_point = number_of_days_start * 24
Input_train, Input_val, Input_test, Output_train, Output_val, Output_test = data_split(Load[start_data_point: start_data_point + number_of_data_points], Temperature[start_data_point: start_data_point + number_of_data_points], MaximumLoad[start_data_point: start_data_point + number_of_data_points], MinimumLoad[start_data_point: start_data_point + number_of_data_points], season[start_data_point: start_data_point + number_of_data_points], weekday[start_data_point: start_data_point + number_of_data_points], festival[start_data_point: start_data_point + number_of_data_points], number_of_training_days, 0.1)

## ----------------------------------------------------------------------------
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, LSTM, concatenate, Activation, add, BatchNormalization,Dropout,Flatten,Conv1D,AveragePooling1D
from keras.layers import multiply, maximum, dot, average
from keras import backend as K
from keras.losses import mean_absolute_percentage_error, hinge
from keras.regularizers import l1, l2
from keras.callbacks import ReduceLROnPlateau
from keras.initializers import glorot_normal
from keras.callbacks import EarlyStopping 
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

def get_input(hour):
    # Define the hourly input and set the shape of the input
    input_Load_day = Input(shape=(7,), name = 'input'+str(hour)+'_Load_day')
    input_Load_week = Input(shape=(8,), name = 'input'+str(hour)+'_Load_week')
    input_Load_month = Input(shape=(3,), name = 'input'+str(hour)+'_Load_month')
    input_Load_hour = Input(shape=(24-hour+1,), name = 'input'+str(hour)+'_Load_hour') 
                                                                           
    input_Temperature_day = Input(shape=(7,), name = 'input'+str(hour)+'_Temperature_day')
    input_Temperature_week = Input(shape=(8,), name = 'input'+str(hour)+'_Temperature_week')
    input_Temperature_month = Input(shape=(3,), name = 'input'+str(hour)+'_Temperature_month')
    
    input_Temperature = Input(shape=(1,))
    
    return (input_Load_day, input_Load_week, input_Load_month, input_Load_hour, input_Temperature_day, input_Temperature_week, input_Temperature_month, input_Temperature)
    
input1_Load_day, input1_Load_week, input1_Load_month, input1_Load_hour, input1_Temperature_day, input1_Temperature_week, input1_Temperature_month, input1_Temperature = get_input(1)
input2_Load_day, input2_Load_week, input2_Load_month, input2_Load_hour, input2_Temperature_day, input2_Temperature_week, input2_Temperature_month, input2_Temperature = get_input(2)
input3_Load_day, input3_Load_week, input3_Load_month, input3_Load_hour, input3_Temperature_day, input3_Temperature_week, input3_Temperature_month, input3_Temperature = get_input(3)
input4_Load_day, input4_Load_week, input4_Load_month, input4_Load_hour, input4_Temperature_day, input4_Temperature_week, input4_Temperature_month, input4_Temperature = get_input(4)
input5_Load_day, input5_Load_week, input5_Load_month, input5_Load_hour, input5_Temperature_day, input5_Temperature_week, input5_Temperature_month, input5_Temperature = get_input(5)
input6_Load_day, input6_Load_week, input6_Load_month, input6_Load_hour, input6_Temperature_day, input6_Temperature_week, input6_Temperature_month, input6_Temperature = get_input(6)
input7_Load_day, input7_Load_week, input7_Load_month, input7_Load_hour, input7_Temperature_day, input7_Temperature_week, input7_Temperature_month, input7_Temperature = get_input(7)
input8_Load_day, input8_Load_week, input8_Load_month, input8_Load_hour, input8_Temperature_day, input8_Temperature_week, input8_Temperature_month, input8_Temperature = get_input(8)
input9_Load_day, input9_Load_week, input9_Load_month, input9_Load_hour, input9_Temperature_day, input9_Temperature_week, input9_Temperature_month, input9_Temperature = get_input(9)
input10_Load_day, input10_Load_week, input10_Load_month, input10_Load_hour, input10_Temperature_day, input10_Temperature_week, input10_Temperature_month, input10_Temperature = get_input(10)
input11_Load_day, input11_Load_week, input11_Load_month, input11_Load_hour, input11_Temperature_day, input11_Temperature_week, input11_Temperature_month, input11_Temperature = get_input(11)
input12_Load_day, input12_Load_week, input12_Load_month, input12_Load_hour, input12_Temperature_day, input12_Temperature_week, input12_Temperature_month, input12_Temperature = get_input(12)
input13_Load_day, input13_Load_week, input13_Load_month, input13_Load_hour, input13_Temperature_day, input13_Temperature_week, input13_Temperature_month, input13_Temperature = get_input(13)
input14_Load_day, input14_Load_week, input14_Load_month, input14_Load_hour, input14_Temperature_day, input14_Temperature_week, input14_Temperature_month, input14_Temperature = get_input(14)
input15_Load_day, input15_Load_week, input15_Load_month, input15_Load_hour, input15_Temperature_day, input15_Temperature_week, input15_Temperature_month, input15_Temperature = get_input(15)
input16_Load_day, input16_Load_week, input16_Load_month, input16_Load_hour, input16_Temperature_day, input16_Temperature_week, input16_Temperature_month, input16_Temperature = get_input(16)
input17_Load_day, input17_Load_week, input17_Load_month, input17_Load_hour, input17_Temperature_day, input17_Temperature_week, input17_Temperature_month, input17_Temperature = get_input(17)
input18_Load_day, input18_Load_week, input18_Load_month, input18_Load_hour, input18_Temperature_day, input18_Temperature_week, input18_Temperature_month, input18_Temperature = get_input(18)
input19_Load_day, input19_Load_week, input19_Load_month, input19_Load_hour, input19_Temperature_day, input19_Temperature_week, input19_Temperature_month, input19_Temperature = get_input(19)
input20_Load_day, input20_Load_week, input20_Load_month, input20_Load_hour, input20_Temperature_day, input20_Temperature_week, input20_Temperature_month, input20_Temperature = get_input(20)
input21_Load_day, input21_Load_week, input21_Load_month, input21_Load_hour, input21_Temperature_day, input21_Temperature_week, input21_Temperature_month, input21_Temperature = get_input(21)
input22_Load_day, input22_Load_week, input22_Load_month, input22_Load_hour, input22_Temperature_day, input22_Temperature_week, input22_Temperature_month, input22_Temperature = get_input(22)
input23_Load_day, input23_Load_week, input23_Load_month, input23_Load_hour, input23_Temperature_day, input23_Temperature_week, input23_Temperature_month, input23_Temperature = get_input(23)
input24_Load_day, input24_Load_week, input24_Load_month, input24_Load_hour, input24_Temperature_day, input24_Temperature_week, input24_Temperature_month, input24_Temperature = get_input(24)
input_Load_max = Input(shape=(1,), name = 'input_D_max')
input_Load_min = Input(shape=(1,), name = 'input_D_min')
input_season = Input(shape=(4,), name = 'input_season')
input_weekday = Input(shape=(2,), name = 'input_weekday')
input_festival = Input(shape=(2,), name = 'input_festival')

# The number of nodes in the middle of the model
num_dense1 = 10
num_dense2 = 10
num_dense1_Load_hour = 10 

# Important layers in the structure of Model E
DenseConcaTemperature_monthid = Dense(num_dense2, activation = 'selu', kernel_initializer = 'lecun_normal', kernel_regularizer=l2(0.0001))
DenseConcatLoad_hour = Dense(num_dense1_Load_hour, activation = 'selu', kernel_initializer = 'lecun_normal', kernel_regularizer=l2(0.0001))
DenseMid = Dense(50, activation = 'selu', kernel_initializer = 'lecun_normal', kernel_regularizer=l2(0.0001))
DenseConcat1 = Dense(num_dense2, activation = 'selu', kernel_initializer = 'lecun_normal', kernel_regularizer=l2(0.0001))
DenseConcat2 = Dense(num_dense2, activation = 'selu', kernel_initializer = 'lecun_normal', kernel_regularizer=l2(0.0001))
DenseConcat3 = Dense(num_dense2, activation = 'selu', kernel_initializer = 'lecun_normal', kernel_regularizer=l2(0.0001))

# Important layers in the structure of Model B
DenseConcaTemperature_monthid1 = Dense(num_dense2, activation = 'selu', kernel_initializer = 'lecun_normal', kernel_regularizer=l2(0.0001))
DenseConcatLoad_hour1 = Dense(num_dense1_Load_hour, activation = 'selu', kernel_initializer = 'lecun_normal', kernel_regularizer=l2(0.0001))
DenseMid1 = Dense(50, activation = 'selu', kernel_initializer = 'lecun_normal', kernel_regularizer=l2(0.0001))
DenseConcat11 = Dense(num_dense2, activation = 'selu', kernel_initializer = 'lecun_normal', kernel_regularizer=l2(0.0001))
DenseConcat21 = Dense(num_dense2, activation = 'selu', kernel_initializer = 'lecun_normal', kernel_regularizer=l2(0.0001))
DenseConcat31 = Dense(num_dense2, activation = 'selu', kernel_initializer = 'lecun_normal', kernel_regularizer=l2(0.0001))

# construction model
def get_layer_1(hour, input_Load_day, input_Load_week, input_Load_month, input_Load_hour, input_Temperature_day, input_Temperature_week, input_Temperature_month, input_Temperature, output_pre=[]):
    
    input_Load_day=tf.reshape(input_Load_day,[-1,7,1])
    Conv1D_Load_day = Conv1D(1, 3,activation= 'selu', kernel_initializer = 'lecun_normal')(input_Load_day)
    Pooling1D_Load_day = AveragePooling1D(2,strides=1)(Conv1D_Load_day)
    Flatten_Load_day = Flatten()(Pooling1D_Load_day)
    input_Load_week=tf.reshape(input_Load_week,[-1,8,1])
    Conv1D_Load_week = Conv1D(1, 3,activation= 'selu', kernel_initializer = 'lecun_normal')(input_Load_week)
    Pooling1D_Load_week = AveragePooling1D(2,strides=1)(Conv1D_Load_week)
    Flatten_Load_week = Flatten()(Pooling1D_Load_week)
    input_Load_month=tf.reshape(input_Load_month,[-1,3,1])
    Conv1D_Load_month = Conv1D(1, 2,activation= 'selu', kernel_initializer = 'lecun_normal')(input_Load_month)
    Pooling1D_Load_month = AveragePooling1D(2,strides=1)(Conv1D_Load_month)
    Flatten_Load_month = Flatten()(Pooling1D_Load_month)
    input_Temperature_day=tf.reshape(input_Temperature_day,[-1,7,1])
    Conv1D_Temperature_day = Conv1D(1, 3,activation= 'selu', kernel_initializer = 'lecun_normal')(input_Temperature_day)
    Pooling1D_Temperature_day = AveragePooling1D(2,strides=1)(Conv1D_Temperature_day)
    Flatten_Temperature_day = Flatten()(Pooling1D_Temperature_day)
    input_Temperature_week=tf.reshape(input_Temperature_week,[-1,8,1])
    Conv1D_Temperature_week = Conv1D(1, 3,activation= 'selu', kernel_initializer = 'lecun_normal')(input_Temperature_week)
    Pooling1D_Temperature_week = AveragePooling1D(2,strides=1)(Conv1D_Temperature_week)
    Flatten_Temperature_week = Flatten()(Pooling1D_Temperature_week)
    input_Temperature_month=tf.reshape(input_Temperature_month,[-1,3,1])
    Conv1D_Temperature_month = Conv1D(1, 2,activation= 'selu', kernel_initializer = 'lecun_normal')(input_Temperature_month)
    Pooling1D_Temperature_month = AveragePooling1D(2,strides=1)(Conv1D_Temperature_month)
    Flatten_Temperature_month = Flatten()(Pooling1D_Temperature_month)
    
    # B structure
    def industrial_load(hour, Flatten_Load_day, Flatten_Load_week, Flatten_Load_month, input_Load_hour):     
        dense_Load_day = Dense(num_dense1, activation = 'selu', kernel_initializer = 'lecun_normal')(Flatten_Load_day)
        dense_Load_week = Dense(num_dense1, activation = 'selu', kernel_initializer = 'lecun_normal')(Flatten_Load_week)
        dense_Load_month = Dense(num_dense1, activation = 'selu', kernel_initializer = 'lecun_normal')(Flatten_Load_month)

        dense_concat1 = DenseConcat11(dense_Load_day)
        dense_concat2 = DenseConcat21(dense_Load_week)
        dense_concat3 = DenseConcat31(dense_Load_month)

        concat_date_info = concatenate([input_season, input_weekday])
        dense_concat_date_info_1 = Dense(5, activation = 'selu', kernel_initializer = 'lecun_normal')(concat_date_info)
        dense_concat_date_info_2 = Dense(5, activation = 'selu', kernel_initializer = 'lecun_normal')(concat_date_info)
        
        concat_mid = concatenate([dense_concat1, dense_concat2, dense_concat3, dense_concat_date_info_1, input_festival])
        dense_concat_mid = DenseConcaTemperature_monthid1(concat_mid)

        if output_pre == []:
            input_Load_hour=tf.reshape(input_Load_hour,[-1,24,1])
            Conv1D_Load_hour = Conv1D(1, 3,activation= 'selu', kernel_initializer = 'lecun_normal')(input_Load_hour)
            Pooling1D_Load_hour = AveragePooling1D(2,strides=1)(Conv1D_Load_hour)
            Flatten_Load_hour = Flatten()(Pooling1D_Load_hour)
            dense_Load_hour = Dense(num_dense1_Load_hour, activation = 'selu', kernel_initializer = 'lecun_normal')(Flatten_Load_hour)
        else:
            concat_Load_hour = concatenate([input_Load_hour] + output_pre)
            concat_Load_hour=tf.reshape(concat_Load_hour,[-1,24,1])
            Conv1D_Load_hour = Conv1D(1, 3,activation= 'selu', kernel_initializer = 'lecun_normal')(concat_Load_hour)
            Pooling1D_Load_hour = AveragePooling1D(2,strides=1)(Conv1D_Load_hour)
            Flatten_Load_hour = Flatten()(Pooling1D_Load_hour)
            dense_Load_hour = Dense(num_dense1_Load_hour, activation = 'selu', kernel_initializer = 'lecun_normal')(Flatten_Load_hour)

        dense_4 = DenseConcatLoad_hour1(concatenate([dense_Load_hour, dense_concat_date_info_2]))           
        concat = concatenate([dense_concat_mid, dense_4])
        dense_mid = DenseMid1(concat)
        dropout_mid = Dropout(0.5)(dense_mid)        
        output = Dense(1, activation = 'linear', kernel_initializer = 'lecun_normal')(dropout_mid)
        return output
    
    # E structure
    def elastic_load(hour, Flatten_Load_day, Flatten_Load_week, Flatten_Load_month, input_Load_hour, Flatten_Temperature_day, Flatten_Temperature_week, Flatten_Temperature_month, input_Temperature):     
        dense_Load_day = Dense(num_dense1, activation = 'selu', kernel_initializer = 'lecun_normal')(Flatten_Load_day)
        dense_Load_week = Dense(num_dense1, activation = 'selu', kernel_initializer = 'lecun_normal')(Flatten_Load_week)
        dense_Load_month = Dense(num_dense1, activation = 'selu', kernel_initializer = 'lecun_normal')(Flatten_Load_month)
        dense_Temperature_day = Dense(num_dense1, activation = 'selu', kernel_initializer = 'lecun_normal')(Flatten_Temperature_day)
        dense_Temperature_week = Dense(num_dense1, activation = 'selu', kernel_initializer = 'lecun_normal')(Flatten_Temperature_day)
        dense_Temperature_month = Dense(num_dense1, activation = 'selu', kernel_initializer = 'lecun_normal')(Flatten_Temperature_month)
        
        concat1 = concatenate([dense_Load_day, dense_Temperature_day])
        dense_concat1 = DenseConcat1(concat1) 
        concat2 = concatenate([dense_Load_week, dense_Temperature_week])
        dense_concat2 = DenseConcat2(concat2)
        concat3 = concatenate([dense_Load_month, dense_Temperature_month])
        dense_concat3 = DenseConcat3(concat3)

        concat_date_info = concatenate([input_season, input_weekday])
        dense_concat_date_info_1 = Dense(5, activation = 'selu', kernel_initializer = 'lecun_normal')(concat_date_info)
        dense_concat_date_info_2 = Dense(5, activation = 'selu', kernel_initializer = 'lecun_normal')(concat_date_info)
        
        concat_mid = concatenate([dense_concat1, dense_concat2, dense_concat3, dense_concat_date_info_1, input_festival])
        dense_concat_mid = DenseConcaTemperature_monthid(concat_mid)
   
        if output_pre == []:
            input_Load_hour=tf.reshape(input_Load_hour,[-1,24,1])
            Conv1D_Load_hour = Conv1D(1, 3,activation= 'selu', kernel_initializer = 'lecun_normal')(input_Load_hour)
            Pooling1D_Load_hour = AveragePooling1D(2,strides=1)(Conv1D_Load_hour)
            Flatten_Load_hour = Flatten()(Pooling1D_Load_hour)
            dense_Load_hour = Dense(num_dense1_Load_hour, activation = 'selu', kernel_initializer = 'lecun_normal')(Flatten_Load_hour)
        else:
            concat_Load_hour = concatenate([input_Load_hour] + output_pre)
            concat_Load_hour=tf.reshape(concat_Load_hour,[-1,24,1])
            Conv1D_Load_hour = Conv1D(1, 3,activation= 'selu', kernel_initializer = 'lecun_normal')(concat_Load_hour)
            Pooling1D_Load_hour = AveragePooling1D(2,strides=1)(Conv1D_Load_hour)
            Flatten_Load_hour = Flatten()(Pooling1D_Load_hour)
            dense_Load_hour = Dense(num_dense1_Load_hour, activation = 'selu', kernel_initializer = 'lecun_normal')(Flatten_Load_hour)

        dense_4 = DenseConcatLoad_hour(concatenate([dense_Load_hour, dense_concat_date_info_2]))
        concat = concatenate([dense_concat_mid, dense_4, input_Temperature])
        dense_mid = DenseMid(concat)
        dropout_mid = Dropout(0.5)(dense_mid)        
        output = Dense(1, activation = 'linear', kernel_initializer = 'lecun_normal')(dropout_mid)
        return output
    
    # Reconstruction of E structure and B structure
    industrial_loads=industrial_load(hour, Flatten_Load_day, Flatten_Load_week, Flatten_Load_month, input_Load_hour)
    elastic_loads=elastic_load(hour, Flatten_Load_day, Flatten_Load_week, Flatten_Load_month, input_Load_hour, Flatten_Temperature_day, Flatten_Temperature_week, Flatten_Temperature_month, input_Temperature)
    output = add([industrial_loads, elastic_loads])
    
    # Save the output layer and connect the next hour load forecasting
    output_pre_new = output_pre + [output]
    return (output, output_pre_new)

# Generate hourly model      
output1, output_pre1 = get_layer_1(1, input1_Load_day, input1_Load_week, input1_Load_month, input1_Load_hour, input1_Temperature_day, input1_Temperature_week, input1_Temperature_month, input1_Temperature)
output2, output_pre2 = get_layer_1(2, input2_Load_day, input2_Load_week, input2_Load_month, input2_Load_hour, input2_Temperature_day, input2_Temperature_week, input2_Temperature_month, input2_Temperature, output_pre1)
output3, output_pre3 = get_layer_1(3, input3_Load_day, input3_Load_week, input3_Load_month, input3_Load_hour, input3_Temperature_day, input3_Temperature_week, input3_Temperature_month, input3_Temperature, output_pre2)
output4, output_pre4 = get_layer_1(4, input4_Load_day, input4_Load_week, input4_Load_month, input4_Load_hour, input4_Temperature_day, input4_Temperature_week, input4_Temperature_month, input4_Temperature, output_pre3)
output5, output_pre5 = get_layer_1(5, input5_Load_day, input5_Load_week, input5_Load_month, input5_Load_hour, input5_Temperature_day, input5_Temperature_week, input5_Temperature_month, input5_Temperature, output_pre4)
output6, output_pre6 = get_layer_1(6, input6_Load_day, input6_Load_week, input6_Load_month, input6_Load_hour, input6_Temperature_day, input6_Temperature_week, input6_Temperature_month, input6_Temperature, output_pre5)
output7, output_pre7 = get_layer_1(7, input7_Load_day, input7_Load_week, input7_Load_month, input7_Load_hour, input7_Temperature_day, input7_Temperature_week, input7_Temperature_month, input7_Temperature, output_pre6)
output8, output_pre8 = get_layer_1(8, input8_Load_day, input8_Load_week, input8_Load_month, input8_Load_hour, input8_Temperature_day, input8_Temperature_week, input8_Temperature_month, input8_Temperature, output_pre7)
output9, output_pre9 = get_layer_1(9, input9_Load_day, input9_Load_week, input9_Load_month, input9_Load_hour, input9_Temperature_day, input9_Temperature_week, input9_Temperature_month, input9_Temperature, output_pre8)
output10, output_pre10 = get_layer_1(10, input10_Load_day, input10_Load_week, input10_Load_month, input10_Load_hour, input10_Temperature_day, input10_Temperature_week, input10_Temperature_month, input10_Temperature, output_pre9)
output11, output_pre11 = get_layer_1(11, input11_Load_day, input11_Load_week, input11_Load_month, input11_Load_hour, input11_Temperature_day, input11_Temperature_week, input11_Temperature_month, input11_Temperature, output_pre10)
output12, output_pre12 = get_layer_1(12, input12_Load_day, input12_Load_week, input12_Load_month, input12_Load_hour, input12_Temperature_day, input12_Temperature_week, input12_Temperature_month, input12_Temperature, output_pre11)
output13, output_pre13 = get_layer_1(13, input13_Load_day, input13_Load_week, input13_Load_month, input13_Load_hour, input13_Temperature_day, input13_Temperature_week, input13_Temperature_month, input13_Temperature, output_pre12)
output14, output_pre14 = get_layer_1(14, input14_Load_day, input14_Load_week, input14_Load_month, input14_Load_hour, input14_Temperature_day, input14_Temperature_week, input14_Temperature_month, input14_Temperature, output_pre13)
output15, output_pre15 = get_layer_1(15, input15_Load_day, input15_Load_week, input15_Load_month, input15_Load_hour, input15_Temperature_day, input15_Temperature_week, input15_Temperature_month, input15_Temperature, output_pre14)
output16, output_pre16 = get_layer_1(16, input16_Load_day, input16_Load_week, input16_Load_month, input16_Load_hour, input16_Temperature_day, input16_Temperature_week, input16_Temperature_month, input16_Temperature, output_pre15)
output17, output_pre17 = get_layer_1(17, input17_Load_day, input17_Load_week, input17_Load_month, input17_Load_hour, input17_Temperature_day, input17_Temperature_week, input17_Temperature_month, input17_Temperature, output_pre16)
output18, output_pre18 = get_layer_1(18, input18_Load_day, input18_Load_week, input18_Load_month, input18_Load_hour, input18_Temperature_day, input18_Temperature_week, input18_Temperature_month, input18_Temperature, output_pre17)
output19, output_pre19 = get_layer_1(19, input19_Load_day, input19_Load_week, input19_Load_month, input19_Load_hour, input19_Temperature_day, input19_Temperature_week, input19_Temperature_month, input19_Temperature, output_pre18)
output20, output_pre20 = get_layer_1(20, input20_Load_day, input20_Load_week, input20_Load_month, input20_Load_hour, input20_Temperature_day, input20_Temperature_week, input20_Temperature_month, input20_Temperature, output_pre19)
output21, output_pre21 = get_layer_1(21, input21_Load_day, input21_Load_week, input21_Load_month, input21_Load_hour, input21_Temperature_day, input21_Temperature_week, input21_Temperature_month, input21_Temperature, output_pre20)
output22, output_pre22 = get_layer_1(22, input22_Load_day, input22_Load_week, input22_Load_month, input22_Load_hour, input22_Temperature_day, input22_Temperature_week, input22_Temperature_month, input22_Temperature, output_pre21)
output23, output_pre23 = get_layer_1(23, input23_Load_day, input23_Load_week, input23_Load_month, input23_Load_hour, input23_Temperature_day, input23_Temperature_week, input23_Temperature_month, input23_Temperature, output_pre22)
output24, output_pre24 = get_layer_1(24, input24_Load_day, input24_Load_week, input24_Load_month, input24_Load_hour, input24_Temperature_day, input24_Temperature_week, input24_Temperature_month, input24_Temperature, output_pre23)

# residual block
def get_res_layer(output, last=False):
    dense_res11 = Dense(20, activation = 'selu', kernel_initializer = 'lecun_normal', kernel_regularizer=l2(0.0001))(output)
    dense_res12 = Dense(24, activation = 'linear', kernel_initializer = 'lecun_normal')(dense_res11)
    dense_res21 = Dense(20, activation = 'selu', kernel_initializer = 'lecun_normal', kernel_regularizer=l2(0.0001))(output)
    dense_res22 = Dense(24, activation = 'linear', kernel_initializer = 'lecun_normal')(dense_res21)

    dense_add = add([dense_res12, dense_res22])
    
    if last:
        output_new = add([dense_add, output], name = 'output')
    else:
        output_new = add([dense_add, output])
    return output_new

output_pre = concatenate(output_pre24)

# The layer of ResNet-Plus
def resnetplus_layer(input_1, input_2, output_list):
    output_res = get_res_layer(input_1)
    output_res_ = get_res_layer(input_2)
    output_res_ave_mid = average([output_res, output_res_])
    output_list.append(output_res_ave_mid)
    output_res_ave = average(output_list)
    return output_res_ave, output_list
    
input_1 = output_pre
input_2 = output_pre
output_list = [output_pre]

# The number of layers of ResNet-Plus
num_resnetplus_layer = 20

# The connection of ResNet-Plus
for i in range(num_resnetplus_layer):
    output_res_ave, output_list = resnetplus_layer(input_1, input_2, output_list)
    input_1 = output_res_ave
    if i == 0:
        input_2 = output_res_ave

output = output_res_ave

# loss function
def penalized_loss(y_true, y_pred):
    beta = 0.5
    loss1 = mean_absolute_percentage_error(y_true, y_pred)
    loss2 = K.mean(K.maximum(K.max(y_pred, axis=1) - input_Load_max, 0.), axis=-1)
    loss3 = K.mean(K.maximum(input_Load_min - K.min(y_pred, axis=1), 0.), axis=-1)
    return loss1 + beta * (loss2 + loss3)

def get_XY(X, Y):
    X_new = []
    Y_new = []
    for i in range(24):
        X_new.append(X[i][1])
        X_new.append(X[i][3])
        X_new.append(X[i][5])
        X_new.append(X[i][0])
        X_new.append(X[i][2])
        X_new.append(X[i][4])
        X_new.append(X[i][6])
        X_new.append(X[i][7]) # temperature
        Y_new.append(Y[i])
    X_new = X_new + [X[0][8],X[0][9],X[0][10],X[0][11],X[0][12]] # for shared input, use the data of hour 1 is enough.
    Y_new = [np.squeeze(np.array(Y_new)).transpose()] # the aggregate output of 24 single outputs
    # Y_new = [np.squeeze(np.array(Y_new)).transpose()] * 2 # aggregate Temperature_weekice!
    return (X_new, Y_new)

# -----------------------------------------------------------------------------
# Declarative model
def get_model():
    model = Model(inputs=[input1_Load_day, input1_Load_week, input1_Load_month, input1_Load_hour, input1_Temperature_day, input1_Temperature_week, input1_Temperature_month, input1_Temperature,\
                          input2_Load_day, input2_Load_week, input2_Load_month, input2_Load_hour, input2_Temperature_day, input2_Temperature_week, input2_Temperature_month, input2_Temperature,\
                          input3_Load_day, input3_Load_week, input3_Load_month, input3_Load_hour, input3_Temperature_day, input3_Temperature_week, input3_Temperature_month, input3_Temperature,\
                          input4_Load_day, input4_Load_week, input4_Load_month, input4_Load_hour, input4_Temperature_day, input4_Temperature_week, input4_Temperature_month, input4_Temperature,\
                          input5_Load_day, input5_Load_week, input5_Load_month, input5_Load_hour, input5_Temperature_day, input5_Temperature_week, input5_Temperature_month, input5_Temperature,\
                          input6_Load_day, input6_Load_week, input6_Load_month, input6_Load_hour, input6_Temperature_day, input6_Temperature_week, input6_Temperature_month, input6_Temperature,\
                          input7_Load_day, input7_Load_week, input7_Load_month, input7_Load_hour, input7_Temperature_day, input7_Temperature_week, input7_Temperature_month, input7_Temperature,\
                          input8_Load_day, input8_Load_week, input8_Load_month, input8_Load_hour, input8_Temperature_day, input8_Temperature_week, input8_Temperature_month, input8_Temperature,\
                          input9_Load_day, input9_Load_week, input9_Load_month, input9_Load_hour, input9_Temperature_day, input9_Temperature_week, input9_Temperature_month, input9_Temperature,\
                          input10_Load_day, input10_Load_week, input10_Load_month, input10_Load_hour, input10_Temperature_day, input10_Temperature_week, input10_Temperature_month, input10_Temperature,\
                          input11_Load_day, input11_Load_week, input11_Load_month, input11_Load_hour, input11_Temperature_day, input11_Temperature_week, input11_Temperature_month, input11_Temperature,\
                          input12_Load_day, input12_Load_week, input12_Load_month, input12_Load_hour, input12_Temperature_day, input12_Temperature_week, input12_Temperature_month, input12_Temperature,\
                          input13_Load_day, input13_Load_week, input13_Load_month, input13_Load_hour, input13_Temperature_day, input13_Temperature_week, input13_Temperature_month, input13_Temperature,\
                          input14_Load_day, input14_Load_week, input14_Load_month, input14_Load_hour, input14_Temperature_day, input14_Temperature_week, input14_Temperature_month, input14_Temperature,\
                          input15_Load_day, input15_Load_week, input15_Load_month, input15_Load_hour, input15_Temperature_day, input15_Temperature_week, input15_Temperature_month, input15_Temperature,\
                          input16_Load_day, input16_Load_week, input16_Load_month, input16_Load_hour, input16_Temperature_day, input16_Temperature_week, input16_Temperature_month, input16_Temperature,\
                          input17_Load_day, input17_Load_week, input17_Load_month, input17_Load_hour, input17_Temperature_day, input17_Temperature_week, input17_Temperature_month, input17_Temperature,\
                          input18_Load_day, input18_Load_week, input18_Load_month, input18_Load_hour, input18_Temperature_day, input18_Temperature_week, input18_Temperature_month, input18_Temperature,\
                          input19_Load_day, input19_Load_week, input19_Load_month, input19_Load_hour, input19_Temperature_day, input19_Temperature_week, input19_Temperature_month, input19_Temperature,\
                          input20_Load_day, input20_Load_week, input20_Load_month, input20_Load_hour, input20_Temperature_day, input20_Temperature_week, input20_Temperature_month, input20_Temperature,\
                          input21_Load_day, input21_Load_week, input21_Load_month, input21_Load_hour, input21_Temperature_day, input21_Temperature_week, input21_Temperature_month, input21_Temperature,\
                          input22_Load_day, input22_Load_week, input22_Load_month, input22_Load_hour, input22_Temperature_day, input22_Temperature_week, input22_Temperature_month, input22_Temperature,\
                          input23_Load_day, input23_Load_week, input23_Load_month, input23_Load_hour, input23_Temperature_day, input23_Temperature_week, input23_Temperature_month, input23_Temperature,\
                          input24_Load_day, input24_Load_week, input24_Load_month, input24_Load_hour, input24_Temperature_day, input24_Temperature_week, input24_Temperature_month, input24_Temperature,\
                          input_Load_max, input_Load_min, input_season, input_weekday, input_festival], \
                  outputs=[output])
    return model
      
model = get_model()
# Model compilation
model.compile(optimizer='adam', loss=penalized_loss)
model.save_weights('model.h5')

# Reset random weights
def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)

mape_list = []
history_list = []
pred_list = []

from keras.callbacks import LearningRateScheduler
def lr_scheduler1(epoch, mode=None):
    lr = 0.001
    return lr

def lr_scheduler2(epoch, mode=None):
    lr = 0.0005
    return lr

def lr_scheduler3(epoch, mode=None):
    lr = 0.0003
    return lr

# Different learning rates
scheduler1 = LearningRateScheduler(lr_scheduler1)
scheduler2 = LearningRateScheduler(lr_scheduler2)
scheduler3 = LearningRateScheduler(lr_scheduler3)

# The number of training times, the number of test sets, the batch size, and the number of weights are set.
NumberOfRepeatedTraining = 5
NumberOfTestSets = 365
BATCH_SIZE = 32
NumberOfWeightsSaved = 11

# generation data set
input_train_fit, output_train_fit = get_XY(Input_train, Output_train)
input_test_pred, output_test_pred = get_XY(Input_test, Output_test)

# model training
for i in range(NumberOfRepeatedTraining):
    model.load_weights('model.h5')
    shuffle_weights(model)
    
    history_1 = model.fit(input_train_fit, output_train_fit, epochs=500, batch_size=BATCH_SIZE, callbacks=[scheduler1])
    
    model.save_weights('complete' + str(i+1) +'1_weights.h5')    
    print(str(i) + ' 1')
    
    history_2 = model.fit(input_train_fit, output_train_fit, epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler1])
    
    model.save_weights('complete' + str(i+1) +'2_weights.h5') 
    print(str(i) + ' 2')
    
    history_3 = model.fit(input_train_fit, output_train_fit, epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler1])
    
    model.save_weights('complete' + str(i+1) +'3_weights.h5') 
    print(str(i) + ' 3')  
            
    history_4 = model.fit(input_train_fit, output_train_fit, epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler2])
    
    model.save_weights('complete' + str(i+1) +'4_weights.h5')    
    print(str(i) + ' 4')  
        
    history_5 = model.fit(input_train_fit, output_train_fit, epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler2])
    
    model.save_weights('complete' + str(i+1) +'5_weights.h5')    
    print(str(i) + ' 5')  
        
    history_6 = model.fit(input_train_fit, output_train_fit, epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler2])
    
    model.save_weights('complete' + str(i+1) +'6_weights.h5')    
    print(str(i) + ' 6')  
    
    history_7 = model.fit(input_train_fit, output_train_fit, epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler2])
    
    model.save_weights('complete' + str(i+1) +'7_weights.h5')    
    print(str(i) + ' 7')  
    
    history_8 = model.fit(input_train_fit, output_train_fit, epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler3])
    
    model.save_weights('complete' + str(i+1) +'8_weights.h5')    
    print(str(i) + ' 8')  
    
    history_9 = model.fit(input_train_fit, output_train_fit, epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler3])
    
    model.save_weights('complete' + str(i+1) +'9_weights.h5')    
    print(str(i) + ' 9')  
    
    history_10 = model.fit(input_train_fit, output_train_fit, epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler3])
    
    model.save_weights('complete' + str(i+1) +'10_weights.h5')    
    print(str(i) + ' 10')  

    history_11 = model.fit(input_train_fit, output_train_fit, epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler3])
    
    model.save_weights('complete' + str(i+1) +'11_weights.h5')    
    print(str(i) + ' 11') 
    
    history_list.append([history_1, history_2, history_3, history_4, history_5, history_6, history_7, history_8, history_9, history_10, history_11])
    
# Save the training time
elapsed = time.perf_counter()-start
print('Elapsed %.3f seconds.' %elapsed)
elapsed=np.array(elapsed).reshape(-1)
dataframe=pd.DataFrame({'time':elapsed})
pd.DataFrame(dataframe).to_csv('time.csv', index=False)

# Test results (MAPE, MAE, RMSE)
from tqdm import tqdm
loss_mape = np.zeros((NumberOfWeightsSaved, NumberOfWeightsSaved))
loss1_mape = np.zeros((NumberOfWeightsSaved, NumberOfWeightsSaved))
loss_mae = np.zeros((NumberOfWeightsSaved, NumberOfWeightsSaved))
loss1_mae = np.zeros((NumberOfWeightsSaved, NumberOfWeightsSaved))
loss_rmse = np.zeros((NumberOfWeightsSaved, NumberOfWeightsSaved))
loss1_rmse = np.zeros((NumberOfWeightsSaved, NumberOfWeightsSaved))
for i in tqdm(range(0, NumberOfWeightsSaved)):
    for j in range(i, NumberOfWeightsSaved):
        p = np.zeros((NumberOfRepeatedTraining*(j-i+1),24*NumberOfTestSets))
        p1 = np.zeros((NumberOfRepeatedTraining*(j-i+1),24*NumberOfTestSets))
        for k in range(NumberOfRepeatedTraining):
            for l in range(i,j+1):
                print(i) 
                model.load_weights('complete' + str(k+1) + str(l+1) + '_weights.h5')
                pred = model.predict(input_test_pred)
                pred1 = pred[0:365,:]
                pred2 = pred[365:(365+365),:]
                p[k*(j-i+1)+l-i,:] = pred1.reshape(24*NumberOfTestSets) 
                p1[k*(j-i+1)+l-i,:] = pred2.reshape(24*NumberOfTestSets) 
        pred_eval = np.mean(p, axis = 0)
        pred_eval1 = np.mean(p1, axis = 0)
        Y_test_eval = np.array(Output_test).transpose().reshape(24*(365+365))
        mape = np.mean(np.divide(np.abs(Y_test_eval[0*24:365*24] - pred_eval), Y_test_eval[0*24:365*24]))
        mape1 = np.mean(np.divide(np.abs(Y_test_eval[365*24:(365+365)*24] - pred_eval1), Y_test_eval[365*24:(365+365)*24]))
        mae = np.mean(np.abs(Y_test_eval[0*24:365*24] - pred_eval))
        mae1 = np.mean(np.abs(Y_test_eval[365*24:(365+365)*24] - pred_eval))
        rmse = np.sqrt(np.mean(np.square(Y_test_eval[0*24:365*24] - pred_eval)))
        rmse1 = np.sqrt(np.mean(np.square(Y_test_eval[365*24:(365+365)*24] - pred_eval)))
        loss_mape[i,j] = mape
        loss1_mape[i,j] = mape1
        loss_mae[i,j] = mae
        loss1_mae[i,j] = mae1
        loss_rmse[i,j] = rmse
        loss1_rmse[i,j] = rmse1