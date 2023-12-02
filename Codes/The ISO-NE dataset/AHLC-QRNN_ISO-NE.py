# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 08:44:50 2023

@author: Long Luo
Some of the data entry code refers to the code in paper "Short-Term_Load_Forecasting_With_Deep_Residual_Networks".
"""
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, concatenate, add, Dropout, Flatten, Conv1D, AveragePooling1D
from keras import backend as K
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.framework.ops import disable_eager_execution
import skfuzzy as fuzz
import datetime
from skfuzzy import control as ctrl
disable_eager_execution()
import time
start = 0
start = time.perf_counter()

parse_dates = ['date']
df = pd.read_csv('selected_data_ISONE.csv', parse_dates=parse_dates, index_col='date')
#Get daily maximum and minimum loads
MaximumLoadDays = df.groupby('date').demand.max().values
MinimumLoadDays = df.groupby('date').demand.min().values
#Get daily loads and temperatures
Load = df.demand.values
Temperature = df.temperature.values
# Handling of maximum and minimum load shapes to facilitate subsequent data processing
MaximumLoad = np.zeros(len(Load))
MinimumLoad = np.zeros(len(Load))
for i in range(len(Load)):
    n_day = int(i/24)
    MaximumLoad[i] = MaximumLoadDays[n_day]
    MinimumLoad[i] = MinimumLoadDays[n_day]
# normalization based on peak values
MaximumLoad = MaximumLoad / 24000
MinimumLoad = MinimumLoad / 24000
Load = Load / 24000
Temperature = Temperature / 100

# add weekday info to the dataset
# the initial value for iter_weekday corresponds to the first day of the dataset
iter_weekday = 6
weekday = np.zeros((24*4324,))
for i in range(4324):
    mod = np.mod(iter_weekday, 7)
    for j in range(24):
        if (mod == 6) or (mod == 0):
            weekday[i*24 + j] = 0
        else:
            weekday[i*24 + j] = 1
    iter_weekday += 1

# Add white noise simulation predictions
def awgn(x, seed=7):
    mu = 0
    sigma = 0.01
    np.random.seed(seed)  
    for i in range(x.shape[0]):
        x[i] += random.gauss(mu, sigma)
    return x

# add festival info to the dataset
iter_date = datetime.date(2003, 3, 1)            
festival = np.zeros((24*4324,))
for i in range(4324):
    month = iter_date.month
    day = iter_date.day
    for j in range(24):
        if (month == 7) and (day == 4):
            festival[i*24 + j] = 1
        if (month == 11) and (iter_date.weekday() == 4) and (day + 7 > 30):
            festival[i*24 + j] = 1
        if (month == 12) and (day == 25):
            festival[i*24 + j] = 1
    iter_date = iter_date + datetime.timedelta(1) 
    
# add season info to the dataset
iter_date = datetime.date(2003, 3, 1)            
season = np.zeros((24*4324,))         
for i in range(4324):
    month = iter_date.month
    day = iter_date.day
    for j in range(24):            
        if (month == 4) | (month == 5) | ((month == 3) and (day > 7)) | ((month == 6) and (day < 8)):
            season[i*24 + j] = 0
        elif (month == 7) | (month == 8) | ((month == 6) and (day > 7)) | ((month == 9) and (day < 8)):
            season[i*24 + j] = 1
        elif (month == 10) | (month == 11) | ((month == 9) and (day > 7)) | ((month == 12) and (day < 8)):
            season[i*24 + j] = 2
        elif (month == 1) | (month == 2) | ((month == 12) and (day > 7)) | ((month == 3) and (day < 8)):
            season[i*24 + j] = 3
    iter_date = iter_date + datetime.timedelta(1)
    
#Segmentation of input data       
def data_split_input(Load, Temperature, MaximumLoad, MinimumLoad, season, weekday, festival, number_of_train_days):
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
    len_dataset = Load.shape[0]
    num_sample = len_dataset-2016
    # 2016 hours (28*3 days) is needed so that we can formulate the first datapoint
    
    for i in range(2016,len_dataset):   
        # the demand values of the most recent 24 hours
        input_1.append(Load[i-24:i])        
        # multiple demand values every 24 hours within a week
        index_input_21 = [i-24, i-48, i-72, i-96, i-120, i-144, i-168]
        input_21_D.append(Load[index_input_21])
        input_21_T.append(Temperature[index_input_21])        
        # multiple demand values every week within two months
        index_input_22 = [i-168, i-336, i-504, i-672, i-840, i-1008, i-1176, i-1344]
        input_22_D.append(Load[index_input_22])
        input_22_T.append(Temperature[index_input_22])        
        # multiple demand values every month within several months
        index_input_23 = [i-672, i-1344, i-2016]
        input_23_D.append(Load[index_input_23])
        input_23_T.append(Temperature[index_input_23])        
        input_3.append(Temperature[i])
        input_4.append(MaximumLoad[i])
        input_5.append(MinimumLoad[i])
        output.append(Load[i])        
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
    Input_3 = awgn(Input_3,1)#White Noise Add Location
    Input_4 = np.array(input_4)
    Input_5 = np.array(input_5)
    Input_season = np.array(input_season)
    Input_weekday = np.array(input_weekday)
    Input_festival = np.array(input_festival)    
    num_train = number_of_train_days * 24
    num_val = 365*24   
    Input_train = []
    Input_val = []
    Input_test = []
    # we prepare 24 sets of data for the 24 sub-networks, each sub-network is aimed at forecasting the load of one hour of the next day
    for i in range(24):
        Input_train.append([Input_1[i:num_train-num_val:24, :24-i], Input_21_D[i:num_train-num_val:24, :], Input_21_T[i:num_train-num_val:24, :], Input_22_D[i:num_train-num_val:24, :], Input_22_T[i:num_train-num_val:24, :], Input_23_D[i:num_train-num_val:24, :], Input_23_T[i:num_train-num_val:24, :], Input_3[i:num_train-num_val:24], Input_4[i:num_train-num_val:24], Input_5[i:num_train-num_val:24], Input_season[i:num_train-num_val:24, :], Input_weekday[i:num_train-num_val:24, :], Input_festival[i:num_train-num_val:24, :]])
        Input_val.append([Input_1[num_train-num_val+i:num_train:24, :24-i], Input_21_D[num_train-num_val+i:num_train:24, :], Input_21_T[num_train-num_val+i:num_train:24, :], Input_22_D[num_train-num_val+i:num_train:24, :], Input_22_T[num_train-num_val+i:num_train:24, :], Input_23_D[num_train-num_val+i:num_train:24, :], Input_23_T[num_train-num_val+i:num_train:24, :], Input_3[num_train-num_val+i:num_train:24], Input_4[num_train-num_val+i:num_train:24], Input_5[num_train-num_val+i:num_train:24], Input_season[num_train-num_val+i:num_train:24, :], Input_weekday[num_train-num_val+i:num_train:24, :], Input_festival[num_train-num_val+i:num_train:24, :]])
        Input_test.append([Input_1[num_train+i:num_sample:24, :24-i], Input_21_D[num_train+i:num_sample:24, :], Input_21_T[num_train+i:num_sample:24, :], Input_22_D[num_train+i:num_sample:24, :], Input_22_T[num_train+i:num_sample:24, :], Input_23_D[num_train+i:num_sample:24, :], Input_23_T[num_train+i:num_sample:24, :], Input_3[num_train+i:num_sample:24], Input_4[num_train+i:num_sample:24], Input_5[num_train+i:num_sample:24], Input_season[num_train+i:num_sample:24, :], Input_weekday[num_train+i:num_sample:24, :], Input_festival[num_train+i:num_sample:24, :]])
    return (Input_train, Input_val, Input_test)

#Segmentation of output data
def data_split_output(Load, number_of_train_days):
    output = []
    len_dataset = Load.shape[0]
    num_sample = len_dataset-2016 
    for i in range(2016,len_dataset):   
        # the demand values of the most recent 24 hours        
        output.append(Load[i])            
    num_train = number_of_train_days * 24
    num_val = 365 * 24
    Output_1 = np.array(output)
    Output_train = []
    Output_val = []
    Output_test = []
    for i in range(24):
        Output_train.append(Output_1[i:num_train-num_val:24])
        Output_val.append(Output_1[num_train-num_val+i:num_train:24])
        Output_test.append(Output_1[num_train+i:num_sample:24])
    return ( Output_train, Output_val, Output_test)

# num_pre_days: the number of days we need before we can get the first sample, in this case: 3*28 days 
number_of_pre_days = 84
number_of_days = 1401
number_of_test_days = 365
number_of_train_days = 952
number_of_data_points = number_of_days * 24
number_of_days_start = number_of_days - number_of_pre_days - number_of_test_days - number_of_train_days
start_data_point = number_of_days_start * 24
Input_train, Input_val, Input_test = data_split_input(Load[start_data_point: start_data_point + number_of_data_points], Temperature[start_data_point: start_data_point + number_of_data_points], MaximumLoad[start_data_point: start_data_point + number_of_data_points], MinimumLoad[start_data_point: start_data_point + number_of_data_points], season[start_data_point: start_data_point + number_of_data_points], weekday[start_data_point: start_data_point + number_of_data_points], festival[start_data_point: start_data_point + number_of_data_points], number_of_train_days)
Output_train, Output_val, Output_test = data_split_output(Load[start_data_point: start_data_point + number_of_data_points], number_of_train_days)

## ----------------------------------------------------------------------------
#Initialization Inputs
def get_input(hour):
    input_Load_day = Input(shape=(7,), name='input'+str(hour)+'_Dd')
    input_Load_week = Input(shape=(8,), name='input'+str(hour)+'_Dw')
    input_Load_month = Input(shape=(3,), name='input'+str(hour)+'_Dm')
    input_Load_hour = Input(shape=(24-hour+1,), name='input'+str(hour)+'_Dr')
    input_Temperature_day = Input(shape=(7,), name='input'+str(hour)+'_Td')
    input_Temperature_week = Input(shape=(8,), name='input'+str(hour)+'_Tw')
    input_Temperature_month = Input(shape=(3,), name='input'+str(hour)+'_Tm')
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
input_D_max = Input(shape=(1,), name = 'input_D_max')
input_D_min = Input(shape=(1,), name = 'input_D_min')
input_season = Input(shape=(4,), name = 'input_season')
input_weekday = Input(shape=(2,), name = 'input_weekday')
input_festival = Input(shape=(2,), name = 'input_festival')

#Setting up the model's intermediate key layers
num_dense1 = 10
num_dense2 = 10
num_dense1_Load_hour = 10 
DenseConcatMid = Dense(num_dense2, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.0001))
DenseConcatLoad_hour = Dense(num_dense1_Load_hour, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.0001))
DenseMid = Dense(80, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.0001))
DenseConcat1 = Dense(num_dense2, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.0001))
DenseConcat2 = Dense(num_dense2, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.0001))
DenseConcat3 = Dense(num_dense2, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.0001))
DenseConcatMid1 = Dense(num_dense2, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.0001))
DenseConcatLoad_hour1 = Dense(num_dense1_Load_hour, activation='selu', kernel_initializer = 'lecun_normal', kernel_regularizer=l2(0.0001))
DenseMid1 = Dense(80, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.0001))
DenseConcat11 = Dense(num_dense2, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.0001))
DenseConcat21 = Dense(num_dense2, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.0001))
DenseConcat31 = Dense(num_dense2, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.0001))

#Building the basic model
def get_layer_1(hour, input_Load_day, input_Load_week, input_Load_month, input_Load_hour, input_Temperature_day, input_Temperature_week, input_Temperature_month, input_Temperature, output_pre=[]):
    input_Load_day=tf.reshape(input_Load_day, [-1, 7, 1])
    Conv1D_Load_day = Conv1D(1, 4, activation= 'selu', kernel_initializer='lecun_normal')(input_Load_day)
    Pooling1D_Load_day = AveragePooling1D(2, strides=1)(Conv1D_Load_day)
    Flatten_Load_day = Flatten()(Pooling1D_Load_day)
    input_Load_week=tf.reshape(input_Load_week, [-1, 8, 1])
    Conv1D_Load_week = Conv1D(1, 4, activation='selu', kernel_initializer='lecun_normal')(input_Load_week)
    Pooling1D_Load_week = AveragePooling1D(2, strides=1)(Conv1D_Load_week)
    Flatten_Load_week = Flatten()(Pooling1D_Load_week)
    input_Load_month=tf.reshape(input_Load_month, [-1, 3, 1])
    Conv1D_Load_month = Conv1D(1, 2, activation='selu', kernel_initializer='lecun_normal')(input_Load_month)
    Pooling1D_Load_month = AveragePooling1D(2, strides=1)(Conv1D_Load_month)
    Flatten_Load_month = Flatten()(Pooling1D_Load_month)
    input_Temperature_day=tf.reshape(input_Temperature_day, [-1, 7, 1])
    Conv1D_Temperature_day = Conv1D(1, 4, activation='selu', kernel_initializer='lecun_normal')(input_Temperature_day)
    Pooling1D_Temperature_day = AveragePooling1D(2, strides=1)(Conv1D_Temperature_day)
    Flatten_Temperature_day = Flatten()(Pooling1D_Temperature_day)
    input_Temperature_week=tf.reshape(input_Temperature_week, [-1, 8, 1])
    Conv1D_Temperature_week = Conv1D(1, 4, activation='selu', kernel_initializer='lecun_normal')(input_Temperature_week)
    Pooling1D_Temperature_week = AveragePooling1D(2,strides=1)(Conv1D_Temperature_week)
    Flatten_Temperature_week = Flatten()(Pooling1D_Temperature_week)
    input_Temperature_month=tf.reshape(input_Temperature_month,[-1,3,1])
    Conv1D_Temperature_month = Conv1D(1, 2,activation= 'selu', kernel_initializer = 'lecun_normal')(input_Temperature_month)
    Pooling1D_Temperature_month = AveragePooling1D(2, strides=1)(Conv1D_Temperature_month)
    Flatten_Temperature_month = Flatten()(Pooling1D_Temperature_month)
    #B structure construction  
    def industrial_load(hour, Flatten_Load_day, Flatten_Load_week, Flatten_Load_month, input_Load_hour):     
        dense_Load_day = Dense(num_dense1, activation='selu', kernel_initializer='lecun_normal')(Flatten_Load_day)
        dense_Load_week = Dense(num_dense1, activation='selu', kernel_initializer='lecun_normal')(Flatten_Load_week)
        dense_Load_month = Dense(num_dense1, activation='selu', kernel_initializer='lecun_normal')(Flatten_Load_month)
        dense_concat1 = DenseConcat11(dense_Load_day)
        dense_concat2 = DenseConcat21(dense_Load_week)
        dense_concat3 = DenseConcat31(dense_Load_month)
        concat_date_info = concatenate([input_season, input_weekday])
        dense_concat_date_info_1 = Dense(5, activation='selu', kernel_initializer='lecun_normal')(concat_date_info)
        dense_concat_date_info_2 = Dense(5, activation='selu', kernel_initializer='lecun_normal')(concat_date_info)
        concat_mid = concatenate([dense_concat1, dense_concat2, dense_concat3, dense_concat_date_info_1, input_festival])
        dense_concat_mid = DenseConcatMid1(concat_mid)
        
        if output_pre == []:
            input_Load_hour=tf.reshape(input_Load_hour, [-1, 24, 1])
            Conv1D_Load_hour = Conv1D(1, 4, activation='selu', kernel_initializer='lecun_normal')(input_Load_hour)
            Pooling1D_Load_hour = AveragePooling1D(2, strides=1)(Conv1D_Load_hour)
            Flatten_Load_hour = Flatten()(Pooling1D_Load_hour)
            dense_Load_hour = Dense(num_dense1_Load_hour, activation='selu', kernel_initializer='lecun_normal')(Flatten_Load_hour)
        else:
            concat_Load_hour = concatenate([input_Load_hour] + output_pre)
            concat_Load_hour=tf.reshape(concat_Load_hour, [-1, 24, 1])
            Conv1D_Load_hour = Conv1D(1, 4, activation='selu', kernel_initializer='lecun_normal')(concat_Load_hour)
            Pooling1D_Load_hour = AveragePooling1D(2, strides=1)(Conv1D_Load_hour)
            Flatten_Load_hour = Flatten()(Pooling1D_Load_hour)
            dense_Load_hour = Dense(num_dense1_Load_hour, activation='selu', kernel_initializer='lecun_normal')(Flatten_Load_hour)

        dense_4 = DenseConcatLoad_hour1(concatenate([dense_Load_hour, dense_concat_date_info_2]))
        concat = concatenate([dense_concat_mid, dense_4])
        dense_mid = DenseMid1(concat)
        dropout_mid = Dropout(0.5)(dense_mid)
        output = Dense(1, activation='linear', kernel_initializer='lecun_normal')(dropout_mid)
        output1 = Dense(99, activation='linear', kernel_initializer='lecun_normal')(dropout_mid)
        return output,output1
    #E structure construction
    def elastic_load(hour, Flatten_Load_day, Flatten_Load_week, Flatten_Load_month, input_Load_hour, Flatten_Temperature_day, Flatten_Temperature_week, Flatten_Temperature_month, input_Temperature):     
        dense_Load_day = Dense(num_dense1, activation='selu', kernel_initializer='lecun_normal')(Flatten_Load_day)
        dense_Load_week = Dense(num_dense1, activation='selu', kernel_initializer='lecun_normal')(Flatten_Load_week)
        dense_Load_month = Dense(num_dense1, activation='selu', kernel_initializer='lecun_normal')(Flatten_Load_month)
        dense_Temperature_day = Dense(num_dense1, activation='selu', kernel_initializer='lecun_normal')(Flatten_Temperature_day)
        dense_Temperature_week = Dense(num_dense1, activation='selu', kernel_initializer='lecun_normal')(Flatten_Temperature_day)
        dense_Temperature_month = Dense(num_dense1, activation='selu', kernel_initializer='lecun_normal')(Flatten_Temperature_month)
        concat1 = concatenate([dense_Load_day, dense_Temperature_day])
        dense_concat1 = DenseConcat1(concat1)
        concat2 = concatenate([dense_Load_week, dense_Temperature_week])
        dense_concat2 = DenseConcat2(concat2)
        concat3 = concatenate([dense_Load_month, dense_Temperature_month])
        dense_concat3 = DenseConcat3(concat3)
        concat_date_info = concatenate([input_season, input_weekday])
        dense_concat_date_info_1 = Dense(5, activation='selu', kernel_initializer='lecun_normal')(concat_date_info)
        dense_concat_date_info_2 = Dense(5, activation='selu', kernel_initializer='lecun_normal')(concat_date_info)
        concat_mid = concatenate([dense_concat1, dense_concat2, dense_concat3, dense_concat_date_info_1, input_festival])
        dense_concat_mid = DenseConcatMid(concat_mid)
        
        if output_pre == []:
            input_Load_hour=tf.reshape(input_Load_hour, [-1, 24, 1])
            Conv1D_Load_hour = Conv1D(1, 4, activation='selu', kernel_initializer='lecun_normal')(input_Load_hour)
            Pooling1D_Load_hour = AveragePooling1D(2, strides=1)(Conv1D_Load_hour)
            Flatten_Load_hour = Flatten()(Pooling1D_Load_hour)
            dense_Load_hour = Dense(num_dense1_Load_hour, activation='selu', kernel_initializer='lecun_normal')(Flatten_Load_hour)
        else:
            concat_Load_hour = concatenate([input_Load_hour] + output_pre)
            concat_Load_hour=tf.reshape(concat_Load_hour, [-1, 24, 1])
            Conv1D_Load_hour = Conv1D(1, 4, activation='selu', kernel_initializer='lecun_normal')(concat_Load_hour)
            Pooling1D_Load_hour = AveragePooling1D(2, strides=1)(Conv1D_Load_hour)
            Flatten_Load_hour = Flatten()(Pooling1D_Load_hour)
            dense_Load_hour = Dense(num_dense1_Load_hour, activation='selu', kernel_initializer='lecun_normal')(Flatten_Load_hour)

        dense_4 = DenseConcatLoad_hour(concatenate([dense_Load_hour, dense_concat_date_info_2]))
        concat = concatenate([dense_concat_mid, dense_4, input_Temperature])
        dense_mid = DenseMid(concat)
        dropout_mid = Dropout(0.5)(dense_mid)
        output = Dense(1, activation='linear', kernel_initializer='lecun_normal')(dropout_mid)
        output1 = Dense(99, activation='linear', kernel_initializer='lecun_normal')(dropout_mid)
        return output, output1
    #reconstruct
    industrial_loads, industrial_loads1 = industrial_load(hour, Flatten_Load_day, Flatten_Load_week, Flatten_Load_month, input_Load_hour)
    elastic_loads, elastic_loads1 = elastic_load(hour, Flatten_Load_day, Flatten_Load_week, Flatten_Load_month, input_Load_hour, Flatten_Temperature_day, Flatten_Temperature_week, Flatten_Temperature_month, input_Temperature)
    output = add([industrial_loads, elastic_loads])
    output1 = add([industrial_loads1, elastic_loads1])
    output_pre_new = output_pre + [output]
    return (output, output1, output_pre_new)
#Connecting 24-hour infrastructure      
model1_output1, model1_output1_1, model1_output_pre1 = get_layer_1(1, input1_Load_day, input1_Load_week, input1_Load_month, input1_Load_hour, input1_Temperature_day, input1_Temperature_week, input1_Temperature_month, input1_Temperature)
model1_output2, model1_output2_1, model1_output_pre2 = get_layer_1(2, input2_Load_day, input2_Load_week, input2_Load_month, input2_Load_hour, input2_Temperature_day, input2_Temperature_week, input2_Temperature_month, input2_Temperature, model1_output_pre1)
model1_output3, model1_output3_1, model1_output_pre3 = get_layer_1(3, input3_Load_day, input3_Load_week, input3_Load_month, input3_Load_hour, input3_Temperature_day, input3_Temperature_week, input3_Temperature_month, input3_Temperature, model1_output_pre2)
model1_output4, model1_output4_1, model1_output_pre4 = get_layer_1(4, input4_Load_day, input4_Load_week, input4_Load_month, input4_Load_hour, input4_Temperature_day, input4_Temperature_week, input4_Temperature_month, input4_Temperature, model1_output_pre3)
model1_output5, model1_output5_1, model1_output_pre5 = get_layer_1(5, input5_Load_day, input5_Load_week, input5_Load_month, input5_Load_hour, input5_Temperature_day, input5_Temperature_week, input5_Temperature_month, input5_Temperature, model1_output_pre4)
model1_output6, model1_output6_1, model1_output_pre6 = get_layer_1(6, input6_Load_day, input6_Load_week, input6_Load_month, input6_Load_hour, input6_Temperature_day, input6_Temperature_week, input6_Temperature_month, input6_Temperature, model1_output_pre5)
model1_output7, model1_output7_1, model1_output_pre7 = get_layer_1(7, input7_Load_day, input7_Load_week, input7_Load_month, input7_Load_hour, input7_Temperature_day, input7_Temperature_week, input7_Temperature_month, input7_Temperature, model1_output_pre6)
model1_output8, model1_output8_1, model1_output_pre8 = get_layer_1(8, input8_Load_day, input8_Load_week, input8_Load_month, input8_Load_hour, input8_Temperature_day, input8_Temperature_week, input8_Temperature_month, input8_Temperature, model1_output_pre7)
model1_output9, model1_output9_1, model1_output_pre9 = get_layer_1(9, input9_Load_day, input9_Load_week, input9_Load_month, input9_Load_hour, input9_Temperature_day, input9_Temperature_week, input9_Temperature_month, input9_Temperature, model1_output_pre8)
model1_output10, model1_output10_1, model1_output_pre10 = get_layer_1(10, input10_Load_day, input10_Load_week, input10_Load_month, input10_Load_hour, input10_Temperature_day, input10_Temperature_week, input10_Temperature_month, input10_Temperature, model1_output_pre9)
model1_output11, model1_output11_1, model1_output_pre11 = get_layer_1(11, input11_Load_day, input11_Load_week, input11_Load_month, input11_Load_hour, input11_Temperature_day, input11_Temperature_week, input11_Temperature_month, input11_Temperature, model1_output_pre10)
model1_output12, model1_output12_1, model1_output_pre12 = get_layer_1(12, input12_Load_day, input12_Load_week, input12_Load_month, input12_Load_hour, input12_Temperature_day, input12_Temperature_week, input12_Temperature_month, input12_Temperature, model1_output_pre11)
model1_output13, model1_output13_1, model1_output_pre13 = get_layer_1(13, input13_Load_day, input13_Load_week, input13_Load_month, input13_Load_hour, input13_Temperature_day, input13_Temperature_week, input13_Temperature_month, input13_Temperature, model1_output_pre12)
model1_output14, model1_output14_1, model1_output_pre14 = get_layer_1(14, input14_Load_day, input14_Load_week, input14_Load_month, input14_Load_hour, input14_Temperature_day, input14_Temperature_week, input14_Temperature_month, input14_Temperature, model1_output_pre13)
model1_output15, model1_output15_1, model1_output_pre15 = get_layer_1(15, input15_Load_day, input15_Load_week, input15_Load_month, input15_Load_hour, input15_Temperature_day, input15_Temperature_week, input15_Temperature_month, input15_Temperature, model1_output_pre14)
model1_output16, model1_output16_1, model1_output_pre16 = get_layer_1(16, input16_Load_day, input16_Load_week, input16_Load_month, input16_Load_hour, input16_Temperature_day, input16_Temperature_week, input16_Temperature_month, input16_Temperature, model1_output_pre15)
model1_output17, model1_output17_1, model1_output_pre17 = get_layer_1(17, input17_Load_day, input17_Load_week, input17_Load_month, input17_Load_hour, input17_Temperature_day, input17_Temperature_week, input17_Temperature_month, input17_Temperature, model1_output_pre16)
model1_output18, model1_output18_1, model1_output_pre18 = get_layer_1(18, input18_Load_day, input18_Load_week, input18_Load_month, input18_Load_hour, input18_Temperature_day, input18_Temperature_week, input18_Temperature_month, input18_Temperature, model1_output_pre17)
model1_output19, model1_output19_1, model1_output_pre19 = get_layer_1(19, input19_Load_day, input19_Load_week, input19_Load_month, input19_Load_hour, input19_Temperature_day, input19_Temperature_week, input19_Temperature_month, input19_Temperature, model1_output_pre18)
model1_output20, model1_output20_1, model1_output_pre20 = get_layer_1(20, input20_Load_day, input20_Load_week, input20_Load_month, input20_Load_hour, input20_Temperature_day, input20_Temperature_week, input20_Temperature_month, input20_Temperature, model1_output_pre19)
model1_output21, model1_output21_1, model1_output_pre21 = get_layer_1(21, input21_Load_day, input21_Load_week, input21_Load_month, input21_Load_hour, input21_Temperature_day, input21_Temperature_week, input21_Temperature_month, input21_Temperature, model1_output_pre20)
model1_output22, model1_output22_1, model1_output_pre22 = get_layer_1(22, input22_Load_day, input22_Load_week, input22_Load_month, input22_Load_hour, input22_Temperature_day, input22_Temperature_week, input22_Temperature_month, input22_Temperature, model1_output_pre21)
model1_output23, model1_output23_1, model1_output_pre23 = get_layer_1(23, input23_Load_day, input23_Load_week, input23_Load_month, input23_Load_hour, input23_Temperature_day, input23_Temperature_week, input23_Temperature_month, input23_Temperature, model1_output_pre22)
model1_output24, model1_output24_1, model1_output_pre24 = get_layer_1(24, input24_Load_day, input24_Load_week, input24_Load_month, input24_Load_hour, input24_Temperature_day, input24_Temperature_week, input24_Temperature_month, input24_Temperature, model1_output_pre23)
#Constructing Cumulative Hidden Layer Connection Structures
def q_hour_layer(model1_output_1, model1_output=[]):
    if model1_output == []:
        output_bound1 = Dense(99, activation='linear',
                            kernel_initializer='lecun_normal')(model1_output_1)
    else:
        output_pre = concatenate([model1_output_1]+model1_output)  
        output_bound1 = Dense(99, activation='linear',
                            kernel_initializer='lecun_normal')(output_pre)
    model1_output = model1_output + [model1_output_1]    
    return output_bound1, model1_output

output_final_1, model1_output1 = q_hour_layer(model1_output1_1)
output_final_2, model1_output2 = q_hour_layer(model1_output2_1, model1_output1)
output_final_3, model1_output3 = q_hour_layer(model1_output3_1, model1_output2)
output_final_4, model1_output4 = q_hour_layer(model1_output4_1, model1_output3)
output_final_5, model1_output5 = q_hour_layer(model1_output5_1, model1_output4)
output_final_6, model1_output6 = q_hour_layer(model1_output6_1, model1_output5)
output_final_7, model1_output7 = q_hour_layer(model1_output7_1, model1_output6)
output_final_8, model1_output8 = q_hour_layer(model1_output8_1, model1_output7)
output_final_9, model1_output9 = q_hour_layer(model1_output9_1, model1_output8)
output_final_10, model1_output10 = q_hour_layer(model1_output10_1, model1_output9)
output_final_11, model1_output11 = q_hour_layer(model1_output11_1, model1_output10)
output_final_12, model1_output12 = q_hour_layer(model1_output12_1, model1_output11)
output_final_13, model1_output13 = q_hour_layer(model1_output13_1, model1_output12)
output_final_14, model1_output14 = q_hour_layer(model1_output14_1, model1_output13)
output_final_15, model1_output15 = q_hour_layer(model1_output15_1, model1_output14)
output_final_16, model1_output16 = q_hour_layer(model1_output16_1, model1_output15)
output_final_17, model1_output17 = q_hour_layer(model1_output17_1, model1_output16)
output_final_18, model1_output18 = q_hour_layer(model1_output18_1, model1_output17)
output_final_19, model1_output19 = q_hour_layer(model1_output19_1, model1_output18)
output_final_20, model1_output20 = q_hour_layer(model1_output20_1, model1_output19)
output_final_21, model1_output21 = q_hour_layer(model1_output21_1, model1_output20)
output_final_22, model1_output22 = q_hour_layer(model1_output22_1, model1_output21)
output_final_23, model1_output23 = q_hour_layer(model1_output23_1, model1_output22)
output_final_24, model1_output24 = q_hour_layer(model1_output24_1, model1_output23)

#Initialize the quantile values and construct the loss function
bound01 = []
for i in range(1, 100):
    bound01.append((100-i)/100)
bound1 = []
for i in range(len(bound01)):
    bound1.append(bound01[i])
quantiles1 = np.array(bound1)
def tilted_loss1(y_true, y_pred, quantiles1):
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")
    e = y_true - y_pred
    loss = K.sum(K.mean(K.maximum(quantiles1 * e, (quantiles1 - 1) * e), axis=0))
    # find the sum of average loss of each quantile
    return loss
#Constructing model inputs and outputs and outputs during testing
def get_XY(X, Y):
    X_new = []
    Y_new = []
    y_new = []
    y1_new = []
    for i in range(24):
        X_new.append(X[i][1])
        X_new.append(X[i][3])
        X_new.append(X[i][5])
        X_new.append(X[i][0])
        X_new.append(X[i][2])
        X_new.append(X[i][4])
        X_new.append(X[i][6])
        X_new.append(X[i][7]) # temperature
        y1_new.append(Y[i])
    for i in range(24):
        for j in range(99):
            y_new.append(Y[i])
        Y_new.append(np.squeeze(np.array(y_new)).transpose())
        y_new = []
    X_new = X_new + [X[0][8], X[0][9], X[0][10], X[0][11], X[0][12]]
    return (X_new, Y_new, np.squeeze(np.array(y1_new)).transpose())

# -----------------------------------------------------------------------------
Input_train_fit, Output_train_fit, y1_train_new = get_XY(Input_train, Output_train)
Input_val_fit, Output_val_fit,y1_val_new = get_XY(Input_val, Output_val)
Input_test_pred, Output_test_pred, y1_test_new = get_XY(Input_test, Output_test)

#build a model
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
                      input_D_max, input_D_min, input_season, input_weekday, input_festival], \
                      outputs=[
                               output_final_1, output_final_2, output_final_3, output_final_4, output_final_5, output_final_6,
                               output_final_7, output_final_8, output_final_9, output_final_10, output_final_11, output_final_12,
                               output_final_13, output_final_14, output_final_15, output_final_16, output_final_17, output_final_18,
                               output_final_19, output_final_20, output_final_21, output_final_22, output_final_23, output_final_24])
    return model

#disrupt the weighting function
def shuffle_weights(model, weights=None):
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)

mape_list = []
history_list = []
pred_list = []

#Segmented learning rates and early stop settings
from keras.callbacks import LearningRateScheduler
def lr_scheduler1(epoch, mode=None):
    lr = 0.001
    return lr
def lr_scheduler2(epoch, mode=None):
    lr = 0.0005
    return lr
def lr_scheduler3(epoch, mode=None):
    lr = 0.0001
    return lr
def lr_scheduler4(epoch, mode=None):
    lr = 0.00005
    return lr
def lr_scheduler5(epoch, mode=None):
    lr = 0.00001
    return lr
scheduler1 = LearningRateScheduler(lr_scheduler1)
scheduler2 = LearningRateScheduler(lr_scheduler2)
scheduler3 = LearningRateScheduler(lr_scheduler3)
scheduler4 = LearningRateScheduler(lr_scheduler4)
scheduler5 = LearningRateScheduler(lr_scheduler5)

early_stopping1 = EarlyStopping(monitor='val_loss', patience=35)
early_stopping2 = EarlyStopping(monitor='val_loss', patience=30)
early_stopping3 = EarlyStopping(monitor='val_loss', patience=25)
early_stopping4 = EarlyStopping(monitor='val_loss', patience=20)
early_stopping5 = EarlyStopping(monitor='val_loss', patience=15)
best_weight_dir = r"best_weights_NNQR.hdf5"
model_checkpoint = ModelCheckpoint(best_weight_dir, monitor="val_loss",
                                   save_best_only=True, save_weights_only=True,
                                   verbose=1)

NUM_TEST = 365
BATCH_SIZE = 91
#Constructing and Training Models
for i in range(1):   
    model = get_model()
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: tilted_loss1(y_true, y_pred, quantiles1))
    model.save_weights('model.h5')
    model.load_weights('model.h5')
    shuffle_weights(model)
    history_1 = model.fit(Input_train_fit, Output_train_fit, validation_data=[Input_val_fit, Output_val_fit],\
                        epochs=1000, batch_size=BATCH_SIZE, callbacks=[scheduler1,early_stopping1, model_checkpoint])
    model.save_weights('complete' + str(i+1) + '1_weights.h5')
    print(str(i) + ' 1')
    history_2 = model.fit(Input_train_fit, Output_train_fit, validation_data=[Input_val_fit, Output_val_fit],\
                        epochs=1000, batch_size=BATCH_SIZE, callbacks=[scheduler2, early_stopping2,model_checkpoint])
    model.save_weights('complete' + str(i+1) + '2_weights.h5')
    print(str(i) + ' 2')
    history_3 = model.fit(Input_train_fit, Output_train_fit, validation_data=[Input_val_fit, Output_val_fit],\
                        epochs=1000, batch_size=BATCH_SIZE, callbacks=[scheduler3, early_stopping3, model_checkpoint])
    model.save_weights('complete' + str(i+1) + '3_weights.h5')
    print(str(i) + ' 3')
    history_4 = model.fit(Input_train_fit, Output_train_fit, validation_data = [Input_val_fit, Output_val_fit],\
                        epochs=1000, batch_size=BATCH_SIZE, callbacks=[scheduler4, early_stopping4, model_checkpoint])
    model.save_weights('complete' + str(i+1) + '4_weights.h5')
    print(str(i) + ' 4')
    history_5 = model.fit(Input_train_fit, Output_train_fit, validation_data=[Input_val_fit, Output_val_fit],\
                        epochs=1000, batch_size=BATCH_SIZE, callbacks=[scheduler5, early_stopping5, model_checkpoint])
    model.save_weights('complete' + str(i+1) + '5_weights.h5')
    print(str(i) + ' 5')
    history_list.append([history_1, history_2, history_3, history_4, history_5])
elapsed = time.perf_counter()-start
print('Elapsed %.3f seconds.' %elapsed)
elapsed=np.array(elapsed).reshape(-1)
dataframe=pd.DataFrame({'time': elapsed})
pd.DataFrame(dataframe).to_csv('time.csv', index=False)

#Get validation set and test set predictions    
output1_pred = []
output2_pred = []
output2_pred_val = []
best_weight_dir = r"best_weights_NNQR.hdf5"
for j in range(1):
    for i in range(4, 5):
        model.load_weights(best_weight_dir)
        pred = model.predict(Input_test_pred)
        pred_val = model.predict(Input_val_fit)
        print(i)
for z in range(365):
    for i in range(24):
        for j in range(99):
            output2_pred.append(pred[i][z][j])
pred_eval2 = np.array(output2_pred)
for z in range(365):
    for i in range(24):
        for j in range(99):
            output2_pred_val.append(pred_val[i][z][j])
pred_eval2_val = np.array(output2_pred_val)

#PinballLoss
def PinballLoss(y_true, output1, q):
    underbias = 0
    overbias = 0
    for i in range(99):
        if q == i+1:
            y_pred=output1[:, 98-i]
    for i in range(len(y_true)):
        if y_true[i] >= y_pred[i]:
            underbias = (q/100 * (y_true[i] - y_pred[i])) + underbias
        if y_true[i] < y_pred[i]:
            overbias = ((1 - q/100) * (y_pred[i] - y_true[i])) + overbias
    return underbias + overbias, y_pred
#winkler
def winkler(y_true, output1, q):
    one = 0
    two = 0
    three = 0
    q1 = (100-q)/2
    q2 = 100-(100-q)/2
    for i in range(99):
        if q1 == i+1:
            L = output1[:, 98-i]
    for i in range(99):
        if q2 == i+1:
            U = output1[:, 98-i]
    for i in range(len(y_true)):
        if y_true[i] < L[i]:
            two = ((U[i]-L[i])+2*(L[i]-y_true[i])/(1-(q/100))) + two
        elif y_true[i] > U[i]:
            three = ((U[i]-L[i])+2*(y_true[i]-U[i])/(1-(q/100))) + three
        else:
            one = (U[i]-L[i]) +one
    return one + two + three

#Obtain the inverse normalized prediction (*320 Simulated inverse normalization)
pred_eval2_val = np.array(output2_pred_val).reshape(-1,99)
pred_eval2_val = np.array(pred_eval2_val)*320
pred_eval2_val_sort = -(np.sort(-pred_eval2_val, axis=1, kind='quicksort', order=None))
pred_eval2 = np.array(output2_pred).reshape(-1, 99)
pred_eval2 = np.array(pred_eval2)*320
pred_eval2_sort = -(np.sort(-pred_eval2, axis=1, kind='quicksort', order=None))
y_test = np.array(y1_test_new).reshape(24*365)*320
y_val = np.array(y1_val_new).reshape(24*365)*320

def PL_WS(y_test, pred_eval2):
    valu = 0
    ioo = []
    val = 0
    win90 = 0
    win50 = 0
    for i in range(1, 100):
        val, io = PinballLoss(y_test, pred_eval2, i)
        valu = val+valu
        ioo.append(io)
    win90 = winkler(y_test, pred_eval2, 90) + win90
    win50 = winkler(y_test, pred_eval2, 50) + win50
    win90 = win90/365/24 
    win50 = win50/365/24
    valu=valu/365/24/99
    valu=np.array(valu).reshape(-1)
    win90=np.array(win90).reshape(-1)
    win50=np.array(win50).reshape(-1)
    return valu, win50, win90

# import dataset
df = pd.DataFrame(pred_eval2)
df.to_csv('pred_eval2.csv', index=False, header=False)
df = pd.DataFrame(y_test)
df.to_csv('y_test.csv', index=False, header=False)
df = pd.DataFrame(pred_eval2_val)
df.to_csv('pred_eval2_val.csv', index=False, header=False)
df = pd.DataFrame(y_val)
df.to_csv('y_val.csv', index=False, header=False)

#Processing inputs for adaptive fuzzy control
pred_eval2_val = pd.read_csv('pred_eval2_val.csv', header=None)
y_val = pd.read_csv('y_val.csv', header=None)
pred_eval2_val = np.array(pred_eval2_val)
y_val = np.array(y_val)
pred_eval2_val = pred_eval2_val.reshape(-1, 24, 99)
pred_eval2_val50 = pred_eval2_val[0:, :, :].reshape(-1, 99)
pred_eval2_val_1h = pred_eval2_val50[23:-1, :].reshape(-1, 99)
y_val = y_val.reshape(-1, 24)
y_val50 = y_val[0:-1, :].reshape(-1)
pred_eval2_val_err = pred_eval2_val[0:-1, :, :].reshape(-1, 99)
pred_eval2_val_input = pred_eval2_val[1:, :, :].reshape(-1, 99)

pred_eval2_test = pd.read_csv('pred_eval2.csv', header=None)
y_test = pd.read_csv('y_test.csv', header=None)
pred_eval2_test = np.array(pred_eval2_test)
y_test = np.array(y_test)
pred_eval2_test = pred_eval2_test.reshape(-1,99)
print(pred_eval2_val50[-24:, :].shape)
pred_eval2_test_err = np.append(pred_eval2_val50[-24:, :], pred_eval2_test[0:-24, :], axis=0)
y_val_err = y_val.reshape(-1,1)
y_test_err = np.append(y_val_err[-24:, ], y_test[0:-24, ], axis=0)
pred_eval2_test_1h = np.append(pred_eval2_val50[-1:, :], pred_eval2_test[0:-1, :], axis=0)
error = y_test_err.reshape(-1)-pred_eval2_test_err[:, 50].reshape(-1)

#adaptive fuzzy control
input_1h = ctrl.Antecedent(np.arange(120, 420+1, 1), 'input_1h')
input_error = ctrl.Antecedent(np.arange(-67, 51+1, 1), 'input_error')
output_adjustment = ctrl.Consequent(np.arange(-38, 36+1, 1), 'output_adjustment')

# Designing the affiliation function
input_1h['low'] = fuzz.trimf(input_1h.universe, [120, 120, (120+420)/2])
input_1h['medium'] = fuzz.trimf(input_1h.universe, [158, (158+338)/2, 338])
input_1h['high'] = fuzz.trimf(input_1h.universe, [(120+420)/2, 420, 420])

input_error['low'] = fuzz.trimf(input_error.universe, [-67, -67, 0])
input_error['medium'] = fuzz.trimf(input_error.universe, [-49, 0, 49])
input_error['high'] = fuzz.trimf(input_error.universe, [0, 51, 51])

output_adjustment['negative'] = fuzz.trimf(output_adjustment.universe, [-38, -38, (-38+36)/2])
output_adjustment['zero'] = fuzz.trimf(output_adjustment.universe, [-28, 0, 28])
output_adjustment['positive'] = fuzz.trimf(output_adjustment.universe, [(-38+36)/2, 36, 36])

#Rule
rule1 = ctrl.Rule(input_error['low']&input_1h['low'], output_adjustment['zero'])
rule2 = ctrl.Rule(input_error['low']&input_1h['medium'], output_adjustment['negative'])
rule3 = ctrl.Rule(input_error['low']&input_1h['high'], output_adjustment['negative'])
rule4 = ctrl.Rule(input_error['medium']&input_1h['low'], output_adjustment['zero'])
rule5 = ctrl.Rule(input_error['medium']&input_1h['medium'], output_adjustment['zero'])
rule6 = ctrl.Rule(input_error['medium']&input_1h['high'], output_adjustment['zero'])
rule7 = ctrl.Rule(input_error['high']&input_1h['low'], output_adjustment['positive'])
rule8 = ctrl.Rule(input_error['high']&input_1h['medium'], output_adjustment['positive'])
rule9 = ctrl.Rule(input_error['high']&input_1h['high'], output_adjustment['zero'])

input_1h.view(), input_error.view(), output_adjustment.view()

load_adjustment_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,rule4, rule5, rule6,rule7, rule8, rule9])
load_adjustment = ctrl.ControlSystemSimulation(load_adjustment_ctrl) 

output_adjust = []

adjust = []
mape = []
mape1 = []
r3 = []
r2 = 0
r1 = []
R2 = 0
def adjust_fuzzy_rules(error_value):
    if error_value > 0:
        print(1) 
        #Setting the affiliation function
        output_adjustment = ctrl.Consequent(np.arange(-38, 36+1, 1), 'output_adjustment')
        output_adjustment['negative'] = fuzz.trimf(output_adjustment.universe, [-38, -38, (-38+36)/2])
        output_adjustment['zero'] = fuzz.trimf(output_adjustment.universe, [-28, 0, 28])
        output_adjustment['positive'] = fuzz.trimf(output_adjustment.universe, [(-38+36)/2, 36, 36])
        #Rule
        rule1 = ctrl.Rule(input_error['low']&input_1h['low'], output_adjustment['zero'])
        rule2 = ctrl.Rule(input_error['low']&input_1h['medium'], output_adjustment['negative'])
        rule3 = ctrl.Rule(input_error['low']&input_1h['high'], output_adjustment['negative'])
        rule4 = ctrl.Rule(input_error['medium']&input_1h['low'], output_adjustment['zero'])
        rule5 = ctrl.Rule(input_error['medium']&input_1h['medium'], output_adjustment['zero'])
        rule6 = ctrl.Rule(input_error['medium']&input_1h['high'], output_adjustment['zero'])
        rule7 = ctrl.Rule(input_error['high']&input_1h['low'], output_adjustment['positive'])
        rule8 = ctrl.Rule(input_error['high']&input_1h['medium'], output_adjustment['positive'])
        rule9 = ctrl.Rule(input_error['high']&input_1h['high'], output_adjustment['zero'])

        load_adjustment_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,rule4, rule5, rule6,rule7, rule8, rule9])
        load_adjustment = ctrl.ControlSystemSimulation(load_adjustment_ctrl)
    else:
        print(2)
        #Setting the affiliation function
        output_adjustment = ctrl.Consequent(np.arange(-18, 16+1, 1), 'output_adjustment')
        output_adjustment['negative'] = fuzz.trimf(output_adjustment.universe, [-18, -18, (-18+16)/2])
        output_adjustment['zero'] = fuzz.trimf(output_adjustment.universe, [-10, 0, 10])
        output_adjustment['positive'] = fuzz.trimf(output_adjustment.universe, [(-18+16)/2, 16, 16])
        #Rule
        rule1 = ctrl.Rule(input_error['low']&input_1h['low'], output_adjustment['zero'])
        rule2 = ctrl.Rule(input_error['low']&input_1h['medium'], output_adjustment['negative'])
        rule3 = ctrl.Rule(input_error['low']&input_1h['high'], output_adjustment['negative'])
        rule4 = ctrl.Rule(input_error['medium']&input_1h['low'], output_adjustment['zero'])
        rule5 = ctrl.Rule(input_error['medium']&input_1h['medium'], output_adjustment['zero'])
        rule6 = ctrl.Rule(input_error['medium']&input_1h['high'], output_adjustment['zero'])
        rule7 = ctrl.Rule(input_error['high']&input_1h['low'], output_adjustment['positive'])
        rule8 = ctrl.Rule(input_error['high']&input_1h['medium'], output_adjustment['positive'])
        rule9 = ctrl.Rule(input_error['high']&input_1h['high'], output_adjustment['zero'])

        load_adjustment_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,rule4, rule5, rule6,rule7, rule8, rule9])
        load_adjustment = ctrl.ControlSystemSimulation(load_adjustment_ctrl)
    return load_adjustment
#Access to forecast results
for j in range(int(y_val_err.shape[0]/24)):
    j = j*24
    for i in range(24):
        print(j+i)
        if error[j+i] > 51:
            error[j+i] = 51
        if error[j+i] < -67:
            error[j+i] = -67
        if pred_eval2_test_1h[j+i][50] > 420:
            pred_eval2_test_1h[j+i][50] = 420
        if pred_eval2_test_1h[j+i][50] < 120:
            pred_eval2_test_1h[j+i][50] = 120
        load_adjustment.input['input_error'] = error[j+i]
        load_adjustment.input['input_1h'] = pred_eval2_test_1h[j+i][50]
        if j != 0:
            if r2 == 0 or(r2 > 0 and r1[i]-r3[i] < 0) or (r2 < 0 and r1[i]-r3[i] > 0):
                load_adjustment = adjust_fuzzy_rules(r1[i]-r3[i])
            r2 = r1[i]-r3[i]
        load_adjustment.compute()  
        output = load_adjustment.output['output_adjustment'] 
        adjusted_load = pred_eval2_test[j+i, :] + output
        r1.append(abs(y_test[j+i]-pred_eval2_test[j+i, 50]))
        r3.append(abs(y_test[j+i]-(pred_eval2_test[j+i, 50] + output)))
        adjust.append(output)
        output_adjust.append(adjusted_load)
    output_adjust = np.array(output_adjust).reshape(-1, 99)
    if j == 0:
        output_adjust_all = output_adjust
    else:
        del r1[0:24]
        del r3[0:24]
        output_adjust_all = np.append(output_adjust_all, output_adjust, axis=0)
    output_adjust = []
# Access to adjusted values
adjust1 = np.array(adjust).reshape(-1, 1)
y=y_test.reshape(-1, 1)
output_adjust_all = np.array(output_adjust_all).reshape(-1, 99)

#Utilizing evaluation indicators to obtain results
valu, win50, win90 = PL_WS(y, output_adjust_all)
output_adjust_all_sort = -(np.sort(-output_adjust_all, axis=1, kind='quicksort', order=None))
valu_sort, win50_sort, win90_sort = PL_WS(y, output_adjust_all_sort)
valu_old, win50_old, win90_old = PL_WS(y, pred_eval2_test)
