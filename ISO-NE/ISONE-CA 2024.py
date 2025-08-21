# -*- coding: utf-8 -*-
"""

@author: XMQ
"""

################################################################################
## Keras implementation of day-ahead prediction of the ISO-NE hourly demand data.

# -----------------------------------------------------------------------------
# load original data file
# modification of the data loading procedure is needed if you have your own dataset

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from keras.layers import Reshape, Dropout, Multiply
from tensorflow.keras import layers
import h5py
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
# tf.compat.v1.enable_eager_execution() # 特殊的 屏蔽lstm加载
start=0
start=time.perf_counter()

parse_dates = ['date']
df = pd.read_csv('ISO-NE (2023-2024).csv', parse_dates=parse_dates, index_col='date')

# missing_rate = 0.01  # 1% 的数据设为缺失状态
# np.random.seed(42)  # 确保随机性的可重复性
#
# # 仅对DataFrame的最后两列处理
# columns_to_process = [-2, -1]  # 选择倒数第二列和最后一列
# for col in columns_to_process:
#     # 生成随机选择掩码
#     mask = np.random.rand(len(df)) < missing_rate
#     # 生成高斯噪声，均值为0°F，标准差为1°F
#     noise = np.random.normal(0, 1, size=mask.sum())
#     # 为选定数据点添加噪声
#     df.iloc[mask, col] += noise

D_max_daily = df.groupby('date').demand.max().values
D_min_daily = df.groupby('date').demand.min().values
D = df.demand.values
T = df.temperature.values


# duplicate max and min daily demand values for 24 hours in a day
D_max = np.zeros(len(D))
D_min = np.zeros(len(D))

for i in range(len(D)):
    n_day = int(i/24)
    D_max[i] = D_max_daily[n_day]
    D_min[i] = D_min_daily[n_day]

# normalization based on peak values
D_max = D_max / 25000.
D_min = D_min / 25000.
D = D / 25000.
T = T / 100.

# add weekday info to the dataset
# the initial value for iter_weekday corresponds to the first day of the dataset
iter_weekday = 7
weekday = np.zeros((24*731,))
for i in range(731):
    mod = np.mod(iter_weekday, 7)
    for j in range(24):
        if (mod == 6) or (mod == 0):
            weekday[i*24 + j] = 0
        else:
            weekday[i*24 + j] = 1
    iter_weekday += 1

# add season and festival info to the dataset
import datetime
iter_date = datetime.date(2023, 1, 1)
season = np.zeros((24*731,))
festival = np.zeros((24*731,))
for i in range(731):
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
    
def data_split(D, T, D_max, D_min, season, weekday, festival, num_train_days, validation_split = 0.04949):
    '''
    prepare the dataset used for training and testing of the model.
    '''
    x_1 = []
    x_21_D = []
    x_21_T = []
    x_22_D = []
    x_22_T = []
    x_23_D = []
    x_23_T = []
    x_3 = []
    x_4 = []
    x_5 = []
    x_season = []
    x_weekday = []
    x_festival = []
    y = []
    
    len_dataset = D.shape[0]
    num_sample = len_dataset-2016
    # 2016 hours (28*3 days) is needed so that we can formulate the first datapoint
    
    for i in range(2016,len_dataset):   
        # the demand values of the most recent 24 hours
        x_1.append(D[i-24:i])
        
        # multiple demand values every 24 hours within a week
        index_x_21 = [i-24, i-48, i-72, i-96, i-120, i-144, i-168]
        x_21_D.append(D[index_x_21])
        x_21_T.append(T[index_x_21])
        
        # multiple demand values every week within two months
        index_x_22 = [i-168, i-336, i-504, i-672, i-840, i-1008, i-1176, i-1344]
        x_22_D.append(D[index_x_22])
        x_22_T.append(T[index_x_22])
        
        # multiple demand values every month within several months
        index_x_23 = [i-672, i-1344, i-2016]
        x_23_D.append(D[index_x_23])
        x_23_T.append(T[index_x_23])
        
        x_3.append(T[i])
        x_4.append(D_max[i])
        x_5.append(D_min[i])
        
        y.append(D[i])
        
        # get one-hot representations of the additional information
        season_onehot = np.zeros(4)
        season_onehot[int(season[i])] = 1 
        x_season.append(season_onehot)

        weekday_onehot = np.zeros(2)
        weekday_onehot[int(weekday[i])] = 1 
        x_weekday.append(weekday_onehot)

        festival_onehot = np.zeros(2)
        festival_onehot[int(festival[i])] = 1 
        x_festival.append(festival_onehot)
        
    X_1 = np.array(x_1)
    X_21_D = np.array(x_21_D)
    X_21_T = np.array(x_21_T)
    X_22_D = np.array(x_22_D)
    X_22_T = np.array(x_22_T)
    X_23_D = np.array(x_23_D)
    X_23_T = np.array(x_23_T)
    X_3 = np.array(x_3)
    X_4 = np.array(x_4)
    X_5 = np.array(x_5)
    X_season = np.array(x_season)
    X_weekday = np.array(x_weekday)
    X_festival = np.array(x_festival)
    Y_1 = np.array(y)
    
    num_train = num_train_days * 24
    num_val = int(num_train * validation_split)
    
    X_train = []
    X_val = []
    X_test = []
    Y_train = []
    Y_val = []
    Y_test = []
    
    # we prepare 24 sets of data for the 24 sub-networks, each sub-network is aimed at forecasting the load of one hour of the next day
    for i in range(24):
        #               0                          1                         2                         3                         4                         5                         6                         7                    8                    9                    10                          11                           12                                              
        X_train.append([X_1[i:num_train - num_val + i:24, :24 - i], X_21_D[i:num_train - num_val + i:24, :],
                        X_21_T[i:num_train - num_val + i:24, :], X_22_D[i:num_train - num_val + i:24, :],
                        X_22_T[i:num_train - num_val + i:24, :], X_23_D[i:num_train - num_val + i:24, :],
                        X_23_T[i:num_train - num_val + i:24, :], X_3[i:num_train - num_val + i:24],
                        X_4[i:num_train - num_val + i:24], X_5[i:num_train - num_val + i:24],
                        X_season[i:num_train - num_val + i:24, :], X_weekday[i:num_train - num_val + i:24, :],
                        X_festival[i:num_train - num_val + i:24, :]])
        X_val.append(
            [X_1[num_train - num_val + i:num_train:24, :24 - i], X_21_D[num_train - num_val + i:num_train:24, :],
             X_21_T[num_train - num_val + i:num_train:24, :], X_22_D[num_train - num_val + i:num_train:24, :],
             X_22_T[num_train - num_val + i:num_train:24, :], X_23_D[num_train - num_val + i:num_train:24, :],
             X_23_T[num_train - num_val + i:num_train:24, :], X_3[num_train - num_val + i:num_train:24],
             X_4[num_train - num_val + i:num_train:24], X_5[num_train - num_val + i:num_train:24],
             X_season[num_train - num_val + i:num_train:24, :], X_weekday[num_train - num_val + i:num_train:24, :],
             X_festival[num_train - num_val + i:num_train:24, :]])
        X_test.append([X_1[num_train + i:num_sample:24, :24 - i], X_21_D[num_train + i:num_sample:24, :],
                       X_21_T[num_train + i:num_sample:24, :], X_22_D[num_train + i:num_sample:24, :],
                       X_22_T[num_train + i:num_sample:24, :], X_23_D[num_train + i:num_sample:24, :],
                       X_23_T[num_train + i:num_sample:24, :], X_3[num_train + i:num_sample:24],
                       X_4[num_train + i:num_sample:24], X_5[num_train + i:num_sample:24],
                       X_season[num_train + i:num_sample:24, :], X_weekday[num_train + i:num_sample:24, :],
                       X_festival[num_train + i:num_sample:24, :]])
        Y_train.append(Y_1[i:num_train-num_val+i:24])
        Y_val.append(Y_1[num_train - num_val + i:num_train:24])
        Y_test.append(Y_1[num_train + i:num_sample:24])


        # 看特征集长度的
    print(f"Expected number of validation samples: {num_val // 24}")
    print(f"Actual number of validation samples in Y_val: {len(Y_val)}")
    for index, val in enumerate(X_val):
        print(f"Validation sample {index} length: {len(val[0])}")  # 检查第一个特征集的长度

    # 到这完事


    return (X_train, X_val, X_test, Y_train, Y_val, Y_test)

# num_pre_days: the number of days we need before we can get the first sample, in this case: 3*28 days 
num_pre_days = 84
num_days = 731
num_test_days = 61
num_train_days = 586
num_data_points = num_days * 24
num_days_start = num_days - num_pre_days - num_test_days - num_train_days
start_data_point = num_days_start * 24
X_train, X_val, X_test, Y_train, Y_val, Y_test = data_split(D[start_data_point: start_data_point + num_data_points], T[start_data_point: start_data_point + num_data_points], D_max[start_data_point: start_data_point + num_data_points], D_min[start_data_point: start_data_point + num_data_points], season[start_data_point: start_data_point + num_data_points], weekday[start_data_point: start_data_point + num_data_points], festival[start_data_point: start_data_point + num_data_points], num_train_days,0.04949)

## ----------------------------------------------------------------------------
# define the model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate, Activation, add, BatchNormalization
from tensorflow.keras.layers import multiply, maximum, dot, average
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_absolute_percentage_error, hinge
from keras.regularizers import l1, l2
from keras.callbacks import ReduceLROnPlateau
# from keras.initializers import glorot_normal
# from keras.callbacks import EarlyStopping
# from keras.optimizers import SGD, adam

def get_input(hour):
    input_Dd = Input(shape=(7,), name='input'+str(hour)+'_Dd')
    input_Dw = Input(shape=(8,), name='input'+str(hour)+'_Dw')
    input_Dm = Input(shape=(3,), name='input'+str(hour)+'_Dm')
    input_Dr = Input(shape=(24-hour+1,), name='input'+str(hour)+'_Dr')
    
    input_Td = Input(shape=(7,), name='input'+str(hour)+'_Td')
    input_Tw = Input(shape=(8,), name='input'+str(hour)+'_Tw')
    input_Tm = Input(shape=(3,), name='input'+str(hour)+'_Tm')
    
    input_T = Input(shape=(1,))
    
    return (input_Dd, input_Dw, input_Dm, input_Dr, input_Td, input_Tw, input_Tm, input_T)
    
input1_Dd, input1_Dw, input1_Dm, input1_Dr, input1_Td, input1_Tw, input1_Tm, input1_T = get_input(1)
input2_Dd, input2_Dw, input2_Dm, input2_Dr, input2_Td, input2_Tw, input2_Tm, input2_T = get_input(2)
input3_Dd, input3_Dw, input3_Dm, input3_Dr, input3_Td, input3_Tw, input3_Tm, input3_T = get_input(3)
input4_Dd, input4_Dw, input4_Dm, input4_Dr, input4_Td, input4_Tw, input4_Tm, input4_T = get_input(4)
input5_Dd, input5_Dw, input5_Dm, input5_Dr, input5_Td, input5_Tw, input5_Tm, input5_T = get_input(5)
input6_Dd, input6_Dw, input6_Dm, input6_Dr, input6_Td, input6_Tw, input6_Tm, input6_T = get_input(6)
input7_Dd, input7_Dw, input7_Dm, input7_Dr, input7_Td, input7_Tw, input7_Tm, input7_T = get_input(7)
input8_Dd, input8_Dw, input8_Dm, input8_Dr, input8_Td, input8_Tw, input8_Tm, input8_T = get_input(8)
input9_Dd, input9_Dw, input9_Dm, input9_Dr, input9_Td, input9_Tw, input9_Tm, input9_T = get_input(9)
input10_Dd, input10_Dw, input10_Dm, input10_Dr, input10_Td, input10_Tw, input10_Tm, input10_T = get_input(10)
input11_Dd, input11_Dw, input11_Dm, input11_Dr, input11_Td, input11_Tw, input11_Tm, input11_T = get_input(11)
input12_Dd, input12_Dw, input12_Dm, input12_Dr, input12_Td, input12_Tw, input12_Tm, input12_T = get_input(12)
input13_Dd, input13_Dw, input13_Dm, input13_Dr, input13_Td, input13_Tw, input13_Tm, input13_T = get_input(13)
input14_Dd, input14_Dw, input14_Dm, input14_Dr, input14_Td, input14_Tw, input14_Tm, input14_T = get_input(14)
input15_Dd, input15_Dw, input15_Dm, input15_Dr, input15_Td, input15_Tw, input15_Tm, input15_T = get_input(15)
input16_Dd, input16_Dw, input16_Dm, input16_Dr, input16_Td, input16_Tw, input16_Tm, input16_T = get_input(16)
input17_Dd, input17_Dw, input17_Dm, input17_Dr, input17_Td, input17_Tw, input17_Tm, input17_T = get_input(17)
input18_Dd, input18_Dw, input18_Dm, input18_Dr, input18_Td, input18_Tw, input18_Tm, input18_T = get_input(18)
input19_Dd, input19_Dw, input19_Dm, input19_Dr, input19_Td, input19_Tw, input19_Tm, input19_T = get_input(19)
input20_Dd, input20_Dw, input20_Dm, input20_Dr, input20_Td, input20_Tw, input20_Tm, input20_T = get_input(20)
input21_Dd, input21_Dw, input21_Dm, input21_Dr, input21_Td, input21_Tw, input21_Tm, input21_T = get_input(21)
input22_Dd, input22_Dw, input22_Dm, input22_Dr, input22_Td, input22_Tw, input22_Tm, input22_T = get_input(22)
input23_Dd, input23_Dw, input23_Dm, input23_Dr, input23_Td, input23_Tw, input23_Tm, input23_T = get_input(23)
input24_Dd, input24_Dw, input24_Dm, input24_Dr, input24_Td, input24_Tw, input24_Tm, input24_T = get_input(24)

input_D_max = Input(shape=(1,), name='input_D_max')
input_D_min = Input(shape=(1,), name='input_D_min')
input_season = Input(shape=(4,), name='input_season')
input_weekday = Input(shape=(2,), name='input_weekday')
input_festival = Input(shape=(2,), name='input_festival')

from keras import backend as K
from tensorflow.keras.layers import Layer

class FuzzyLayer(Layer):

    def __init__(self, output_dim, initialiser_centers=None, initialiser_sigmas=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.output_dim = output_dim
        self.initialiser_centers = initialiser_centers
        self.initialiser_sigmas = initialiser_sigmas
        super(FuzzyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.fuzzy_degree = self.add_weight(name='fuzzy_degree',
                                            shape=(input_shape[-1], self.output_dim),
                                            initializer=self.initialiser_centers if self.initialiser_centers is not None else 'uniform',
                                            trainable=True)
        self.sigma = self.add_weight(name='sigma',
                                     shape=(input_shape[-1], self.output_dim),
                                     initializer=self.initialiser_sigmas if self.initialiser_sigmas is not None else 'ones',
                                     trainable=True)
        super(FuzzyLayer, self).build(input_shape)

    def call(self, input, **kwargs):
        x = K.repeat_elements(K.expand_dims(input, axis=-1), self.output_dim, -1)

        fuzzy_out = K.exp(-K.sum(K.square(x - self.fuzzy_degree) / (self.sigma ** 2), axis=-2, keepdims=False))

        return fuzzy_out

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.output_dim,)


def lstm_layer(input):
    expanded_input = tf.expand_dims(input, axis=1)
    Dense1 = layers.LSTM(units = 10,use_bias=True)(expanded_input)
    reshaped = Reshape((10,))(Dense1)
    return reshaped


# def lstm_layer(input):
#     Dense1 = Dense(10, activation='selu', kernel_initializer='lecun_normal')(input)
#     return Dense1


def get_basic_structure(hour, input_Dd, input_Dw, input_Dm, input_Dr, input_Td, input_Tw, input_Tm, input_T,
                        output_pre=[]):
    '''
    get the module with the basic structure.
    output_pre is used to replace the recent 24-hour inputs with the outputs of basic-structure modules of previous hours.
    '''
    num_dense = 10


    # L_month
    concat_d = concatenate([input_Dd, input_Td])
    dense_d = lstm_layer(concat_d)

    fuzzy_layer_L_month = FuzzyLayer(output_dim=14)
    fuzzy_L_month = fuzzy_layer_L_month(concat_d)


    concat_w = concatenate([input_Dw, input_Tw])
    dense_w = lstm_layer(concat_w)

    fuzzy_layer_L_week = FuzzyLayer(output_dim=16)
    fuzzy_L_week = fuzzy_layer_L_week(concat_w)

    concat_m = concatenate([input_Dm, input_Tm])
    dense_m = lstm_layer(concat_m)

    fuzzy_layer_L_day = FuzzyLayer(output_dim=6)
    fuzzy_L_day = fuzzy_layer_L_day(concat_m)


    concat_date_info = concatenate([input_season, input_weekday])
    dense_concat_date_info_1 = lstm_layer(concat_date_info)

    fuzzy_layer_S_W = FuzzyLayer(output_dim=6)
    fuzzy_S_W = fuzzy_layer_S_W(concat_date_info)

    # dense_concat_date_info_2 = lstm_layer(concat_date_info)

    # print(fuzzy_L_month.shape, fuzzy_L_week.shape, fuzzy_L_day.shape, fuzzy_S_W.shape, input_festival.shape)
    concat_FC2 = concatenate([dense_d, dense_w, dense_m, dense_concat_date_info_1, input_festival])
    concat_FC2 = Dense(10, activation='selu', kernel_initializer='lecun_normal')(concat_FC2)


    Fuzzy_FC2 = concatenate([fuzzy_L_month, fuzzy_L_week, fuzzy_L_day, fuzzy_S_W, input_festival])
    Fuzzy_FC2 = Dense(10, activation='selu', kernel_initializer='lecun_normal')(Fuzzy_FC2)
    # Fuzzy_FC2= Dense(10, activation='selu', kernel_initializer='lecun_normal')(Fuzzy_FC2)
    # Fuzzy_FC2 = Dropout(0.3)(Fuzzy_FC2)

    # dense_FC2 = Dense(10, activation='selu', kernel_initializer='lecun_normal')(concat_FC2 )
    # FC2 = Dropout(0.3)(dense_FC2)



    inputs_FC_2 = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))([concat_FC2,Fuzzy_FC2])
    output_FC_2 = tf.keras.layers.Attention()([inputs_FC_2, inputs_FC_2])
    output_FC_2 = tf.keras.layers.Flatten()(output_FC_2)
    FC2 = Dense(10, activation='selu', kernel_initializer='lecun_normal')(output_FC_2)
    FC2 = Dropout(0.3)(FC2)






    if output_pre == []:
        dense_Dr = lstm_layer(input_Dr)

        fuzzy_layer_L_hour= FuzzyLayer(output_dim=24)
        fuzzy_L_hour = fuzzy_layer_L_hour(input_Dr)

    else:
        concat_Dr = concatenate([input_Dr] + output_pre)
        dense_Dr = lstm_layer(concat_Dr)

        fuzzy_layer_L_hour = FuzzyLayer(output_dim=24)
        fuzzy_L_hour = fuzzy_layer_L_hour(concat_Dr)


    concat_FC1 = (concatenate([dense_Dr, dense_concat_date_info_1]))
    concat_FC1 = Dense(10, activation='selu', kernel_initializer='lecun_normal')(concat_FC1)

    Fuzzy_fc1 = concatenate([fuzzy_L_hour, fuzzy_S_W])
    Fuzzy_fc1 = Dense(10, activation='selu', kernel_initializer='lecun_normal')(Fuzzy_fc1)


    inputs_FC_1 = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))([concat_FC1,Fuzzy_fc1])
    output_FC_1 = tf.keras.layers.Attention()([inputs_FC_1, inputs_FC_1])
    output_FC_1 = tf.keras.layers.Flatten()(output_FC_1)
    FC1 = Dense(num_dense, activation='selu', kernel_initializer='lecun_normal')(output_FC_1)
    FC1 = Dropout(0.3)(FC1)



    Direct_connection1 = concatenate([dense_Dr, fuzzy_L_hour])
    Direct_connection = Dense(10, activation='selu', kernel_initializer='lecun_normal')(Direct_connection1)


    dense_pre_output = concatenate([FC1, FC2, Direct_connection,input_T])





    dense_pre_output = Dense(10, activation='selu', kernel_initializer='lecun_normal')(dense_pre_output)
    output = Dense(1, activation='linear', name='output' + str(hour), kernel_initializer='lecun_normal')(dense_pre_output)

    output_pre_new = output_pre + [output]
    return (output, output_pre_new)


output1, output_pre1 = get_basic_structure(1, input1_Dd, input1_Dw, input1_Dm, input1_Dr, input1_Td, input1_Tw, input1_Tm, input1_T)
output2, output_pre2 = get_basic_structure(2, input2_Dd, input2_Dw, input2_Dm, input2_Dr, input2_Td, input2_Tw, input2_Tm, input2_T, output_pre1)
output3, output_pre3 = get_basic_structure(3, input3_Dd, input3_Dw, input3_Dm, input3_Dr, input3_Td, input3_Tw, input3_Tm, input3_T, output_pre2)
output4, output_pre4 = get_basic_structure(4, input4_Dd, input4_Dw, input4_Dm, input4_Dr, input4_Td, input4_Tw, input4_Tm, input4_T, output_pre3)
output5, output_pre5 = get_basic_structure(5, input5_Dd, input5_Dw, input5_Dm, input5_Dr, input5_Td, input5_Tw, input5_Tm, input5_T, output_pre4)
output6, output_pre6 = get_basic_structure(6, input6_Dd, input6_Dw, input6_Dm, input6_Dr, input6_Td, input6_Tw, input6_Tm, input6_T, output_pre5)
output7, output_pre7 = get_basic_structure(7, input7_Dd, input7_Dw, input7_Dm, input7_Dr, input7_Td, input7_Tw, input7_Tm, input7_T, output_pre6)
output8, output_pre8 = get_basic_structure(8, input8_Dd, input8_Dw, input8_Dm, input8_Dr, input8_Td, input8_Tw, input8_Tm, input8_T, output_pre7)
output9, output_pre9 = get_basic_structure(9, input9_Dd, input9_Dw, input9_Dm, input9_Dr, input9_Td, input9_Tw, input9_Tm, input9_T, output_pre8)
output10, output_pre10 = get_basic_structure(10, input10_Dd, input10_Dw, input10_Dm, input10_Dr, input10_Td, input10_Tw, input10_Tm, input10_T, output_pre9)
output11, output_pre11 = get_basic_structure(11, input11_Dd, input11_Dw, input11_Dm, input11_Dr, input11_Td, input11_Tw, input11_Tm, input11_T, output_pre10)
output12, output_pre12 = get_basic_structure(12, input12_Dd, input12_Dw, input12_Dm, input12_Dr, input12_Td, input12_Tw, input12_Tm, input12_T, output_pre11)
output13, output_pre13 = get_basic_structure(13, input13_Dd, input13_Dw, input13_Dm, input13_Dr, input13_Td, input13_Tw, input13_Tm, input13_T, output_pre12)
output14, output_pre14 = get_basic_structure(14, input14_Dd, input14_Dw, input14_Dm, input14_Dr, input14_Td, input14_Tw, input14_Tm, input14_T, output_pre13)
output15, output_pre15 = get_basic_structure(15, input15_Dd, input15_Dw, input15_Dm, input15_Dr, input15_Td, input15_Tw, input15_Tm, input15_T, output_pre14)
output16, output_pre16 = get_basic_structure(16, input16_Dd, input16_Dw, input16_Dm, input16_Dr, input16_Td, input16_Tw, input16_Tm, input16_T, output_pre15)
output17, output_pre17 = get_basic_structure(17, input17_Dd, input17_Dw, input17_Dm, input17_Dr, input17_Td, input17_Tw, input17_Tm, input17_T, output_pre16)
output18, output_pre18 = get_basic_structure(18, input18_Dd, input18_Dw, input18_Dm, input18_Dr, input18_Td, input18_Tw, input18_Tm, input18_T, output_pre17)
output19, output_pre19 = get_basic_structure(19, input19_Dd, input19_Dw, input19_Dm, input19_Dr, input19_Td, input19_Tw, input19_Tm, input19_T, output_pre18)
output20, output_pre20 = get_basic_structure(20, input20_Dd, input20_Dw, input20_Dm, input20_Dr, input20_Td, input20_Tw, input20_Tm, input20_T, output_pre19)
output21, output_pre21 = get_basic_structure(21, input21_Dd, input21_Dw, input21_Dm, input21_Dr, input21_Td, input21_Tw, input21_Tm, input21_T, output_pre20)
output22, output_pre22 = get_basic_structure(22, input22_Dd, input22_Dw, input22_Dm, input22_Dr, input22_Td, input22_Tw, input22_Tm, input22_T, output_pre21)
output23, output_pre23 = get_basic_structure(23, input23_Dd, input23_Dw, input23_Dm, input23_Dr, input23_Td, input23_Tw, input23_Tm, input23_T, output_pre22)
output24, output_pre24 = get_basic_structure(24, input24_Dd, input24_Dw, input24_Dm, input24_Dr, input24_Td, input24_Tw, input24_Tm, input24_T, output_pre23)

def get_res_layer(output, last=False):
    '''
    obtain one basic layer in the deep residual network
    '''
    dense_res11 = Dense(20, activation='selu', kernel_initializer='lecun_normal')(output)
    dense_res12 = Dense(24, activation='linear', kernel_initializer='lecun_normal')(dense_res11)
    
    dense_res21 = Dense(20, activation='selu', kernel_initializer='lecun_normal')(output)
    dense_res22 = Dense(24, activation='linear', kernel_initializer='lecun_normal')(dense_res21)

    dense_res31 = Dense(20, activation='selu', kernel_initializer='lecun_normal')(output)
    dense_res32 = Dense(24, activation='linear', kernel_initializer='lecun_normal')(dense_res31)

    dense_res41 = Dense(20, activation='selu', kernel_initializer='lecun_normal')(output)
    dense_res42 = Dense(24, activation='linear', kernel_initializer='lecun_normal')(dense_res41)
    
    dense_add = add([dense_res12, dense_res22, dense_res32, dense_res42])
    
    if last:
        output_new = add([dense_add, output], name='output')
    else:
        output_new = add([dense_add, output])
    return output_new

output_pre = concatenate(output_pre24)

def resnetplus_layer(input_1, input_2, output_list):
    '''
    obtain one layer in ResNetPlus.
    '''
    output_res = get_res_layer(input_1)
    output_res_ = get_res_layer(input_2)
    output_res_ave_mid = average([output_res, output_res_])
    output_list.append(output_res_ave_mid)
    output_res_ave = average(output_list)
    return output_res_ave, output_list
    
input_1 = output_pre
input_2 = output_pre
output_list = [output_pre]

num_resnetplus_layer = 15

for i in range(num_resnetplus_layer):
    output_res_ave, output_list = resnetplus_layer(input_1, input_2, output_list)
    input_1 = output_res_ave
    if i == 0:
        input_2 = output_res_ave

output = output_res_ave

def penalized_loss(y_true, y_pred):
    '''
    the loss that penalizes the model when the forcast demand is output of the boundaries for the day's actual demand.
    '''
    beta = 0.5
    loss1 = mean_absolute_percentage_error(y_true, y_pred)
    loss2 = K.mean(K.maximum(K.max(y_pred, axis=1) - input_D_max, 0.), axis=-1)
    loss3 = K.mean(K.maximum(input_D_min - K.min(y_pred, axis=1), 0.), axis=-1)
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
    X_new = X_new + [X[0][8], X[0][9], X[0][10], X[0][11], X[0][12]]
    Y_new = [np.squeeze(np.array(Y_new)).transpose()] # the aggregate output of 24 single outputs
    return (X_new, Y_new)

## ----------------------------------------------------------------------------
# compile and train the model

X_train_fit, Y_train_fit = get_XY(X_train, Y_train)
X_val_fit, Y_val_fit = get_XY(X_val, Y_val)
X_test_pred, Y_test_pred = get_XY(X_test, Y_test)

def get_model():
    model = Model(inputs=[input1_Dd, input1_Dw, input1_Dm, input1_Dr, input1_Td, input1_Tw, input1_Tm, input1_T,\
                      input2_Dd, input2_Dw, input2_Dm, input2_Dr, input2_Td, input2_Tw, input2_Tm, input2_T,\
                      input3_Dd, input3_Dw, input3_Dm, input3_Dr, input3_Td, input3_Tw, input3_Tm, input3_T,\
                      input4_Dd, input4_Dw, input4_Dm, input4_Dr, input4_Td, input4_Tw, input4_Tm, input4_T,\
                      input5_Dd, input5_Dw, input5_Dm, input5_Dr, input5_Td, input5_Tw, input5_Tm, input5_T,\
                      input6_Dd, input6_Dw, input6_Dm, input6_Dr, input6_Td, input6_Tw, input6_Tm, input6_T,\
                      input7_Dd, input7_Dw, input7_Dm, input7_Dr, input7_Td, input7_Tw, input7_Tm, input7_T,\
                      input8_Dd, input8_Dw, input8_Dm, input8_Dr, input8_Td, input8_Tw, input8_Tm, input8_T,\
                      input9_Dd, input9_Dw, input9_Dm, input9_Dr, input9_Td, input9_Tw, input9_Tm, input9_T,\
                      input10_Dd, input10_Dw, input10_Dm, input10_Dr, input10_Td, input10_Tw, input10_Tm, input10_T,\
                      input11_Dd, input11_Dw, input11_Dm, input11_Dr, input11_Td, input11_Tw, input11_Tm, input11_T,\
                      input12_Dd, input12_Dw, input12_Dm, input12_Dr, input12_Td, input12_Tw, input12_Tm, input12_T,\
                      input13_Dd, input13_Dw, input13_Dm, input13_Dr, input13_Td, input13_Tw, input13_Tm, input13_T,\
                      input14_Dd, input14_Dw, input14_Dm, input14_Dr, input14_Td, input14_Tw, input14_Tm, input14_T,\
                      input15_Dd, input15_Dw, input15_Dm, input15_Dr, input15_Td, input15_Tw, input15_Tm, input15_T,\
                      input16_Dd, input16_Dw, input16_Dm, input16_Dr, input16_Td, input16_Tw, input16_Tm, input16_T,\
                      input17_Dd, input17_Dw, input17_Dm, input17_Dr, input17_Td, input17_Tw, input17_Tm, input17_T,\
                      input18_Dd, input18_Dw, input18_Dm, input18_Dr, input18_Td, input18_Tw, input18_Tm, input18_T,\
                      input19_Dd, input19_Dw, input19_Dm, input19_Dr, input19_Td, input19_Tw, input19_Tm, input19_T,\
                      input20_Dd, input20_Dw, input20_Dm, input20_Dr, input20_Td, input20_Tw, input20_Tm, input20_T,\
                      input21_Dd, input21_Dw, input21_Dm, input21_Dr, input21_Td, input21_Tw, input21_Tm, input21_T,\
                      input22_Dd, input22_Dw, input22_Dm, input22_Dr, input22_Td, input22_Tw, input22_Tm, input22_T,\
                      input23_Dd, input23_Dw, input23_Dm, input23_Dr, input23_Td, input23_Tw, input23_Tm, input23_T,\
                      input24_Dd, input24_Dw, input24_Dm, input24_Dr, input24_Td, input24_Tw, input24_Tm, input24_T,\
                      input_D_max, input_D_min, input_season, input_weekday, input_festival], \
                      outputs=[output])
    return model
      
model = get_model()
model.compile(optimizer='adam', loss=penalized_loss)
# model.save_weights('model.h5')

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
    lr = 0.0006
    return lr


def lr_scheduler3(epoch, mode=None):
    lr = 0.0004
    return lr


scheduler1 = LearningRateScheduler(lr_scheduler1)
scheduler2 = LearningRateScheduler(lr_scheduler2)
scheduler3 = LearningRateScheduler(lr_scheduler3)

num_repeat = 5
NUM_TEST = 61
BATCH_SIZE = 32
NUM_SNAPSHOT = 10

for i in range(5):
    # model.load_weights('model.h5')
    shuffle_weights(model)

    history_1 = model.fit(X_train_fit, Y_train_fit, validation_data=(X_val_fit, Y_val_fit),\
                          epochs=400, batch_size=BATCH_SIZE, callbacks=[scheduler1])

    model.save_weights('complete' + str(i + 1) + '1_weights.h5')
    print(str(i) + ' 1')

    history_2 = model.fit(X_train_fit, Y_train_fit, validation_data=(X_val_fit, Y_val_fit),\
                          epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler1])

    model.save_weights('complete' + str(i + 1) + '2_weights.h5')
    print(str(i) + ' 2')

    history_3 = model.fit(X_train_fit, Y_train_fit, validation_data=(X_val_fit, Y_val_fit),\
                          epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler1])

    model.save_weights('complete' + str(i + 1) + '3_weights.h5')
    print(str(i) + ' 3')

    history_4 = model.fit(X_train_fit, Y_train_fit, validation_data=(X_val_fit, Y_val_fit),\
                          epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler1])

    model.save_weights('complete' + str(i + 1) + '4_weights.h5')
    print(str(i) + ' 4')

    history_5 = model.fit(X_train_fit, Y_train_fit, validation_data=(X_val_fit, Y_val_fit),\
                          epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler1])

    model.save_weights('complete' + str(i + 1) + '5_weights.h5')
    print(str(i) + ' 5')

    history_6 = model.fit(X_train_fit, Y_train_fit, validation_data=(X_val_fit, Y_val_fit),\
                          epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler2])

    model.save_weights('complete' + str(i + 1) + '6_weights.h5')
    print(str(i) + ' 6')

    history_7 = model.fit(X_train_fit, Y_train_fit, validation_data=(X_val_fit, Y_val_fit),\
                          epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler2])

    model.save_weights('complete' + str(i + 1) + '7_weights.h5')
    print(str(i) + ' 7')

    history_8 = model.fit(X_train_fit, Y_train_fit, validation_data=(X_val_fit, Y_val_fit),\
                          epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler2])

    model.save_weights('complete' + str(i + 1) + '8_weights.h5')
    print(str(i) + ' 8')

    history_9 = model.fit(X_train_fit, Y_train_fit, validation_data=(X_val_fit, Y_val_fit),\
                          epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler3])

    model.save_weights('complete' + str(i + 1) + '9_weights.h5')
    print(str(i) + ' 9')

    history_10 = model.fit(X_train_fit, Y_train_fit, validation_data=(X_val_fit, Y_val_fit),\
                           epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler3])

    model.save_weights('complete' + str(i + 1) + '10_weights.h5')
    print(str(i) + ' 10')

    history_list.append([history_1, history_2, history_3, history_4, history_5, history_6, history_7,
                         history_8, history_9, history_10])
#

    
loss = np.zeros((NUM_SNAPSHOT, NUM_SNAPSHOT))
for i in tqdm(range(0, NUM_SNAPSHOT)):
    for j in range(i, NUM_SNAPSHOT):
        p = np.zeros((num_repeat * (j - i + 1), 24 * NUM_TEST))
        for k in range(num_repeat):
            for l in range(i, j + 1):
                model.load_weights('complete' + str(k + 1) + str(l + 1) + '_weights.h5')
                pred = model.predict(X_test_pred)
                p[k * (j - i + 1) + l - i, :] = pred.reshape(24 * NUM_TEST)
        pred_eval = np.mean(p, axis=0)
        Y_test_eval = np.array(Y_test).transpose().reshape(24 * NUM_TEST)
        mape = np.mean(np.divide(np.abs(Y_test_eval - pred_eval), Y_test_eval))
        print(mape)

        # 加的
        mae = np.mean(np.abs(Y_test_eval - pred_eval))
        print("MAE:", mae)

        # Calculate RMSE
        rmse = np.sqrt(np.mean((Y_test_eval - pred_eval) ** 2))
        print("RMSE:", rmse)

        # Calculate R-squared6
        ss_total = np.sum((Y_test_eval - np.mean(Y_test_eval)) ** 2)
        ss_residual = np.sum((Y_test_eval - pred_eval) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        print("R-squared:", r_squared)


        loss[i, j] = mape

        mape_df = pd.DataFrame(loss, columns=[f'Snapshot{j + 1}' for j in range(NUM_SNAPSHOT)],
                               index=[f'Snapshot{i + 1}' for i in range(NUM_SNAPSHOT)])

        # 将DataFrame保存为Excel文件
        mape_df.to_excel('mape_values222.xlsx')

        # print("MAPE values success saved to 'mape_values.xlsx'")

        # 将数据框保存到Excel文件中


end = time.perf_counter()
dur = end - start


df1 = pd.DataFrame([dur], columns=['TIME'])

# 将数据框保存到Excel文件中
df1.to_excel('all_time.xlsx', index=False)


print('\n','\n',"="*20,"程序跑完时长:",dur)