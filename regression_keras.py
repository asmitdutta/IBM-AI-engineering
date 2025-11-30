import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import keras

import warnings
warnings.simplefilter('ignore', FutureWarning)

filepath='concrete_data.csv'
concrete_data = pd.read_csv(filepath)

concrete_data.head()
concrete_data.describe()
concrete_data.isnull().sum()
concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

predictors.head()
target.head()

predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

n_cols = predictors_norm.shape[1] # number of predictors

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# build the model
model = regression_model()

# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)