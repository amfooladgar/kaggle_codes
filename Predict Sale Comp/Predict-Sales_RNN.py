# Part1: Data Preprocessing

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import datetime
%matplotlib inline
from tensorflow import set_random_seed
from numpy.random import seed

from plotly.offline import plot
import plotly.graph_objs as go

#fig = go.Figure(data=[go.Bar(y=[1, 3, 2])])
#plot(fig, auto_open=True)

# Importing the triaing set
dataset_train = pd.read_csv('sales_train.csv')
dataset_test = pd.read_csv('test.csv')

dataset_train['date'] = pd.to_datetime(dataset_train.date, format='%d.%m.%Y')
dataset_train.describe()
dataset_test.describe()
list_of_stores_in_test_set = np.sort(np.array(dataset_test.shop_id.unique()))
list_of_items_in_test_set = np.sort(np.array(dataset_test.item_id.unique()))

temp=dataset_train
dataset_train.sort_values(by='date',inplace=True)
train_gp=dataset_train.groupby(['shop_id','item_id', 'date'],as_index=False)
train_gp = train_gp.agg({'item_cnt_day':['mean']})

date_start = dataset_train['date'].min().date()
date_end = dataset_train['date'].max().date()
numdays = date_end - date_start
train_date_interval = [date_start + datetime.timedelta(days=x) for x in range(numdays.days)]

num_stores = dataset_train.max().shop_id
num_items = dataset_train.max().item_id
num_train_days = numdays.days
Complete_train_set = pd.DataFrame()


l=0
index = 0
for i in range(1033):
    for j in np.sort(np.array(dataset_test.shop_id.unique())):
        for k in np.sort(np.array(dataset_test.item_id.unique())):
            if (train_gp.loc[train_gp['date']==date1].loc[train_gp['shop_id']==j].loc[train_gp['item_id']==k].empty):
                cnt_value = 0
            else:
                cnt_value = train_gp.loc[train_gp['date']==date1].loc[train_gp['shop_id']==j].loc[train_gp['item_id']==k].iloc[0].values[-1]
            df1 = pd.DataFrame([[j, k, train_date_interval[i], cnt_value ]],columns=train_gp.columns)
            Complete_train_set=Complete_train_set.append(df1,ignore_index = True)


Complete_train_set.columns = ['shop_id','item_id',  'date', 'item_cnt_day']
Complete_train_set.head()

###  dataset_train.sort_values(by='date',inplace=True)

# date_block_num - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
# shop_id - unique identifier of a shop --> There are 59 shops
# item_id - unique identifier of a product --> There are 22169 items
###   training_set = dataset_train.iloc[:,2:7].values


# Time period of the train dataset: Daily historical data from January 2013 to October 2015.
print('Min date from train set: %s' % dataset_train['date'].min().date())
print('Max date from train set: %s' % dataset_train['date'].max().date())

# ------------------------------- Plot Data --------------------------------------------
# Basic EDA
# To explore the time series data first we need to aggregate the sales by day

daily_sales = dataset_train.groupby('date', as_index=False)['item_cnt_day'].sum()
store_daily_sales = dataset_train.groupby(['shop_id', 'date'], as_index=False)['item_cnt_day'].sum()
item_daily_sales = dataset_train.groupby(['item_id', 'date'], as_index=False)['item_cnt_day'].sum()

#Overall daily sales
daily_sales_sc = go.Scatter(x=daily_sales['date'], y=daily_sales['item_cnt_day'])
layout = go.Layout(title='Daily sales', xaxis=dict(title='Date'), yaxis=dict(title='item_cnt_day'))
fig = go.Figure(data=[daily_sales_sc], layout=layout)
plot(fig,auto_open=True)

#Daily sales by store
store_daily_sales_sc = []
for store in store_daily_sales['shop_id'].unique():
    current_store_daily_sales = store_daily_sales[(store_daily_sales['shop_id'] == store)]
    store_daily_sales_sc.append(go.Scatter(x=current_store_daily_sales['date'], y=current_store_daily_sales['item_cnt_day'], name=('Store %s' % store)))

layout = go.Layout(title='Store daily sales', xaxis=dict(title='Date'), yaxis=dict(title='item_cnt_day'))
fig = go.Figure(data=store_daily_sales_sc, layout=layout)
plot(fig,auto_open=True)

# Daily sales by item
item_daily_sales_sc = []
for item in item_daily_sales['item_id'].unique():
    current_item_daily_sales = item_daily_sales[(item_daily_sales['item_id'] == item)]
    item_daily_sales_sc.append(go.Scatter(x=current_item_daily_sales['date'], y=current_item_daily_sales['item_cnt_day'], name=('Item %s' % item)))

layout = go.Layout(title='Item daily sales', xaxis=dict(title='Date'), yaxis=dict(title='Sales'))
fig = go.Figure(data=item_daily_sales_sc, layout=layout)
plot(fig,auto_open=True)

#-----------------------------------------End of plotting Data -------------------------------

# -----------------------------------Pre-processing ------------------------------



def series_to_supervised(data, window=1, lag=1, dropnan=True):
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    # Current timestep (t=0)
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    # Target timestep (t=lag)
    cols.append(data.shift(-lag))
    names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# We will use the current timestep and the last 29 to forecast next day ahead
window = 29
lag = 1
series = series_to_supervised(train_gp.drop('date', axis=1), window=window, lag=lag)
series.head()

# Drop rows with different item or store values than the shifted columns

last_item = 'item_id(t-%d)' % window
last_store = 'shop_id(t-%d)' % window
series = series[(series['shop_id(t+1)'] == series[last_store])]
series = series[(series['item_id(t+1)'] == series[last_item])]

# Remove unwanted columns
columns_to_drop = [('%s(t+%d)' % (col, lag)) for col in ['item_id', 'shop_id']]
for i in range(window, 0, -1):
    columns_to_drop += [('%s(t-%d)' % (col, i)) for col in ['item_id', 'shop_id']]
series.drop(columns_to_drop, axis=1, inplace=True)
series.drop(['item_id(t)', 'shop_id(t)'], axis=1, inplace=True)
series.describe()


# Feature Scaling
# Categorical OneHotEncoding for categorical features
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[0,1])
training_set = onehotencoder.fit_transform(training_set).toarray()
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)


# Train/validation split
# Label
labels_col = 'item_cnt_day(t+%d)' % lag
labels = series[labels_col]
series = series.drop(labels_col, axis=1)

X_train, X_valid, Y_train, Y_valid = train_test_split(series, labels.values, test_size=0.4, random_state=0)
print('Train set shape', X_train.shape)
print('Validation set shape', X_valid.shape)
X_train.head()

X_train, X_valid, Y_train, Y_valid = np.array(X_train),np.array(X_valid),np.array(Y_train),np.array(Y_valid)

# Reshaping 
X_train_series  = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_valid_series  = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
print('Train set shape', X_train_series.shape)
print('Validation set shape', X_valid_series.shape)


# Part2: Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN
model_lstm = Sequential()

# Adding the first LSTM layer and some Dropout regularization
model_lstm.add(LSTM(units = 50, return_sequences= True, input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
model_lstm.add(Dropout(0.2))

# Adding the second LSTM layer and some Dropout regularization
model_lstm.add(LSTM(units = 50, return_sequences= True))
model_lstm.add(Dropout(0.2))

## Adding the third LSTM layer and some Dropout regularization
#model_lstm.add(LSTM(units = 50, return_sequences= True))
#model_lstm.add(Dropout(0.2))

# Adding the fourth LSTM layer and some Dropout regularization
model_lstm.add(LSTM(units = 50))
model_lstm.add(Dropout(0.2))

# Adding the output layer
model_lstm.add(Dense(units=1))

# Compiling the RNN
model_lstm.compile(optimizer= 'adam', loss = 'mean_squared_error')

# Fitting the RNN to Training set
epochs = 40
batch = 256
model_lstm.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid), epochs = epochs, batch_size=batch)


# Part 3: Making the predictions and visualising the results 

#LSTM on train and validation
lstm_train_pred = model_lstm.predict(X_train_series)
lstm_valid_pred = model_lstm.predict(X_valid_series)
print('Train rmse:', np.sqrt(mean_squared_error(Y_train, lstm_train_pred)))
print('Validation rmse:', np.sqrt(mean_squared_error(Y_valid, lstm_valid_pred)))

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values


