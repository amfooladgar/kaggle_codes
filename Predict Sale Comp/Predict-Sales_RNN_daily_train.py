# Part1: Data Preprocessing

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

import datetime
#%matplotlib inline
#from tensorflow import set_random_seed
from numpy.random import seed


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
list_of_stores_in_train_set = np.sort(np.array(dataset_train.shop_id.unique()))
list_of_items_in_train_set = np.sort(np.array(dataset_train.item_id.unique()))

list_of_all_available_items_both_train_test=[x for x in list_of_items_in_test_set if x in list_of_items_in_train_set]
list_of_all_available_stores_both_train_test=[x for x in list_of_stores_in_test_set if x in list_of_stores_in_train_set]

number_of_months =  np.size(dataset_train.date_block_num.unique())

temp=dataset_train
dataset_train.sort_values(by='date',inplace=True)
#train_gp=dataset_train.groupby(['shop_id','item_id', 'date'],as_index=False)
#train_gp = train_gp.agg({'item_cnt_day':['mean']})

dataset_train.sort_values(['shop_id', 'item_id'], inplace=True)


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



# We will use the current timestep and the last window-size months to forecast next month ahead
window = 29
lag = 1
#series = series_to_supervised(store_item_Monthly_sales.drop('date_block_num', axis=1), window=window, lag=lag)
series = series_to_supervised(dataset_train.drop(['date_block_num','date'], axis=1), window=window, lag=lag)

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
predictable_shops_items_pair_Col =['item_id(t)', 'shop_id(t)']
predictable_shops_items_pair = series[predictable_shops_items_pair_Col]
series.drop(['item_id(t)', 'shop_id(t)'], axis=1, inplace=True)

series.describe()



# Train/validation split
# Label
labels_col = ['item_price(t+1)','item_cnt_day(t+1)']
labels = series[labels_col]
series = series.drop(labels_col, axis=1)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
series_scaled = sc_X.fit_transform(series)
labels_scaled = sc_y.fit_transform(labels)



predictable_shops_items_pair_train,predictable_shops_items_pair_valid = train_test_split(predictable_shops_items_pair, test_size=0.2, random_state=0)

X_train, X_valid, Y_train, Y_valid = train_test_split(series_scaled, labels_scaled, test_size=0.2, random_state=0)
print('Train set shape', X_train.shape)
print('Validation set shape', X_valid.shape)
#X_train.head()

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
from sklearn.metrics import mean_squared_error

# Initializing the RNN
model_lstm = Sequential()

# Adding the first LSTM layer and some Dropout regularization
model_lstm.add(LSTM(units = 50, return_sequences= True, input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
model_lstm.add(Dropout(0.2))

# Adding the second LSTM layer and some Dropout regularization
model_lstm.add(LSTM(units = 50, return_sequences= True))
model_lstm.add(Dropout(0.2))

# Adding the third LSTM layer and some Dropout regularization
model_lstm.add(LSTM(units = 50, return_sequences= True))
model_lstm.add(Dropout(0.2))

# Adding the fourth LSTM layer and some Dropout regularization
model_lstm.add(LSTM(units = 50))
model_lstm.add(Dropout(0.2))

# Adding the output layer
model_lstm.add(Dense(units=2))

# Compiling the RNN
model_lstm.compile(optimizer= 'adam', loss = 'mean_squared_error')

# Fitting the RNN to Training set
epochs = 20
batch = 256
model_lstm.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid), epochs = epochs, batch_size=batch)


# Part 3: Making the predictions and visualising the results 

#LSTM on train and validation
lstm_train_pred = model_lstm.predict(X_train_series)
lstm_valid_pred = model_lstm.predict(X_valid_series)

Y_train_pred = sc_y.inverse_transform(lstm_train_pred)
Y_valid_pred = sc_y.inverse_transform(lstm_valid_pred)
Y_train_inversed = sc_y.inverse_transform(Y_train)
Y_valid_inversed = sc_y.inverse_transform(Y_valid)

#lstm_train_pred = sc.inverse_transform(lstm_train_pred)
#lstm_valid_pred = sc.inverse_transform(lstm_valid_pred)

print('Train rmse:', np.sqrt(mean_squared_error(Y_train, lstm_train_pred)))
print('Validation rmse:', np.sqrt(mean_squared_error(Y_valid, lstm_valid_pred)))





#--------------------------------- RUN on test set by extending the set to include the test sets as well
dataset_test = pd.read_csv('test.csv')
dataset_test['date_block_num'] = 34
dataset_test['item_price'] = 0
dataset_test['item_cnt_day'] = 0
dataset_test.drop(['ID'], axis=1, inplace=True)
dataset_test2=dataset_test

from sklearn.preprocessing import StandardScaler


for i in range(2):
    dataset_test2 = pd.concat([dataset_test2,dataset_test],axis=0,sort=False,ignore_index=True)

    number_of_shop_item_entries=len(dataset_test2.groupby(['shop_id','item_id'],as_index=False))
    dataset_train.drop(['date'], axis=1, inplace=True)
    dataset_total = pd.concat([dataset_train,dataset_test2],axis=0,sort=False,ignore_index=True)
    
    rows_to_be_dropped = (len(dataset_total)-len(dataset_test2)-1000000)
    #dataset_total.drop(['ID'], axis=1, inplace=True)
    dataset_total.drop(dataset_total.index[:rows_to_be_dropped],axis=0, inplace=True)
    #inputs = sc_X.fit_transform(inputs)
    
    #X_test = np.array(X_test)    
    #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    dataset_total.sort_values(['shop_id', 'item_id'], inplace=True)
    test_series = series_to_supervised(dataset_total, window=window, lag=lag)
    
    # Drop rows with different item or store values than the shifted columns
    
    last_item = 'item_id(t-%d)' % window
    last_store = 'shop_id(t-%d)' % window
    test_series = test_series[(test_series['shop_id(t+1)'] == test_series[last_store])]
    test_series = test_series[(test_series['item_id(t+1)'] == test_series[last_item])]
    
    
    # Remove unwanted columns
    columns_to_drop = [('%s(t+%d)' % (col, lag)) for col in ['item_id', 'shop_id']]
    for i in range(window, 0, -1):
        columns_to_drop += [('%s(t-%d)' % (col, i)) for col in ['item_id', 'shop_id','date_block_num']]
    test_series.drop(columns_to_drop, axis=1, inplace=True)
    test_series.reset_index(inplace=True)
    test_series.drop('index', axis=1, inplace=True)
    predictable_shops_items_pair_test_Col =['item_id(t)', 'shop_id(t)','date_block_num(t+1)']
    predictable_shops_items_pair_test = test_series[predictable_shops_items_pair_test_Col]
    test_series.drop(['item_id(t)', 'shop_id(t)','date_block_num(t+1)','date_block_num(t)'], axis=1, inplace=True)
    test_series.describe()
    
    # Label
    labels_col = ['item_price(t+1)','item_cnt_day(t+1)']
    labels_test = test_series[labels_col]
    test_series = test_series.drop(labels_col, axis=1)
    

    sc_X_test = StandardScaler()
    sc_y_test = StandardScaler()
    test_series_scaled = sc_X_test.fit_transform(test_series)
    labels_test_scaled = sc_y_test.fit_transform(labels_test)
    X_test = np.array(test_series_scaled)    
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    
    predicted_test_set_sales = model_lstm.predict(X_test)
    predicted_test_set_sales_original_values = sc_y_test.inverse_transform(predicted_test_set_sales)
    labels_test_inversed = sc_y_test.inverse_transform(labels_test_scaled)
    predictable_shops_items_pair_test.reset_index(inplace=True)
    predictable_shops_items_pair_test.drop('index', axis=1, inplace=True)
    test_df = pd.DataFrame(predicted_test_set_sales_original_values,columns = ['item_price','item_cnt_day'])
    final_predicted_test_set = predictable_shops_items_pair_test.join(test_df)
    final_predicted_test_set.rename(columns={"shop_id(t)": "shop_id", "item_id(t)": "item_id","date_block_num(t+1)":"date_block_num"},inplace=True)
    
    final_predicted_test_removed_unwanted = final_predicted_test_set[final_predicted_test_set['date_block_num']==34]
    final_predicted_test_removed_unwanted = final_predicted_test_set.groupby(['shop_id', 'item_id','date_block_num'], as_index=False)['item_cnt_day'].sum()
    dataset_test2 = pd.concat([dataset_total,final_predicted_test_removed_unwanted],axis=0,sort=False,ignore_index=True)



dataset_test = pd.read_csv('test.csv')
To_submit_results = dataset_test
To_submit_results['item_cnt_day'] =np.nan
To_submit_results = pd.merge(To_submit_results,final_predicted_test_removed_unwanted, on=['shop_id', 'item_id'], how='outer')
To_submit_results.drop('item_cnt_day_x', axis=1, inplace=True)
To_submit_results.rename(columns={"item_cnt_day_y": "item_cnt_month"},inplace=True)

Ali_submission = To_submit_results
Ali_submission.drop(['item_id', 'shop_id', ], axis=1, inplace=True)
Ali_submission.fillna(0,inplace=True)
Ali_submission.drop(Ali_submission.index[214200:],axis=0, inplace=True)

#for j in list_of_stores_in_test_set:
#    for k in list_of_items_in_test_set:
#        if (not final_predicted_test_set.loc[final_predicted_test_set['shop_id(t)']==j].loc[final_predicted_test_set['item_id(t)']==k].empty):
#            final_predicted_test_set.loc[final_predicted_test_set['shop_id(t)']==j].loc[final_predicted_test_set['item_id(t)']==k].item_cnt_day.values[-1]
#        else:
            
To_submit_results.to_csv(r'Ali_submission_daily_1.csv',index=False)


# ----------------------to save and load the trained model
import pickle
# to export the trained model
with open('model_lstm_trained_daily', 'wb') as model_lstm_trained_daily:
    pickle.dump(model_lstm, model_lstm_trained_daily)
    
#to Import the trained model
with open('model_lstm_trained_daily', 'rb') as model_lstm_trained_daily:
    model_lstm = pickle.load(model_lstm_trained_daily)
