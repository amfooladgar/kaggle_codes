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

#[store_item_Monthly_sales.loc[store_item_Monthly_sales['item_id']==col] for col in [list_of_items_in_test_set.item_id]]


store_item_Monthly_sales1 = dataset_train.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)['item_price'].mean()
store_item_Monthly_sales2 = dataset_train.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)['item_cnt_day'].sum()
store_item_Monthly_sales = store_item_Monthly_sales1.merge(store_item_Monthly_sales2, on=['date_block_num', 'shop_id', 'item_id'])


temp = dataset_test.copy()
temp['item_cnt_day']=0
df_months = temp.groupby(['shop_id', 'item_id'], as_index=False)['item_cnt_day'].last()
#df_months['item_cnt_day'] = 0

#Complete_train_set = pd.DataFrame(columns=['date_block_num', 'shop_id', 'item_id'])
#for i in range(number_of_months):
#    for j in list_of_all_available_stores_both_train_test:
#        for k in list_of_all_available_items_both_train_test:
#            df1 = pd.DataFrame([[i, j, k]],columns=['date_block_num', 'shop_id', 'item_id'])
#            Complete_train_set=Complete_train_set.append(df1,ignore_index = True)
##            if (store_item_Monthly_sales.loc[store_item_Monthly_sales['date_block_num']==i].loc[store_item_Monthly_sales['shop_id']==j].loc[store_item_Monthly_sales['item_id']==k].empty):
###                print('This item ', k,' is not available in this shop ', j,' at this month ', i , 'in Training set!!')
##                
##                if (store_item_Monthly_sales.loc[store_item_Monthly_sales['shop_id']==j].loc[store_item_Monthly_sales['item_id']==k].empty):
###                    print('This item ',k,' is nor  available in this shop ',j, ' in Training set!!: ')
###                    if(store_item_Monthly_sales.loc[store_item_Monthly_sales['item_id']==k].empty):
###                        print('This item ',k,'is not available in Training set at all!!')
###                        item_price =  np.nan
###                        
###                    else:
##                    item_price = store_item_Monthly_sales.loc[store_item_Monthly_sales['item_id']==k].values.mean()                    
##                else:
##                    item_price = store_item_Monthly_sales.loc[store_item_Monthly_sales['shop_id']==j].loc[store_item_Monthly_sales['item_id']==k].values.mean()
##                cnt_value = 0
##                
##            else:
##                cnt_value = store_item_Monthly_sales.loc[store_item_Monthly_sales['date_block_num']==i].loc[store_item_Monthly_sales['shop_id']==j].loc[store_item_Monthly_sales['item_id']==k].iloc[0].values[-1]
##                item_price = store_item_Monthly_sales.loc[store_item_Monthly_sales['date_block_num']==i].loc[store_item_Monthly_sales['shop_id']==j].loc[store_item_Monthly_sales['item_id']==k].iloc[0].values[-2]
##            df1 = pd.DataFrame([[i, j, k, item_price,cnt_value ]],columns=store_item_Monthly_sales.columns)
##            Complete_train_set=Complete_train_set.append(df1,ignore_index = True)
#
#Complete_train_set.to_csv(r'complete_train_set.csv')
#Complete_train_set2 = Complete_train_set.merge(store_item_Monthly_sales, on = ['date_block_num', 'shop_id', 'item_id'], how='outer')
#Complete_train_set2.sort_values(['shop_id', 'item_id'], inplace=True)
##



Complete_train_set = pd.DataFrame()
temp = dataset_test.copy()
temp.drop(['ID'], axis=1, inplace=True)

#temp['item_cnt_day']=0
#temp=temp.groupby(['shop_id', 'item_id'], as_index=False)['item_cnt_day'].last()

for i in range(number_of_months):
#    df_months = temp.groupby(['shop_id', 'item_id'], as_index=False)['item_cnt_day'].last() 
    temp['date_block_num']=i
    df_months = temp.copy()
    df_months=df_months.merge(store_item_Monthly_sales1.loc[store_item_Monthly_sales1['date_block_num']==i], on=['date_block_num','shop_id', 'item_id'], how = 'left')
    df_months=df_months.merge(store_item_Monthly_sales2.loc[store_item_Monthly_sales1['date_block_num']==i], on=['date_block_num','shop_id', 'item_id'], how = 'left')
#    df_months.sort_values(['date_block_num'], inplace=True)
#    df_months.groupby(['item_id'])['item_price'].fillna(method='ffill', inplace = True)
    Complete_train_set = Complete_train_set.append(df_months)


Complete_train_set.sort_values(['item_id'])['item_price'].fillna(method='ffill', inplace = True)
Complete_train_set.groupby(['item_id'])['item_cnt_day'].fillna(0, inplace = True)
Complete_train_set['item_price'].dropna(inplace=True)



store_item_Monthly_sales= Complete_train_set.merge(store_item_Monthly_sales1, on=['date_block_num','shop_id', 'item_id'], how = 'outer')

store_item_Monthly_sales['item_price'] = store_item_Monthly_sales.apply(
        lambda row : store_item_Monthly_sales.loc[store_item_Monthly_sales['item_id']==row['item_id']].item_price.mean() if np.isnan(row['item_price']) else row['item_price'], axis =1 
        )
tet = Complete_train_set.copy()

list_of_items_to_limit = np.sort(np.array(Complete_train_set.item_id.unique()))
A = Complete_train_set[Complete_train_set['item_price'].isnull()].groupby(['item_id']).size()
#A.reset
#list_of_items_to_limit=list_of_items_to_limit[A!=0]
#A = A[A!=0]
#A = A.astype(int)
Complete_train_set.sort_values(['item_id','item_price'], inplace = True)
[Complete_train_set['item_cnt_day'].fillna(0, inplace= True)]
[Complete_train_set['item_price'].fillna(method='ffill',limit=A[i], inplace= True) for i in list_of_items_to_limit]

def fill_nan_item_price_cnt(data):
    list_of_items_to_limit = np.sort(np.array(data.item_id.unique()))
    A = data[data['item_price'].isnull()].groupby(['item_id']).size()
    filleddata=data.sort_values(['item_id','item_price'])
    [filleddata['item_cnt_day'].fillna(0, inplace= True)]
    [filleddata['item_price'].fillna(method='ffill',limit=A[i], inplace= True) for i in list_of_items_to_limit]
    return filleddata
    
result_completeSet = fill_nan_item_price_cnt(Complete_train_set)


tmp_store_items = store_item_Monthly_sales.head(1000)
tmp_store_items.loc['date_block_num'].fillnan(0, inplace = True)

tmp_store_items['item_price'] = tmp_store_items.apply(
        lambda row : tmp_store_items.loc[tmp_store_items['item_id']==row['item_id']].item_price.mean() if np.isnan(row['item_price']) else row['item_price'], axis =1 
        )

store_item_Monthly_sales= store_item_Monthly_sales.merge(store_item_Monthly_sales2, on=['shop_id', 'item_id'], how = 'outer')
store_item_Monthly_sales.reset_index(inplace=True)
store_item_Monthly_sales.drop('index', axis=1, inplace=True)
#store_item_Monthly_sales = store_item_Monthly_sales.merge(Complete_train_set, on=['date_block_num','shop_id', 'item_id'], how = 'outer')
store_item_Monthly_sales.drop('item_cnt_day_x', axis=1, inplace=True)
#store_item_Monthly_sales.drop('item_price_y', axis=1, inplace=True)
store_item_Monthly_sales.rename(columns={"item_cnt_day_y": "item_cnt_day"},inplace=True)
#store_item_Monthly_sales.rename(columns={"item_price_x": "item_price"},inplace=True)
store_item_Monthly_sales.sort_values(['date_block_num'], inplace=True)
store_item_Monthly_sales.sort_values(['shop_id', 'item_id'], inplace=True)
#store_item_Monthly_sales.dropna(inplace=True)
#store_item_Monthly_sales=store_item_Monthly_sales[(store_item_Monthly_sales['item_price'] != np.nan)]
tmp = store_item_Monthly_sales.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)['item_price'].mean()
store_item_Monthly_sales.fillna(0,inplace=True)
store_item_Monthly_sales.sort_values(['shop_id', 'item_id'], inplace=True)

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
window = 10
lag = 1
#series = series_to_supervised(store_item_Monthly_sales.drop('date_block_num', axis=1), window=window, lag=lag)
series = series_to_supervised(store_item_Monthly_sales, window=window, lag=lag)

series.head()
#series.describe()

# Encoding categorical data



# Drop rows with different item or store values than the shifted columns

last_item = 'item_id(t-%d)' % window
last_store = 'shop_id(t-%d)' % window
series = series[(series['shop_id(t+1)'] == series[last_store])]
series = series[(series['item_id(t+1)'] == series[last_item])]

#for i in range(window, 0, -1):
#    one_hot= pd.get_dummies(series['shop_id(t-%d)' % i ])
#    series = pd.concat([series, one_hot], axis=1)
#one_hot= pd.get_dummies(series['shop_id(t)'])
#series = pd.concat([series, one_hot], axis=1)
#one_hot= pd.get_dummies(series['date_block_num(t)'])
#series = pd.concat([series, one_hot], axis=1)

#for i in range(window, 0, -1):
#    one_hot= pd.get_dummies(series['date_block_num(t-%d)' %  i])
#    series = pd.concat([series, one_hot], axis=1)

#one_hot= pd.get_dummies(series['shop_id(t+1)'])
#series = pd.concat([series, one_hot], axis=1)

# Remove unwanted columns
columns_to_drop = [('%s(t+%d)' % (col, lag)) for col in ['item_id', 'shop_id','date_block_num']]
for i in range(window, 0, -1):
    columns_to_drop += [('%s(t-%d)' % (col, i)) for col in ['item_id', 'shop_id','date_block_num']]
series.drop(columns_to_drop, axis=1, inplace=True)
predictable_shops_items_pair_Col =['item_id(t)', 'shop_id(t)']
predictable_shops_items_pair = series[predictable_shops_items_pair_Col]
series.drop(['item_id(t)', 'shop_id(t)','date_block_num(t)'], axis=1, inplace=True)


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
#sc_X_item_price = MinMaxScaler(feature_range=(0,1))
#sc_X_cnt_sales = MinMaxScaler(feature_range=(0,1))
#labels_norm_item_price = ['item_price(t-%d)' % i for i in range(window, 0, -1)]
#labels_norm_item_price.append('item_price(t)')
#labels_norm_item_price.append('item_price(t+1)')
#labels_norm_cnt_sales = ['item_cnt_day(t-%d)' % i for i in range(window, 0, -1)]
#labels_norm_cnt_sales.append('item_cnt_day(t)')
#series_item_price_scaled = sc_X_item_price.fit_transform(series[labels_norm_item_price])
#series_cnt_sales_scaled = sc_X_cnt_sales.fit_transform(series[labels_norm_cnt_sales])



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
#model_lstm.add(LSTM(units = 50, return_sequences= True))
#model_lstm.add(Dropout(0.2))

# Adding the third LSTM layer and some Dropout regularization
#model_lstm.add(LSTM(units = 50, return_sequences= True))
#model_lstm.add(Dropout(0.2))

# Adding the fourth LSTM layer and some Dropout regularization
model_lstm.add(LSTM(units = 50))
model_lstm.add(Dropout(0.2))

# Adding the output layer
model_lstm.add(Dense(units=2))

# Compiling the RNN
model_lstm.compile(optimizer= 'adam', loss = 'mean_squared_error')

# Fitting the RNN to Training set
epochs = 10
batch = 64
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

# Getting the predicted number of sales for November 2015
dataset_total = pd.concat((store_item_Monthly_sales,dataset_test),axis=0,sort=False)
dataset_total.sort_values(['shop_id', 'item_id'], inplace=True)
#df1 = dataset_total.loc[dataset_total['date_block_num']== 32]
#df2 = dataset_total.loc[dataset_total['date_block_num']== 33]
#df3 = dataset_total.loc[dataset_total['date_block_num']== 34]
#dataset_total_last_window = pd.concat((df1,df2,df3),axis=0,sort=False)

#test_series = series_to_supervised(dataset_total.drop('date_block_num', axis=1), window=window, lag=lag)
test_series = series_to_supervised(dataset_total, window=window, lag=lag)

# Drop rows with different item or store values than the shifted columns

last_item = 'item_id(t-%d)' % window
last_store = 'shop_id(t-%d)' % window
test_series = test_series[(test_series['shop_id(t+1)'] == test_series[last_store])]
test_series = test_series[(test_series['item_id(t+1)'] == test_series[last_item])]

#
#one_hot= pd.get_dummies(test_series['shop_id(t)'])
#test_series = pd.concat([test_series, one_hot], axis=1)
#
#one_hot= pd.get_dummies(test_series['date_block_num(t)'])
#test_series = pd.concat([test_series, one_hot], axis=1)
#
#for i in range(window, 0, -1):
#    one_hot= pd.get_dummies(test_series['date_block_num(t-%d)' %  i])
#    test_series = pd.concat([test_series, one_hot], axis=1)
#

# Remove unwanted columns
columns_to_drop = [('%s(t+%d)' % (col, lag)) for col in ['item_id', 'shop_id','date_block_num']]
for i in range(window, 0, -1):
    columns_to_drop += [('%s(t-%d)' % (col, i)) for col in ['item_id', 'shop_id','date_block_num']]
test_series.drop(columns_to_drop, axis=1, inplace=True)
test_series.reset_index(inplace=True)
test_series.drop('index', axis=1, inplace=True)
predictable_shops_items_pair_test_Col =['item_id(t)', 'shop_id(t)']
predictable_shops_items_pair_test = test_series[predictable_shops_items_pair_test_Col]
test_series.drop(['item_id(t)', 'shop_id(t)','date_block_num(t)'], axis=1, inplace=True)


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
final_predicted_test_set.rename(columns={"shop_id(t)": "shop_id", "item_id(t)": "item_id"},inplace=True)

final_predicted_test_removed_unwanted = final_predicted_test_set.groupby(['shop_id', 'item_id'], as_index=False)['item_cnt_day'].last()
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
            
To_submit_results.to_csv(r'Ali_submission4_monthly_window3_addedFeatureShopId.csv',index=False)


# ----------------------to save and load the trained model
import pickle
# to export the trained model
with open('model_lstm_trained_monthly_window2_featuresShpIdBlkId_1Epoch', 'wb') as model_lstm_trained_monthly_window2_featuresShpIdBlkId_1Epoch:
    pickle.dump(model_lstm, model_lstm_trained_monthly_window2_featuresShpIdBlkId_1Epoch)
    
#to Import the trained model
with open('model_lstm_trained_monthly_window2_featuresShpIdBlkId_1Epoch', 'rb') as model_lstm_trained_monthly_window2_featuresShpIdBlkId_1Epoch:
    model_lstm = pickle.load(model_lstm_trained_monthly_window2_featuresShpIdBlkId_1Epoch)
