import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline 
from pprint import pprint

#---------------------------Data------------------------------------------------------------------
train_data = '../housing/all/train.csv' #save train path 
train = pd.read_csv(train_data)

test_data = '../housing/all/test.csv' #save test path 
test = pd.read_csv(test_data)

#-----------------1 make sure cols of test and train are =-----------------------------------------
test_id = test.Id

train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

y = train.SalePrice
train.drop(['SalePrice'], axis=1, inplace=True)

#-------------------------------2 Join Test and Train---------------------------------------------
merged = pd.concat([train,test], axis=0, sort=False)

#-------------------------------3 assign NaN values-----------------------------------------------
values = {}
values_list = []
not_val = []

for c, value in enumerate(merged.dtypes):
    if value == np.int64:
        values[merged.columns[c]] = 0
        not_val.append(merged.columns[c])
    elif value == np.object:
        values[merged.columns[c]] = ""
        values_list.append(merged.columns[c])
merged.fillna(value = values, inplace=True)

#-------------------------------4 One Hot Encoding------------------------------------------------
one_hot_df = pd.get_dummies(merged, columns= values_list)
one_hot_df.fillna(0, inplace=True)

#----------------------5 Merge enocdings without original encoding columns------------------------ 
result = pd.concat([merged[not_val], one_hot_df], axis=1, sort=False)

#---------------6 Split at the row index that the train dataframe is the same width---------------
train_df = result.iloc[:len(y), :]
test_df = result.iloc[len(y):, :]

#------------------------------------7 Model------------------------------------------------------
X = train_df
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = .4, random_state = 75) 
#change test_size and rnd state***

#unscaled pipeline for reference 
unscaled = make_pipeline(PCA(n_components= 2), GaussianNB())
unscaled.fit(train_X, train_y)
pred_test = unscaled.predict(val_X)
print('unscaled value:' , mean_absolute_error(val_y ,pred_test))

#scaled pipeline
scaled = make_pipeline(StandardScaler(), PCA(n_components= 2), GaussianNB())
scaled.fit(train_X, train_y)
pred_test_scaled = scaled.predict(val_X)
print('Scaled MAE: ', mean_absolute_error(val_y, pred_test_scaled))

state_options = [1,5,10,15,20,30,35,40,45,50,55,65,75,80,100]
for random_state_option in state_options:
    forest_model = RandomForestRegressor( n_estimators = 100, n_jobs = -1, random_state= random_state_option, max_features = 100)
#pprint(forest_model.get_params())

forest_model.fit(train_X, train_y) 
predictions = forest_model.predict(val_X)
actual_preds = forest_model.predict(test_df)
print('Random_Forest_without_scaling: ', mean_absolute_error(val_y, predictions))
#print(actual_preds)

#------------------------------------submission----------------------------------------------------
sub_dict = {'SalePrice': actual_preds, 'Id': test_id}

sub_df = pd.DataFrame(sub_dict)
sub_df[["Id", "SalePrice"]].to_csv('submission.csv', index=False)
