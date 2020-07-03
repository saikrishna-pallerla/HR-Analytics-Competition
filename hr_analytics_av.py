# -*- coding: utf-8 -*-
"""HR Analytics_AV.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mRnciayCWoNc5T8YqgHW4oLQzsL7SCny
"""

'''
Analytics Vidhya - HR Analytics
'''

import pandas as pd
import  numpy as np
import os
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb

from google.colab import drive
drive.mount('/content/gdrive')

# Commented out IPython magic to ensure Python compatibility.
##Change directory
# %cd /content/gdrive/My Drive/AnalyticsVidhya/HR_Analytics

ls

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train.head()

train.describe()

train.isnull().mean()

cols=list(set(train.columns)-set(['is_promoted','employee_id']))
print([(col,train[col].nunique()) for col in cols],len(train))
y='is_promoted'
numeric_cols=['age','avg_training_score','length_of_service']
cat_cols=list(set(cols)-set(numeric_cols))

x=[{col:train[col].nunique()} for col in cols]

ax = train["age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train["age"].plot(kind='density', color='teal')
ax.set(xlabel='age')
plt.xlim(20,60)
plt.show()

def plot_distribution(train,feature):
  a1=train.groupby([feature],as_index=False).agg({'is_promoted':'count'})['is_promoted'].tolist()  
  arr1=np.array(train.loc[train['is_promoted']==1].groupby([feature],as_index=False).agg({'is_promoted':'count'})['is_promoted'])
  arr2=np.array(train.loc[train['is_promoted']==0].groupby([feature],as_index=False).agg({'is_promoted':'count'})['is_promoted'])
  list1=list(np.round(np.array(arr1)/a1,2))
  list2=list(np.round(np.array(arr2)/a1,2))
  feature_list=train[feature].unique().tolist()
  p1 = plt.bar(feature_list, list1, width=0.3)
  p2 = plt.bar(feature_list, list2,width=0.3,bottom=list1)

  plt.ylabel('is_promoted')
  plt.title('# promotions by {}'.format(feature))
  plt.xticks(feature_list)
  plt.legend((p1[0], p2[0]), ('no', 'yes'))
  return plt

def density_plot(train,feature):
  ax = sns.kdeplot(train[feature][train.is_promoted == 1], color="darkturquoise", shade=True)
  sns.kdeplot(train[feature][train.is_promoted == 0], color="lightcoral", shade=True)
  plt.legend(['Yes', 'No'])
  plt.title('Density Plot of {} for Promoted and Non Promoted'.format(feature))
  ax.set(xlabel=feature)
  plt.xlim(train[feature].min(),train[feature].max())
  plt.show()

fig = plt.figure(figsize=(12, 6))
i=1
for col in ['KPIs_met >80%','gender','recruitment_channel','awards_won?']:
  plt.subplot(2,2,i)
  i+=1
  plot_distribution(train,col)

i=1
for col in numeric_cols:
  i+=1
  density_plot(train,col)

train['education'].fillna('Unknown',inplace=True)
train['previous_year_rating'].fillna(3,inplace=True)

test['education'].fillna('Unknown',inplace=True)
test['previous_year_rating'].fillna(3,inplace=True)

train['education'].value_counts()

ohe_dict={}
train_ohe=train.copy()
for col in cat_cols:
  ohe=OneHotEncoder(handle_unknown='ignore',drop=None).fit(X=train[[col]])
  ohe_dict[col]=ohe
  ohe_df=pd.DataFrame(ohe.transform(train[[col]]).toarray())
  ohe_df.columns=[col+"_"+str(i) for i in ohe_df.columns]
  train_ohe=train_ohe.join(ohe_df)
  train_ohe.drop(columns=col,inplace=True)
  
test_ohe=test.copy()
for col in cat_cols:
  ohe=ohe_dict[col]
  ohe_df=pd.DataFrame(ohe.transform(test[[col]]).toarray())
  ohe_df.columns=[col+"_"+str(i) for i in ohe_df.columns]
  test_ohe=test_ohe.join(ohe_df)
  test_ohe.drop(columns=col,inplace=True)

scaler_dict={}
scaler_train=train_ohe.copy()
for col in numeric_cols:
  scaler = StandardScaler()
  scaler.fit(train[[col]])
  scaler_dict[col]=scaler
  scaler_train[col+'_sc']=scaler.transform(train_ohe[[col]])
  scaler_train.drop(columns=col,inplace=True)

scaler_test=test_ohe.copy()
for col in numeric_cols:
  scaler=scaler_dict[col]
  scaler_test[col+'_sc']=scaler.transform(test_ohe[[col]])
  scaler_test.drop(columns=col,inplace=True)

cols_x=list(set(scaler_train.columns)-set(['employee_id','is_promoted']))
cols_y=['is_promoted']
X,y=scaler_train[cols_x],scaler_train[cols_y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

lgb_clf = lgb.LGBMClassifier()
param_grid = {
    'n_estimators': [400, 700],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [20,25],
    # 'num_leaves': [50, 100, 200],
    'reg_alpha': [1.1,  1.3],
    'reg_lambda': [1.1,1.3],
    'min_split_gain': [0.3, 0.4],
    'subsample': [0.7, 0.8],
    'subsample_freq': [20]
}

model = GridSearchCV(lgb_clf, param_grid, cv=5, scoring='f1')

model.fit(X_train,y_train)

model.best_estimator_.score(X_train,y_train)

model.best_estimator_.score(X_test,y_test)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
threshold = list(np.arange(0.1,1,0.02))
f1_list = []
for t in threshold:
    y_pred1 = [1 if i > t else 0 for i in y_prob]
    f1_list.append(f1_score(y_test,y_pred1))
plt.plot(threshold,f1_list)

f1_score(y_test,y_pred)

cutoff=threshold[np.argmax(f1_list)]
print(cutoff)
y_final_pred=np.where(y_prob>=cutoff,1,0)
f1_score(y_test,y_final_pred)

test_prob = model.best_estimator_.predict_proba(scaler_test[cols_x])[:,1]
test_pred=np.where(test_prob>=cutoff,1,0)
test_sub = pd.DataFrame({"employee_id":test["employee_id"],
                        "is_promoted":test_pred})
test_sub.to_csv("lgbm_predictions.csv",index=False)