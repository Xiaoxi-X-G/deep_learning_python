# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 21:27:36 2018

@author: Xiaoxi
"""

# ref: https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# stacking

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

import re

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier)
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import KFold

from plotly.offline import plot
import plotly.graph_objs as go

train = pd.read_csv('../data/titanic/train.csv')
test = pd.read_csv('../data/titanic/test.csv')


print(test['Cabin'].head(100))
print(type(test['Cabin'].iloc[0]))

full_data = [train, test]

## Feature Engineering #########################################
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1) # typo in ref, where .group(0)
    
for dataset in full_data:
    # 1. Has_Cabin
    dataset['Has_Cabin'] = dataset['Cabin'].apply(lambda x: 0 if type(x)==float else 1)
    
    # 2. IsAlone
    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp'] + 1
    dataset['IsAlone'] = dataset['FamilySize'].apply(lambda x: 1 if x==1 else 0)
    
    # 3. Name title
    dataset['Title'] = dataset['Name'].apply(get_title)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
           'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping).fillna(0).astype(int)

    # 4. Embark places
    embark_mapping = {'S': 0, 'C': 1, 'Q': 2} 
    dataset['Embarked'] = dataset['Embarked'].map(embark_mapping).fillna(3).astype(int)

    # 5. map gender
    dataset['Sex'] = dataset['Sex'].map({'male':0, 'female':1}).fillna(3).astype(int)
    
    # 5. Fare categories 
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare']  = 3
    dataset['Fare'] = dataset['Fare'].fillna(0).astype(int)
    
    # 6. Age categories
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].fillna(0).astype(int)


#####Feature Selection ######################################################
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
test = test.drop(drop_elements, axis = 1)


##### correlation #############################################################
# Correlation plot can tell us is that there are not too many features strongly 
# correlated with one another.
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


#### Stacking model ###########################################################
# 1. split data into training, meta_testing
# 2. concept of K fold cross-validation applied to training dataset on different models
# 3. each validation fold produce a prediction 
# 4. with M models, generate M features, or M meta feature or meta training 
# 5. the meta training are used to training stacking model
#
# 6. the meta_testing fit into the model trained by using entire dataset, then 
# feed into the stacking model to find accuracy
# Or the meta_testing fit into each K fold models, and using k-average result to 
# get M intermediate results to fit nito the stacking model to find accuracy (used in this example)

kfold = 5
seed = 0
ntrain = train.shape[0]
ntest = test.shape[0]
kf = KFold(n_splits = kfold, random_state = seed, shuffle = True) 

class SklearnHelper(object):
    def __init__(self, clf, seed = 0, params = None):
        params['random_state'] = seed
        self.clf = clf(**params)
    
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
        
    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x,y)
        
    def feature_importance(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)
        return self.clf.fit(x, y).feature_importances_
    

def get_oof(clf, x_train, y_train, x_test, train):
    oof_train = np.zeros((ntrain,)) #each model predict results used as meta_training set 
    oof_test_skf = np.empty((kfold, ntest)) # meta test results for each train fold model    
    oof_test = np.zeros((ntest,)) # average acorss k-fold used as the meta_test
    
    #print('the loop starts...')
    for i,  (train_index, test_index) in enumerate(kf.split(train)):  # has to put .split(train) here, otherwise, it only works one time if put outside function inline 115
        #print(i)
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        
        clf.train(x_tr, y_tr)
        
        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test) 
        
    oof_test[:] = oof_test_skf.mean(axis = 0) # row sum
    return oof_train.reshape(-1, 1), oof_test.reshape(-1,1) # (n,) --> (n,1)
    

# set up modelling
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500, # number of tree
    'warm_start': True,
    'max_depth': 6,
    'min_samples_leaf':2,
    'max_features': 'sqrt',
    'verbose': 0
}

ada_params = {  # more penalty on mistakes
    'n_estimators': 500,
    'learning_rate': 0.75
}

gb_params = { # modelling on residuals error using rf
     'n_estimators': 500,       # number of trees
     'max_depth': 5,
     'min_samples_leaf': 2,
     'verbose': 0
}

svc_params = {
    'kernel': 'linear',
    'C': 0.025
}


###
rf = SklearnHelper(clf = RandomForestClassifier, seed = seed, params = rf_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=seed, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=seed, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=seed, params=svc_params)


###
y_train = train['Survived'].ravel() # flatten to array
train = train.drop(['Survived'], axis = 1)
x_train = train.values
x_train.shape
x_train.sum().sum()

x_test = test.values
x_test.shape
x_test.sum().sum()

###
rf_oof_train, rf_oof_test = get_oof(clf = rf, x_train = x_train, y_train = y_train, x_test = x_test, train = train) # Random Forest
ada_oof_train, ada_oof_test = get_oof(clf = ada, x_train=x_train, y_train=y_train, x_test=x_test, train = train) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(clf = gb, x_train= x_train, y_train= y_train, x_test = x_test, train = train) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(clf =svc,x_train=x_train, y_train=y_train, x_test=x_test, train = train) # Support Vector Classifier

### feature importance
rf_feature = rf.feature_importance(x_train,y_train)
ada_feature = ada.feature_importance(x_train, y_train)
gb_feature = gb.feature_importance(x_train,y_train)


cols = train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_feature,
      'AdaBoost feature importances': ada_feature,
    'Gradient Boost feature importances': gb_feature
    })


################################# plotly
def scatter_plot_feature(model_name):    
    trace = go.Scatter(
            y = feature_dataframe[model_name+ ' feature importances'].values,
            x = feature_dataframe['features'],
            mode = 'markers',
            marker = dict(
                    sizemode = 'diameter',
                    sizeref = 1,
                    size = 25,
                    color = feature_dataframe[model_name + ' feature importances'].values,
                    colorscale = 'Portland',
                    showscale = True
                    )
            )    
    plot_data = [trace]    
    layout = go.Layout(
            autosize = True,
            title = model_name + ' feature importance',
            yaxis = dict(
                    title = 'feature importance',
                    gridwidth = 2,
                    ticklen = 5
                    )
            )    
    fig = go.Figure(data = plot_data, layout = layout)
    plot(fig)

model_name = 'Gradient Boost' #'Random Forest' 'Gradient Boost' 'AdaBoost'
scatter_plot_feature(model_name)


################################################################################
# second level prediction using first level output

base_prediction_train = pd.DataFrame({
        'RandomForest': rf_oof_train.ravel(),
        'AdaBoost': ada_oof_train.ravel(),
        'GradientBoost': gb_oof_train.ravel()
        })
    
base_prediction_test = pd.DataFrame({
        'RandomForest': rf_oof_test.ravel(),
        'AdaBoost': ada_oof_test.ravel(),
        'GradientBoost': gb_oof_test.ravel()
        })
    
# check correlation among predictors, a less correlated featurs is preferred
base_prediction_train.dropna(axis = 0, inplace = True)
#base_prediction_train = base_prediction_train.apply(pd.to_numeric, axis = 1)
basemodel_corr = base_prediction_train.astype(float).corr()

trace = go.Heatmap(z = basemodel_corr.values,
                   x = basemodel_corr.columns.values,
                   y = basemodel_corr.columns.values,
                   colorscale='Viridis',
                   showscale = True)

plot_data = [trace]
layout = go.Layout(
        autosize = True,
        title = 'base model correlation',
        yaxis = dict(
                title = 'y-axis',
                titlefont = dict(
                            family = 'Courier New',
                            size = 14,
                            color = '#007dba'
                        )
                )        
        )
fig = go.Figure(data = plot_data, layout = layout)
plot(fig)


##############################################################################
### train meta model using prediction results from base model
x_train = np.concatenate((rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate((rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)

predictions = gbm.predict(x_test)

#### plot 
meta_trainResult = pd.DataFrame({
        'rf': rf_oof_train.ravel(),
        'ada': ada_oof_train.ravel(),
        'gb': gb_oof_train.ravel(),
        'svc': svc_oof_train.ravel(),
        'MEGA': gbm.predict(x_train)
        })

    
def df_heatmap(df):
    trace = go.Heatmap(z = df.values,
                       x = df.columns.values,
                       y = df.columns.values,
                       colorscale='Viridis',
                       showscale = True)
    
    plot_data = [trace]
    layout = go.Layout(
            autosize = True,
            title = 'model correlation',
            yaxis = dict(
                    title = 'y-axis',
                    titlefont = dict(
                                family = 'Courier New',
                                size = 14,
                                color = '#007dba'
                            )
                    )        
            )
    fig = go.Figure(data = plot_data, layout = layout)
    plot(fig)    


#
df_heatmap(meta_trainResult.astype(float).corr())