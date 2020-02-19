#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### Reading data

import glob
import pandas as pd
import csv

def readData(fn):
    df = pd.concat([pd.read_csv(f) for f in glob.glob(fn)], sort = False)
    return df
df = readData('/home/bassantahmed/Documents/CSED20/Data Mining/wdbc.data')
print(df.shape)
df[df.columns[2:]].plot.hist(figsize=(10,10))


# In[2]:


#### Corr. matrix

from scipy.stats import pearsonr
import numpy as np
from matplotlib import cm as cm
import matplotlib.pyplot as plt

features = df.columns

ticks = [0.4] * 32
for i in range(32):
    ticks[i] = ticks[i] + i

fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap('jet', 50)
cax = ax1.imshow(df.corr(method ='pearson'), interpolation="nearest")
ax1.grid(True)
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
ax1.set_xticklabels(features, rotation = 'vertical', fontsize=7)
ax1.set_yticklabels(features,fontsize=7)
fig.colorbar(cax)
plt.show()


# In[3]:


#boxplot to find out if there is any outliers
df.boxplot()
plt.xticks(rotation='vertical')


# In[4]:


#using stratified split to split the data
from sklearn.model_selection import StratifiedShuffleSplit

y = df['Diagnosis'].values
X = df[features[2:]].values
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)


# In[5]:


# split the data into 70% training and 30% testing
X_train = []
y_train = []
X_test = []
y_test = []
list1 = sss.split(X, y)
for train_index, test_index in list1:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[6]:


#redraw the boxplot to see if data is still normalized
df_temp = pd.DataFrame(X_train, columns = features[2:])
df_temp.boxplot()
plt.xticks(rotation='vertical')


# In[7]:


### using z-score normalization to normalize the data

from scipy import stats
df_z = pd.DataFrame()
for feature in (features[2:]):
    if df_temp[feature].std() == 0:
        df_z[feature] = df_temp[feature]
    else:
        df_z[feature] = (df_temp[feature] - df_temp[feature].mean())/df_temp[feature].std()
df_z.boxplot()
plt.xticks(rotation='vertical')


# In[8]:


# plotting the relation between number of components and covered var. ratio to determine the number of components needed
from sklearn.decomposition import PCA

random_state = 42
pca = PCA(n_components=len(features[2:]), random_state=random_state)
pca.fit(df_z)
plt.figure(figsize=(20,8))
plt.title('Relation Between Number of PCA Components taken and Covered Variance Ratio')
plt.xlabel('Number of Taken PCA Components')
plt.ylabel('Covered Variance Ratio')
plt.xticks([i for i in range(1, len(features[2:]) + 1)])
plt.plot([i for i in range(1, len(features[2:]) + 1)], pca.explained_variance_ratio_.cumsum(), color='red', marker='o', linestyle='dashed', linewidth=2, markersize=12)
plt.show()
print(sum(pca.explained_variance_ratio_[0:20]))


# In[9]:


# selecting 20 components according to the plot to capture almost 99.58% of the variance
pca = PCA(n_components=20)
dataset = pca.fit_transform(df_z[features[2:22]])
dataset_df = pd.DataFrame(dataset, columns=features[2:22])


# In[10]:


#plot the corr. matrix of the selected 20 component
ticks = [0.5] * 20
for i in range(20):
    ticks[i] = ticks[i] + i

fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap('jet', 50)
cax = ax1.imshow(dataset_df.corr(method ='pearson'), interpolation="nearest")
ax1.grid(True)
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
ax1.set_xticklabels(features[2:22], rotation = 'vertical', fontsize=7)
ax1.set_yticklabels(features[2:22],fontsize=7)
fig.colorbar(cax)
plt.show()


# In[28]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

parameters = {'max_depth': np.arange(1, 10)}
dtc = DecisionTreeClassifier(random_state=0)
clf = GridSearchCV(dtc, parameters, cv=10)
clf.fit(X_train, y_train)


# In[29]:


y_pred = clf.predict(X_test)


# In[71]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred, average="macro", pos_label='M')
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
print ('acc' , acc, 'pre', pre, 'recall', recall, 'f1', f1)


# In[47]:


from sklearn.ensemble import AdaBoostClassifier

ada_parameters = {'n_estimators': np.arange(1, 100), 'learning_rate' : [0.0000001,1]}
adac = AdaBoostClassifier(n_estimators=10, random_state=0)
ada_clf = GridSearchCV(adac, ada_parameters, cv=10)
ada_clf.fit(X_train, y_train)


# In[74]:


ada_y_pred = ada_clf.predict(X_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
acc_ada = accuracy_score(y_test, ada_y_pred)
pre_ada = precision_score(y_test, ada_y_pred, average="macro", pos_label='M')
recall_ada = recall_score(y_test, ada_y_pred, average="macro")
f1_ada = f1_score(y_test, ada_y_pred, average="macro")
print ('acc' , acc_ada, 'pre', pre_ada, 'recall', recall_ada, 'f1', f1_ada)


# In[73]:


from sklearn.ensemble import RandomForestClassifier

rf_parameters = {'n_estimators': np.arange(1, 100), 'max_depth' : [1,200]}
rfc = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=0)
rf_clf = GridSearchCV(rfc, rf_parameters, cv=10)
rf_clf.fit(X_train, y_train)


# In[75]:


rf_y_pred = rf_clf.predict(X_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
acc_rf = accuracy_score(y_test, rf_y_pred)
pre_rf = precision_score(y_test, rf_y_pred, average="macro", pos_label='M')
recall_rf = recall_score(y_test, rf_y_pred, average="macro")
f1_rf = f1_score(y_test, rf_y_pred, average="macro")
print ('acc' , acc_rf, 'pre', pre_rf, 'recall', recall_rf, 'f1', f1_rf)


# In[ ]:




