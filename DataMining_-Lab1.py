#!/usr/bin/env python
# coding: utf-8

# In[318]:


#### Reading data

import glob
import pandas as pd
import csv

def readData(fn):
    df = pd.concat([pd.read_csv(f, skiprows = 2) for f in glob.glob(fn)], sort = False)
    return df
df = readData('/home/bassantahmed/Downloads/segmentation.*')
#df.isnull().values.any().sum()
#df[pd.isnull(df).any(axis=1)]
df = df.dropna()


# In[383]:


#### histograms with bins = 5

import matplotlib.pyplot as plt
from random import randint

get_ipython().run_line_magic('matplotlib', 'inline')

colors = []
features = []
ticks = [0.4] * 19
classes = df.CLASS.unique()

for i in range(19):
    colors.append('#%06X' % randint(0, 0xFFFFFF))
    ticks[i] = ticks[i] + i

for c in (classes):
    df_temp =  df['CLASS'] == c
    df1 = df[df_temp]
    df1 = df1.drop('CLASS', axis=1)
    features = df1.columns
    plt.hist(df1.values, 5, density=True, histtype='bar', color=colors)
    plt.legend(labels=features,prop={'size': 8},loc=(1.04,0))
    plt.savefig('hist_' + c + '.png',bbox_inches="tight")


# In[320]:


#### histograms with bins = 10

for c in (classes):
    df_temp =  df['CLASS'] == c
    df1 = df[df_temp]
    df1 = df1.drop('CLASS', axis=1)
    plt.hist(df1.values, 10, density=True, histtype='bar', color=colors)
    plt.legend(labels=df1.columns,prop={'size': 8},loc=(1.04,0))
    plt.savefig('hist2_' + c + '.png',bbox_inches="tight")


# In[321]:


#### Boxplots
df.boxplot()
plt.xticks(rotation='vertical')


# In[386]:


#### Corr

from scipy.stats import pearsonr
import numpy as np
from matplotlib import cm as cm

fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap('jet', 50)
cax = ax1.imshow(df.corr(method ='pearson'), interpolation="nearest", cmap=cmap)
ax1.grid(True)
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
ax1.set_xticklabels(features, rotation = 'vertical')
ax1.set_yticklabels(features,fontsize=8)
fig.colorbar(cax)
plt.show()


# In[323]:


#### min_max normalization

import pandas as pd
from sklearn import preprocessing

x = df.iloc[:,1:20].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_min_max = pd.DataFrame(x_scaled)


# In[324]:


### z-score normalization

from scipy import stats
df_z = pd.DataFrame()
for feature in (features):
    if df[feature].std(ddof=0) == 0:
        df_z[feature] = df[feature]
    else:
        df_z[feature] = (df[feature] - df[feature].mean())/df[feature].std(ddof=0)


# In[325]:


#### Boxplots
df_min_max.boxplot()
plt.xticks(rotation='vertical')


# In[326]:


#### Boxplots
df_z.boxplot()
plt.xticks(rotation='vertical')


# In[351]:


from sklearn.decomposition import PCA

pca = PCA().fit(df_z)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.show()


# In[353]:


#df.isnull().values.any().sum()
pca = PCA(n_components=12)
dataset = pca.fit_transform(df_z)
print(np.sum(pca.explained_variance_ratio_))


# In[354]:


dataset_df = pd.DataFrame(dataset, columns=features[:12])


# In[394]:


fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap('jet', 50)
cax = ax1.imshow(dataset_df.corr(method ='pearson'), interpolation="nearest", cmap=cmap)
ax1.grid(True)
ax1.set_xticks(ticks[0:12])
ax1.set_yticks(ticks[0:12])
ax1.set_xticklabels(features[:12],fontsize=8, rotation='vertical')
ax1.set_yticklabels(features[:12],fontsize=8)
fig.colorbar(cax)
plt.show()


# In[356]:


from sklearn.feature_selection import SelectKBest, chi2
reduced_data = pd.DataFrame(SelectKBest(chi2, k=12).fit_transform(df_min_max.iloc[:,0:12],df['CLASS']), columns=features[:12])


# In[393]:


fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap('jet', 50)
cax = ax1.imshow(reduced_data.corr(method ='pearson'), interpolation="nearest", cmap=cmap)
ax1.grid(True)
ax1.set_xticks(ticks[0:12])
ax1.set_yticks(ticks[0:12])
ax1.set_xticklabels(features[:12],fontsize=8, rotation='vertical')
ax1.set_yticklabels(features[:12],fontsize=8)
fig.colorbar(cax)
plt.show()


# In[ ]:





# In[ ]:




