#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# In[5]:


df = pd.read_csv('E:/Coding/DSBDA/PRACTICAL/Air n Heart/air quality.csv',encoding='ISO-8859-1')
df


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.columns


# In[9]:


df['so2'] = df['so2'].astype('float32')
df['no2'] = df['no2'].astype('float32')
df['rspm'] = df['rspm'].astype('float32')
df['spm'] = df['spm'].astype('float32')
df['date'] = df['date'].astype('string')

df.info()


# In[10]:


df=df.drop_duplicates()


# In[11]:


df.isna().sum()


# In[14]:


per=df.isna().sum()*100/len(df)
per


# In[15]:


per.sort_values(ascending=False)


# In[16]:


df=df.drop(['stn_code', 'agency','sampling_date','location_monitoring_station','pm2_5'],axis=1)


# In[17]:


df.head()


# In[18]:


for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype =='string':
        df[col]=df[col].fillna(df[col].mode()[0])
    else:
        df[col]=df[col].fillna(df[col].mean())


# In[19]:


df.isnull().sum()


# In[20]:


df.isna().sum()


# In[21]:


df1 = df[['state','location']]
df1


# In[22]:


df2 = df[['type','so2']]
df2


# In[23]:


df_concat=pd.concat([df1,df2],axis =1)
df_concat


# In[24]:


def remove_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    threshold = 1.5 * IQR
    outlier_mask = (column < Q1 - threshold) | (column > Q3 + threshold)
    return column[~outlier_mask]


# In[25]:


col_name = ['so2', 'no2', 'rspm', 'spm']
for col in col_name:
    df[col] = remove_outliers(df[col])


# In[26]:


col_label=['state','location','type']
encoder=LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])


# In[27]:


df


# In[28]:


plt.figure(figsize=(10, 6))  # Adjust the figure size if needed

for col in col_name:
    sns.boxplot(data=df[col])
    plt.title(col)
    plt.show()


# In[ ]:




