#!/usr/bin/env python
# coding: utf-8

# <a target="_blank" href="https://colab.research.google.com/github/RanjeetKumbhar01/TE_IT_DSBDA_ASSIGNMENTS_SPPU/blob/main/Group_B/B_2/b_2_Heart.ipynb">
#   <img align="left" alt="Goolge Colab"  src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# In[1]:


# import pandas library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Reading csv file
df = pd.read_csv("Heart.csv")
df.head()


# ## Data Cleaning

# In[3]:


df = df.drop_duplicates()


# In[4]:


# Count ,min,max ,etc of each column
df.describe()


# In[5]:


# Information about each column data
df.info()


# In[6]:


#Finding null values in each column
df.isna().sum()


# ## Data Integration

# In[7]:


df.head()


# In[8]:


df.fbs.unique()


# In[9]:


subSet1 = df[['age','cp','chol','thalachh']]


# In[10]:


subSet2 = df[['exng','slp','output']]


# In[11]:


merged_df = subSet1.merge(right=subSet2,how='cross')
merged_df.head()


# ## Error Correcting

# In[12]:


df.columns


# In[13]:


def remove_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    threshold = 1.5 * IQR
    outlier_mask = (column < Q1 - threshold) | (column > Q3 + threshold)
    return column[~outlier_mask]


# In[14]:


# Remove outliers for each column using a loop
col_name = ['cp','thalachh','exng','oldpeak','slp','caa']
for col in col_name:
    df[col] = remove_outliers(df[col])


# In[15]:


plt.figure(figsize=(10, 6))  # Adjust the figure size if needed

for col in col_name:
    sns.boxplot(data=df[col])
    plt.title(col)
    plt.show()


# In[16]:


df = df.dropna()


# In[17]:


df.isna().sum()


# In[18]:


df = df.drop('fbs',axis=1)


# In[19]:


# Compute correlations between features and target
correlations = df.corr()['output'].drop('output')

# Print correlations
print("Correlation with the Target:")
print(correlations)
print()

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[20]:


# df.isna().sum()


# ## Data Split

# In[21]:


# splitting data using train test split
x = df[['cp','thalachh','exng','oldpeak','slp','caa']]
y = df.output
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

x_train.shape,x_test.shape,y_train.shape,y_test.shape


# ## Data transformation

# In[22]:


from sklearn.preprocessing import StandardScaler


# In[23]:


scaler = StandardScaler()


# In[24]:


x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[ ]:





# ## Data model building

# In[25]:


y_train= np.array(y_train).reshape(-1, 1)
y_test= np.array(y_test).reshape(-1, 1)


# In[26]:


y_train.shape


# In[27]:


model = LogisticRegression()
model.fit(x_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test_scaled)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[28]:


#Classification model using Decision Tree
from sklearn.tree import DecisionTreeClassifier
tc=DecisionTreeClassifier(criterion='entropy')
tc.fit(x_train_scaled,y_train)
y_pred=tc.predict(x_test_scaled)

print("Training Accuracy Score :",accuracy_score(y_pred,y_test))
print("Training Confusion Matrix  :",confusion_matrix(y_pred,y_test))

