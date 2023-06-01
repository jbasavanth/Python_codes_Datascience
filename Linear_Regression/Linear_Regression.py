#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df = pd.read_csv('USA_Housing.csv')


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.columns


# In[10]:


sns.pairplot(df)


# In[12]:


sns.distplot(df['Price'])


# In[14]:


sns.heatmap(df.corr(), annot = True)


# In[15]:


df.columns


# In[16]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[17]:


y = df['Price']


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


lm = LinearRegression()


# In[22]:


lm.fit(X_train,y_train)


# In[23]:


print(lm.intercept_)


# In[24]:


lm.coef_


# In[25]:


X_train.columns


# In[27]:


cdf = pd.DataFrame(lm.coef_,X.columns, columns=['Coeff'])


# In[28]:


cdf


# ## Predictions

# In[29]:


predictions = lm.predict(X_test)


# In[30]:


predictions


# In[31]:


plt.scatter(predictions,y_test)


# In[32]:


sns.distplot((y_test-predictions))


# In[33]:


from sklearn import metrics


# In[34]:


metrics.mean_absolute_error(y_test,predictions)


# In[35]:


metrics.mean_squared_error(y_test,predictions)


# In[36]:


np.sqrt(metrics.mean_squared_error(y_test,predictions))


# In[ ]:




