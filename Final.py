#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
import matplotlib as mpl
import itertools
data = pd.read_csv('Car_sales.csv')
data.head()


# In[3]:


df = pd.DataFrame(data)
df


# In[9]:


data_numeric=df.drop(columns=['Manufacturer']).values
data_numeric = df.select_dtypes(include = ['int64','float64']) #data cleaning
data_numeric    #new dataframe


# In[11]:


x = data_numeric.drop(columns = ['Sales_in_thousands']).values
y = data_numeric['Sales_in_thousands'].values 
df['__year_resale_value'].fillna(int(df['__year_resale_value'].mean()), inplace=True)
df['Sales_in_thousands'].fillna(int(df['Sales_in_thousands'].mean()), inplace=True)
df['Price_in_thousands'].fillna(int(df['Price_in_thousands'].mean()), inplace=True)
df['Power_perf_factor'].fillna(int(df['Power_perf_factor'].mean()), inplace=True)
df['Engine_size'].fillna(int(df['Engine_size'].mean()), inplace=True)
df['Horsepower'].fillna(int(df['Horsepower'].mean()), inplace=True)
df['Wheelbase'].fillna(int(df['Wheelbase'].mean()), inplace=True)
df['Width'].fillna(int(df['Width'].mean()), inplace=True)
df['Length'].fillna(int(df['Length'].mean()), inplace=True)
df['Curb_weight'].fillna(int(df['Curb_weight'].mean()), inplace=True)
df['Fuel_capacity'].fillna(int(df['Fuel_capacity'].mean()), inplace=True)
df['Fuel_efficiency'].fillna(int(df['Fuel_efficiency'].mean()), inplace=True)


# In[13]:


from sklearn.model_selection import train_test_split #default 75% training , 25% testing data(if you want to change, change tst_size value)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state = 0) #random state is 0 so that all get same values 


# In[14]:


from sklearn.ensemble import RandomForestRegressor


# In[15]:


ModelRFR=RandomForestRegressor(n_estimators=10,random_state=0)


# In[16]:


ModelRFR.fit(x_train,y_train)


# In[17]:


y_pred = ModelRFR.predict(x_test) 


# In[18]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[19]:


import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)


# In[ ]:


li=[] #list for user input
for i in range(0,11):
    x=input()
    li.append(x)
for i in li:
    i=float(i)
xi=[] #to make list 2-d 
xi.append(li)
y_pred = ModelRFR.predict(xi) 
y_pred


