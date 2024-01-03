#!/usr/bin/env python
# coding: utf-8

# ### SAIMPLE LINEAR REGRESSION 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/100-days-of-machine-learning/main/day48-simple-linear-regression/placement.csv")


# In[3]:


df.head()


# In[4]:


df.isnull()


# In[5]:


df.describe()


# In[6]:


plt.scatter(df['cgpa'],df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')


# In[7]:


X = df.iloc[:,0:1]
y = df.iloc[:,-1]


# In[8]:


y


# In[9]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LinearRegression


# In[10]:


lr = LinearRegression()


# In[11]:


lr.fit(X_train,y_train)


# In[12]:


X_test


# In[13]:


y_test


# In[14]:


lr.predict(X_test.iloc[0].values.reshape(1,1))


# In[15]:


plt.scatter(df['cgpa'],df['package'])
plt.plot(X_train,lr.predict(X_train),color='red')
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')


# In[16]:


m = lr.coef_
b = lr.intercept_

# y = mx + b
m * 8.58 + b


# In[17]:


m * 9.5 + b


# In[18]:


m * 100 + b


# ### MATHEMATICAL WAY

# In[19]:


class MeraLR:
    
    def __init__(self):
        self.m = None
        self.b = None
        
    def fit(self,X_train,y_train):
        
        num = 0
        den = 0
        
        for i in range(X_train.shape[0]):
            
            num = num + ((X_train[i] - X_train.mean())*(y_train[i] - y_train.mean()))
            den = den + ((X_train[i] - X_train.mean())*(X_train[i] - X_train.mean()))
        
        self.m = num/den
        self.b = y_train.mean() - (self.m * X_train.mean())
        print(self.m)
        print(self.b)       
    
    def predict(self,X_test):
        
        print(X_test)
        
        return self.m * X_test + self.b


# In[20]:


df.head()


# In[21]:


X = df.iloc[:,0].values
y = df.iloc[:,1].values


# In[22]:


X


# In[23]:


y


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[25]:


X_train.shape


# In[26]:


lr = MeraLR()


# In[27]:


lr.fit(X_train,y_train)


# In[28]:


X_train.shape[0]


# In[29]:


X_train[0]



# In[30]:


X_train.mean()



# In[31]:


X_test[0]



# In[32]:


print(lr.predict(X_test[0]))


# In[ ]:




