#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')
import seaborn as sn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


df=pd.read_csv("mushrooms_11.csv")
df.head()


# In[29]:


df.describe()


# In[30]:


df.isnull().sum()


# In[33]:


X=df.drop('class',axis=1)
y=df['class']
X.head()


# In[35]:


from sklearn.preprocessing import LabelEncoder
Encoder_X=LabelEncoder()
for col in X.columns:
    X[col]=Encoder_X.fit_transform(X[col])
Encoder_y=LabelEncoder()
y=Encoder_y.fit_transform(y)


# In[36]:


X.head() 


# In[40]:


X.to_csv('encoded_X_values.csv')


# In[41]:


y


# In[42]:


X.dtypes


# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[53]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train= sc.fit_transform(X_train)
Y_test=sc.transform(X_test)


# In[54]:


type(X_train)


# In[58]:


from sklearn.ensemble import RandomForestClassifier
My_model= RandomForestClassifier(n_estimators=50,criterion='entropy',random_state=42)
result=My_model.fit(X_train,y_train)


# In[62]:


predictions=result.predict(X_test)
X_test


# In[63]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test,predictions))


# In[66]:


from sklearn.metrics import confusion_matrix
conf_matrix=confusion_matrix(predictions,y_test)
confusion_df=pd.DataFrame(conf_matrix, index=['Actual 0','Actual 1'],columns=['predicted 0','predicted 1'])
sn.heatmap(confusion_df,cmap='coolwarm',annot=True)


# In[70]:


from sklearn import metrics
print('\n** Classification Report:\n',metrics.classification_report(y_test,predictions))


# In[72]:


pred_new = result.predict([[5,2,4,1,6,1,0,1,4,0,2,7,7,0,2,1,4,2,3,5.5,7,4]])
pred_new


# In[ ]:




