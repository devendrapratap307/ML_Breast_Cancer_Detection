#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

data.data


# In[3]:


data.feature_names


# In[4]:


data.target


# In[5]:


data.target_names


# ## Create dataframe

# In[6]:


df = pd.DataFrame(np.c_[data.data , data.target] , columns = [list(data.feature_names) + ['target']])
# df.head()
#df.info()
pd.set_option('display.max_columns', None)

df.head()


# In[7]:


df.tail()


# In[8]:


df.shape


# ## Split data

# In[9]:


X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

X.shape , y.shape


# In[10]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 2020)

X_train.shape, X_test.shape,y_train.shape , y_test.shape


# ## Train Model 1 - support vector machine classification

# In[11]:


from sklearn.svm import SVC
classification_rbf = SVC(kernel = 'rbf')
classification_rbf.fit(X_train,y_train)

classification_rbf.score(X_test,y_test)


# ## Feature Selection

# In[12]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)


# In[13]:


X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)


# ## ML Model 2 Training after scaling

# In[14]:


classification_rbf_2 = SVC(kernel = 'rbf')
classification_rbf_2.fit(X_train_sc , y_train)

classification_rbf_2.score(X_test_sc , y_test)


# ## Model 3 Training after scaling 

# In[15]:


classification_poly = SVC(kernel = 'poly',degree = 3)
classification_poly.fit(X_train_sc , y_train)

classification_poly.score(X_test_sc , y_test)


# ## Model 4 Training after scaling

# In[16]:


classification_linear = SVC(kernel = 'linear')
classification_linear.fit(X_train_sc , y_train)

classification_linear.score(X_test_sc , y_test)


# ## Prediction

# In[17]:


patient1_data = [ 11.42,17.0,130,1297,0.14,0.13,0.30,0.1,0.2,0.05,0.74,0.7,1.15,3.4,74.0,0.01,0.04,0.05,0.01,0.017,0.005,23,25,98,1956,0.13,0.20,0.8,0.4,0.1]
patient1 = np.array([patient1_data])
patient1_sc = sc.transform(patient1)

patient1_sc


# In[18]:


# model4 used - accuracy : 0.964

classification_linear.predict(patient1_sc)


# In[19]:


pred = classification_linear.predict(patient1_sc)
if pred[0] == 0:
    print('Patient has cancer (malignant Tumor)')
else:
    print('Patient has no cancer (benign)')


# In[20]:


# patient2 has no cancer
patient2 = df.iloc[48:49 , 0:-1]
patient2_sc = sc.transform(patient2)


# patient3 has cancer
patient3 = df.iloc[2:3 , 0:-1]
patient3_sc = sc.transform(patient3)


# ## Model 5 Training - Decision Tree

# In[21]:


from sklearn.tree import DecisionTreeClassifier
classifier_gini = DecisionTreeClassifier(criterion = 'gini')
classifier_gini.fit(X_train , y_train)

classifier_gini.score(X_test , y_test)


# ## Model 6 Traing - Random Forest

# In[22]:


from sklearn.ensemble import RandomForestClassifier
classifier_gini_rf = RandomForestClassifier(n_estimators = 100, criterion = 'gini')
classifier_gini_rf.fit(X_train , y_train)

classifier_gini_rf.score(X_test , y_test)


# ## Model 7 Training - K Nearest Neighbor Classification

# In[23]:


from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5)
classifier_knn.fit(X_train , y_train)

classifier_knn.score(X_test , y_test)


# ## Model 8 Training - Naive Bayes Classifier

# In[24]:


from sklearn.naive_bayes import GaussianNB
classifier_gNB = GaussianNB()
classifier_gNB.fit(X_train , y_train)

classifier_gNB.score(X_test , y_test)


# In[25]:


from sklearn.naive_bayes import BernoulliNB
classifier_bNB = BernoulliNB()
classifier_bNB.fit(X_train , y_train)

classifier_bNB.score(X_test , y_test)


# In[26]:


from sklearn.naive_bayes import MultinomialNB
classifier_mNB = MultinomialNB()
classifier_mNB.fit(X_train , y_train)

classifier_mNB.score(X_test , y_test)


# ## Prediction - using naive bayes classifier : GaussianNB (Model 8)

# In[27]:


pred = classifier_gNB.predict(patient1)
if pred[0] == 0:
    print('Patient has cancer (malignant Tumor)')
else:
    print('Patient has no cancer (benign)')


# ## Model 9 Training - Logistic Regression

# In[28]:


from sklearn.linear_model import LogisticRegression
logisticR = LogisticRegression()
logisticR.fit(X_train_sc, y_train)

logisticR.score(X_test_sc , y_test)


# In[29]:


pred = logisticR.predict(patient1)
if pred[0] == 0:
    print('Patient has cancer (malignant Tumor)')
else:
    print('Patient has no cancer (benign)')

