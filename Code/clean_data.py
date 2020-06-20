#!/usr/bin/env python
# coding: utf-8

# ## Clean data

# In[107]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# #### Read in dataset (Student Performance in Maths)

# In[76]:


student_dataset = pd.read_csv("../Dataset/student-mat.csv",sep = ";")
student_dataset.head()

# There are 3 different outcomes since there are 3 different test results
# Split data based on outcomes

G1_dataset = student_dataset.drop(columns = ["G2","G3"])
G2_dataset = student_dataset.drop(columns = ["G1","G3"])
G3_dataset = student_dataset.drop(columns = ["G1","G2"])


# In[77]:


student_dataset.head()


# In[78]:


G1_dataset.head()


# In[79]:


G2_dataset.head()


# In[80]:


G3_dataset.head()


# In[40]:


student_dataset.info()


# #### Dealing with non numerical data

# In[32]:


#https://pythonprogramming.net/working-with-non-numerical-data-machine-learning-tutorial/

def change_to_numerical(data):
    features = data.columns.values
    for feature in features:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if data[feature].dtype != np.int64:
            feature_contents = data[feature].values.tolist()
            elements = set(feature_contents)
            i = 0
            for element in elements:
                if element not in text_digit_vals:
                    text_digit_vals[element] = i
                    i+=1

            data[feature] = list(map(convert_to_int, data[feature]))


# In[86]:


change_to_numerical(student_dataset)
student_dataset.head()


# In[81]:


change_to_numerical(G1_dataset)
G1_dataset.head()


# In[82]:


change_to_numerical(G2_dataset)
G2_dataset.head()


# In[83]:


change_to_numerical(G3_dataset)
G3_dataset.head()


# In[43]:


student_dataset.info()


# #### Normalize data

# In[84]:


#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

def normalize(data):
    scaler = StandardScaler()
    scaler.fit(data)
    normalized_data = scaler.fit_transform(data)
    
    return normalized_data


# In[96]:


normalized_student_array = normalize(student_dataset)
print(normalized_student_array[:2])


# In[97]:


normalized_G1_array = normalize(G1_dataset)
print(normalized_G1_array[:2])


# In[98]:


normalized_G2_array = normalize(G2_dataset)
print(normalized_G2_array[:2])


# In[99]:


normalized_G3_array = normalize(G3_dataset)
print(normalized_G3_array[:2])


# #### Split data

# In[134]:


student_x_train , student_x_test , student_y_train , student_y_test = train_test_split(
    normalized_student_array[:,:-1] , normalized_student_array[:,32], test_size=0.2)

print("train input data : ", student_x_train.shape)
print("test input data : ", student_x_test.shape)
print("train output data : ", student_y_train.shape)
print("test output data : ", student_y_test.shape)


# In[135]:


G1_x_train , G1_x_test , G1_y_train , G1_y_test = train_test_split(
    normalized_G1_array[:,:-1] , normalized_G1_array[:,30], test_size=0.2)

print("train input data : ", G1_x_train.shape)
print("test input data : ", G1_x_test.shape)
print("train output data : ", G1_y_train.shape)
print("test output data : ", G1_y_test.shape)


# In[136]:


G2_x_train , G2_x_test , G2_y_train , G2_y_test = train_test_split(
    normalized_G2_array[:,:-1] , normalized_G2_array[:,30], test_size=0.2)

print("train input data : ", G2_x_train.shape)
print("test input data : ", G2_x_test.shape)
print("train output data : ", G2_y_train.shape)
print("test output data : ", G2_y_test.shape)


# In[137]:


G3_x_train , G3_x_test , G3_y_train , G3_y_test = train_test_split(
    normalized_G3_array[:,:-1] , normalized_G3_array[:,30], test_size=0.2)

print("train input data : ", G3_x_train.shape)
print("test input data : ", G3_x_test.shape)
print("train output data : ", G3_y_train.shape)
print("test output data : ", G3_y_test.shape)


# ### Functions to return cleaned data

# In[138]:


def student_G1_data():
    return G1_x_train , G1_x_test , G1_y_train , G1_y_test


# In[139]:


def student_G2_data():
    return G2_x_train , G2_x_test , G2_y_train , G2_y_test


# In[140]:


def student_G3_data():
    return G3_x_train , G3_x_test , G3_y_train , G3_y_test


# In[ ]:




