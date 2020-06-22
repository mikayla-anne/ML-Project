#!/usr/bin/env python
# coding: utf-8

# ## Clean data

# In[77]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# #### Read in dataset (Student Performance in Maths)

# In[78]:


student_dataset = pd.read_csv("../Dataset/student-mat.csv",sep = ";")
student_dataset.head()

# There are 3 different outcomes since there are 3 different test results
# Split data based on outcomes

G1_dataset = student_dataset.drop(columns = ["G2","G3"])
G2_dataset = student_dataset.drop(columns = ["G1","G3"])
G3_dataset = student_dataset.drop(columns = ["G1","G2"])


# #### Dealing with non numerical data

# In[79]:


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


# In[80]:


change_to_numerical(student_dataset)


# #### Split input data and output data

# In[81]:


student_data = student_dataset.drop(columns=['G1','G2','G3'])


# In[82]:


G1_values= student_dataset['G1']
G2_values= student_dataset['G2']
G3_values= student_dataset['G3']


# #### Split train and test data

# In[83]:


G1_x_train , G1_x_test , G1_y_train , G1_y_test = train_test_split(student_data , G1_values, test_size=0.2)
G2_x_train , G2_x_test , G2_y_train , G2_y_test = train_test_split(student_data , G2_values, test_size=0.2)
G3_x_train , G3_x_test , G3_y_train , G3_y_test = train_test_split(student_data , G3_values, test_size=0.2)


# ### Functions to return cleaned data

# In[84]:


def student_G1_data():
    return G1_x_train , G1_x_test , G1_y_train , G1_y_test


# In[85]:


def student_G2_data():
    return G2_x_train , G2_x_test , G2_y_train , G2_y_test


# In[86]:


def student_G3_data():
    return G3_x_train , G3_x_test , G3_y_train , G3_y_test


# In[ ]:





# In[ ]:




