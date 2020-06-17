#!/usr/bin/env python
# coding: utf-8

# ## Clean data

# In[1]:


import pandas as pd
import numpy as np


# #### Read in dataset (Student Performance)

# In[8]:


student_dataset = pd.read_csv("../Dataset/student-mat.csv",sep = ";")
student_dataset.head()


# #### Read in dataset (Caesarian)

# In[9]:


# Still need to fix the file. change from arff to csv


# ### Functions to return cleaned data

# In[10]:


def student_data():
    return student_dataset


# In[11]:


def caesarian_data():
    return 0


# In[ ]:




