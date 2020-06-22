#!/usr/bin/env python
# coding: utf-8

# ## Clean data

# In[316]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns; sns.set(color_codes=True)
from sklearn.decomposition import PCA


# #### Read in dataset (Student Performance in Maths)

# In[309]:


student_dataset = pd.read_csv("../Dataset/student-mat.csv",sep = ";")


# #### Dealing with non numerical data

# In[310]:


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


# In[311]:


change_to_numerical(student_dataset)


# #### Visualize data

# In[266]:


features = student_dataset.columns.values
for feature in features:
    fig, ax = plt.subplots()
    ax.plot(student_dataset[feature])
    ax.set_title(feature)

student_dataset.columns.values


# In[313]:


sns.set(font_scale=1.4)
f, ax = plt.subplots(figsize=(30, 30))
corr = student_dataset.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Student Performance Correlation Heatmap', fontsize=14)


# #### Split input data and output data

# In[330]:


student_dataset


# In[ ]:





# In[170]:


student_data = student_dataset.drop(columns=['G1','G2','G3'])


# In[171]:


G1_values= student_dataset['G1']
G2_values= student_dataset['G2']
G3_values= student_dataset['G3']


# #### Split train and test data

# In[172]:


G1_x_train , G1_x_test , G1_y_train , G1_y_test = train_test_split(student_data , G1_values, test_size=0.2)
G2_x_train , G2_x_test , G2_y_train , G2_y_test = train_test_split(student_data , G2_values, test_size=0.2)
G3_x_train , G3_x_test , G3_y_train , G3_y_test = train_test_split(student_data , G3_values, test_size=0.2)


# #### PCA to increase learning speed

# In[339]:


# still working on this
# change student dataset to training datasets


# In[332]:


pca = PCA(n_components=15)
comp = pca.fit_transform(student_dataset)
np.sum(pca.explained_variance_ratio_)


# In[336]:


pca = PCA(0.95)
pca.fit(student_dataset)
print(pca.n_components_)
train_img_pca = pca.transform(student_dataset)


# ### Functions to return cleaned data

# In[345]:


def student_G1_data():
    return G1_x_train.to_numpy() , G1_x_test.to_numpy() , G1_y_train.to_numpy() , G1_y_test.to_numpy()


# In[346]:


def student_G2_data():
    return G2_x_train.to_numpy() , G2_x_test.to_numpy() , G2_y_train.to_numpy() , G2_y_test.to_numpy()


# In[347]:


def student_G3_data():
    return G3_x_train.to_numpy() , G3_x_test.to_numpy() , G3_y_train.to_numpy() , G3_y_test.to_numpy()


# In[ ]:


def pca_G1_data():
    return 0


# In[ ]:


def pca_G2_data():
    return 0


# In[ ]:


def pca_G3_data():
    return 0


# In[ ]:




