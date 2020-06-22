#!/usr/bin/env python
# coding: utf-8

# ## Clean data

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns; sns.set(color_codes=True)
from sklearn.decomposition import PCA


# #### Read in dataset (Student Performance in Maths)

# In[61]:


student_dataset = pd.read_csv("../Dataset/student-mat.csv",sep = ";")
student_dataset


# #### Dealing with non numerical data

# In[62]:


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


# In[63]:


change_to_numerical(student_dataset)


# #### Visualize data

# In[6]:


features = student_dataset.columns.values
for feature in features:
    fig, ax = plt.subplots()
    ax.plot(student_dataset[feature])
    ax.set_title(feature)

student_dataset.columns.values


# In[7]:


sns.set(font_scale=1.4)
f, ax = plt.subplots(figsize=(30, 30))
corr = student_dataset.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Student Performance Correlation Heatmap', fontsize=14)


# #### Split input data and output data

# In[76]:


student_input = student_dataset.drop(columns=['G3'])
student_output = G3_values= student_dataset['G3']


# #### Split train and test data

# In[77]:


x_train , x_test , y_train , y_test = train_test_split(student_input , student_output, test_size=0.2)


# #### PCA to increase learning speed

# In[339]:


# still working on this
# change student dataset to training datasets


# In[109]:


pca = PCA(n_components=15)
comp = pca.fit_transform(student_input)
#print(np.sum(pca.explained_variance_ratio_))
#print(comp)


# In[110]:


pca = PCA(0.95)
pca.fit(student_dataset)
#print(pca.n_components_)
train_img_pca = pca.transform(student_dataset)
#print(train_img_pca)


# ### Functions to return cleaned data

# In[14]:


def student_np():
    return x_train.to_numpy() , x_test.to_numpy() , y_train.to_numpy() , y_test.to_numpy()


# In[18]:


def student_df():
    return x_train , x_test , y_train , y_test


# In[19]:


def pca_data():
    return 0


# ## Decision tree

# In[79]:


x_train , x_test , y_train , y_test = student_np()


# In[27]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn import metrics 


# In[106]:


clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)


# In[107]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[108]:


print(metrics.confusion_matrix(y_test, y_pred))


# In[ ]:




