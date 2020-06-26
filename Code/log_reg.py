#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import math


# In[2]:


data_file = open('data_banknote_authentication.txt', 'r')
data_string = data_file.read()
data_strings = np.array(data_string.split('\n'))
x_data = []
y_data = []
data = []


# In[3]:


print(data_strings)
data_strings[0].split(',')
for i in range(data_strings.shape[0]):
    a = data_strings[i].split(',')
    x_data.append(a[0:4])
    y_data.append(a[4])
    data.append(data_strings[i].split(','))
    
x_data = np.array(x_data)   
y_data = np.array(y_data)
data = np.array(data)
np.random.shuffle(data)


# In[4]:


a = data[761]
a = list(a)
print(a[0:4])
a.insert(0,1)
print(a)


# In[5]:


float(data[1][4])+float(data[1371][4])
print(data[1300][4])


# In[6]:


def con(a):
    b = []
    for i in range(len(a)):
        b.append(float(a[i]))
    return b


# In[7]:


def sigmoid(x):
    return 1/(1+math.exp(-x))


# In[8]:


def f(x, θ): # Regression function
    a = np.dot(x, θ)
    return sigmoid(a) # linear regression using the dot product


# In[9]:


ui = data[0:824] #training data, contains 60 percent of the data
uii = data[824:1098] #validation data, contains 20 percent of the data
uiii = data[1098:1373] #testing data, contains the remaining 20 percent of the code
dataa = ui


# In[10]:


α = 1e-6 # dlearning rate
θ = np.ones(5) # initialize our parameters
θ_old = np.zeros(5) # initialize the old parameter values (it's different from the parameter values so we can enter the while loop below)
while np.sqrt(np.sum(np.power(θ - θ_old, 2))) > 0.001: # while euclidean norm > 0.001 (so ϵ = 0.001) 
    θ_old = θ # set old parameter values to parameter values before they are updated
    for i in range(dataa.shape[0]): # loop over each row of the design matrix (each data point)
        a = dataa[i]
        a = list(a[0:4])
        a.insert(0,1)
        a = np.array(a)
        a = con(a)
        θ = θ + α*(  (float(dataa[i][4])-(f(a, θ) )) * np.array(a)) # update the parameters using the update rule
        #print(θ)
                   

print("Model Parameters: ", θ) # Print model parameters after convergence


# In[11]:


dataa = uii
while np.sqrt(np.sum(np.power(θ - θ_old, 2))) > 0.001: # while euclidean norm > 0.1 (so ϵ = 0.001) 
    θ_old = θ # set old parameter values to parameter values before they are updated
    for i in range(dataa.shape[0]): # loop over each row of the design matrix (each data point)
        a = dataa[i]
        a = list(a[0:4])
        a.insert(0,1)
        a = np.array(a)
        a = con(a)
        θ = θ + α*(  (float(dataa[i][4])-(f(a, θ) )) * np.array(a)) # update the parameters using the update rule
        #print(θ)
                   

print("Model Parameters: ", θ) # Print model parameters after convergence


# In[12]:


def check(p):
    if p>=0.5:
        return 1
    else:
        return 0


# In[26]:


right = 0
wrong = 0
for i in range(uiii.shape[0]):
    #print(uiii[i][4])
    b = list(con(uiii[i][0:4]))
    #print(b)
    b.insert(0,1)
    #print(check(f(b,θ)))
    if check(f(b,θ)) == int(uiii[i][4]):
        right = right + 1
    else:
        wrong = wrong + 1
        
        
print("Percentage of testing data we got right ",((right/274)*100))
print("Percentage of testing data we got wrong ",((wrong/274)*100))


# In[ ]:





# In[25]:


confusion_matrix = np.matrix([[0,0],[0,0]])
for i in range(uiii.shape[0]):
    b = list(con(uiii[i][0:4]))
    b.insert(0,1)
    if check(f(b,θ)) == int(uiii[i][4]):
        if check(f(b,θ))== 0:
            confusion_matrix[0,0] = confusion_matrix[0,0]+1
        else:
            confusion_matrix[1,1] = confusion_matrix[1,1]+1
    else:
        if check(f(b,θ)) == 1:
            confusion_matrix[0,1] = confusion_matrix[0,1]+1
        else:
            confusion_matrix[1,0] = confusion_matrix[1,0]+1
            
print(confusion_matrix)
print("The top left coner is the amount of fake bills we predicted as fake")
print("The top right coner is the amount of fake bills we predicted as real")
print("The bottom right coner is the amount of real bills we predicted as real")
print("The bottom left coner is the amount of real bills we predicted as fake")
print("Therefor our accuracy is", ((confusion_matrix[0,0]+confusion_matrix[1,1])/274)*100)


# In[ ]:




