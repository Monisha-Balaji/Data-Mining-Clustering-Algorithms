#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import itertools
from sklearn.decomposition import PCA
import matplotlib.pyplot as pl
import numpy as np
from random import randint
import math


# In[57]:


# User Input for Dataset & Number of Clusters
file_name = input("Enter file name: ")
clusters = int(input("Enter Number of Clusters:"))
iterations= int(input("Enter Number of Iterations"))
smooth = float(input("Enter Smoothing Value"))
threshold = float(input("Enter Threshold Value"))


# In[58]:


choice = int(input("Enter 1 for user input of initialization parameters and 2 for random"))


# In[59]:


print(smooth)


# In[60]:


# Reading file into Dataframe
Data = pd.read_csv(file_name, sep='\t', lineterminator='\n', header=None)
#Data


# In[61]:


#Extracting attributes and storing it
points = Data.iloc[:,2:]
#points


# In[62]:


#Creating array for all attributes 
points_arr=points.iloc[:,:].values
points_arr
print(np.shape(points_arr))


# In[63]:


size = len(points.index)
print(size)


# In[64]:


cols = len(points_arr[0])
print(cols)
#print(points_arr.shape[0])


# In[65]:


#Removing Outliers for iyer.txt
if file_name=="iyer.txt":
    max=30
    for i in range(size):
        flag=False
        for j in range(cols):
            if points_arr[i][j]>30:
                flag=True
                break
        if flag==True:
            print("Delete: ",i)
            points_a=np.delete(points_arr, i, 0)
            df=Data.drop(i)
            Data=df
    points_arr=points_a
    print(len(points_arr))
    size=len(points_arr)
    print(size)
    points = Data.iloc[:,2:]
    print(len(points))


# In[73]:


if choice==1:
    print("Enter Weight/Prior Probability")
    weight_pi=np.zeros(clusters,dtype=float)
    for i in range(clusters):
        weight_pi[i]=(float(input()))
    cov=np.zeros([clusters,cols,cols], dtype=float)
    #centroids = np.zeros([clusters,cols], dtype = float)
    print("Enter Covariance:")
    for i in range(clusters):
        b=[]
        for j in range(cols):
            a=[]
            for k in range(cols):
                a.append(float(input()))
            b.append(a)
        cov[i]=b
    for i in range(clusters):
        np.fill_diagonal(cov[i], cov[i].diagonal() + smooth)
    mu=np.zeros([clusters,cols],dtype=float)
    print("Enter Mean:")
    for i in range(clusters):
        a=[]
        for j in range(cols):
            a.append(float(input()))
        mu[i]=a
    prob = np.zeros((size,clusters))
else:
    #Initialize Learnable parameters
    weight_pi = np.ones(clusters)/clusters
    cov = np.array([np.eye(cols)] * clusters) 
    for i in range(clusters):
        np.fill_diagonal(cov[i], cov[i].diagonal() + smooth)
    prob = np.zeros((size,clusters))
    mu = points_arr[np.random.randint(0, size, size=clusters)]


# In[74]:


print(mu)
print(cov)
print(weight_pi)
#print(weight_pi.dtype)


# In[75]:


print(np.linalg.det(cov))


# In[76]:


from scipy.stats import multivariate_normal as mvn
def Estep():
    #print("cov:",np.shape(cov),"mu:", np.shape(mu),"weights_pi:", np.shape(weight_pi))
    prob = np.zeros((clusters, size))
    for j in range(clusters):
        for i in range(size):
            #print(weight_pi[j])
            print(mvn(mu[j], cov[j]).pdf(points_arr[i]))
            prob[j, i] = weight_pi[j] * mvn(mu[j], cov[j]).pdf(points_arr[i])
    #print(prob)
    prob /= prob.sum(0)
    #print("Prob:",prob)
    return prob


# In[77]:


def Mstep(prob):
    weight_pi = np.zeros(clusters)
    for j in range(clusters):
        for i in range(size):
            weight_pi[j] += prob[j, i]
    #print("wt1:",weight_pi)
    weight_pi /= size
    #print("wt:",weight_pi)
    mu = np.zeros((clusters, cols))
    for j in range(clusters):
        for i in range(cols):
            mu[j] += prob[j, i] * points_arr[i]
        mu[j] /= prob[j, :].sum()      
    cov = np.zeros((clusters, cols, cols))
    for j in range(clusters):
        for i in range(size):
            ys = np.reshape(points_arr[i]- mu[j], (cols,1))
            cov[j] += prob[j, i] * np.dot(ys, ys.T)
        cov[j] /= prob[j,:].sum()
    #Adding smoothing value
    for i in range(clusters):
        np.fill_diagonal(cov[i], cov[i].diagonal() + smooth)
    return mu,cov,weight_pi
      


# In[78]:


ll_old=0
for i in range(iterations):
    print("Iterations:", i)
    prob=Estep()
    mu,cov,weight_pi = Mstep(prob)
    print("Mean",mu)
    print("Covariance",cov)
    print("Weights",weight_pi)
    ll_new = 0.0
    for i in range(size):
        s = 0
        for j in range(clusters):
            s += weight_pi[j] * mvn(mu[j], cov[j]).pdf(points_arr[i])
        ll_new += np.log(s)
    print(ll_new,ll_old)
    print("diff",np.abs(ll_new - ll_old))
    if np.abs(ll_new - ll_old) < threshold:
        print("Difference reached threshold")
        break
    ll_old = ll_new


# In[ ]:


clust_dict={}

for c in range(1,clusters+1):
        clust_dict[c]=''
        
for i in range(size):
    max_val = np.amax(prob[:,i])
    index = np.where(prob[:,i] == max_val)
    clust = int(index[0])+1
    if clust_dict[clust] == '':
        clust_dict[clust] = str(i+1)
    else:
        clust_dict[clust] = clust_dict[clust]+","+str(i+1) 


# In[ ]:


#Function to assign Cluster IDs to individual rows
def create_label(clust_dict):
    l=1
    label = np.zeros(size)
    for key in clust_dict:
        s=clust_dict[key]
        arr = s.split(',')
        for a in arr:
            label[int(a)-1]=l
        l+=1
    return label


# In[ ]:


#Creating Cluster IDs for each row
cluster_set=create_label(clust_dict)
print(type(cluster_set[0]))


# In[ ]:


#Extracting ground truth from the data set and storing it in an array
ground_set = Data.iloc[:,1:2]
print(type(ground_set))
ground_set = ground_set.to_numpy()
print(type(ground_set))
print(np.shape(ground_set))


# In[ ]:


# Creating Array of Ground Truth Clusters
P = np.zeros([size,size], dtype=int)
for i in range(0,size):
    for j in range(0,size):
        if ground_set[i]==ground_set[j]:
            P[i][j]=1
        else:
            P[i][j]=0


# In[ ]:


# Creating Array of Clusters reported by Clustering Algorithm (Heirarchical)
C = np.zeros([size,size], dtype=int)
for i in range(0,size):
    for j in range(0,size):
        if cluster_set[i]==cluster_set[j]:
            C[i][j]=1
        else:
            C[i][j]=0


# In[ ]:


#Calculating  M00,M01,M10 & M11 which will be used in Rand Index and Jaccard Coeeficient Calculations
M00=0
M01=0
M10=0
M11=0
for i in range(0,size):
    for j in range(0, size):
        if P[i][j]==0 and C[i][j]==0:
            M00+=1
        elif C[i][j]==0 and P[i][j]==1:
            M01+=1
        elif C[i][j]==1 and P[i][j]==0:
            M10+=1
        else:
            M11+=1


# In[ ]:


#Rand Index Calculation
rand = (M11+M00)/(M11+M00+M01+M10)
print("Rand Index:",rand)


# In[ ]:


#Jaccard Coefficient Calculation
jaccard_coeff = (M11)/(M11+M01+M10)
print("Jaccard Coefficient:",jaccard_coeff)


# In[ ]:


#Function to visualize using PCA
def PCA_Plot(points,cluster_set,heading):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(points)
    #print(principalComponents)
    target = np.unique(cluster_set)
    target=target.astype(np.int64)
    #print(target[1])
    color = []
    n = clusters
    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))
    #print(color)
    pl.figure(figsize=(12, 7))
    for t in target:
        #x contains all the x values whose target =t
        x=[]
        y=[]
        for i in range(0,len(principalComponents)):
            if cluster_set[i]==t:
                x.append(principalComponents[i,0])
                y.append(principalComponents[i,1])
        pl.scatter(x,y,cmap=pl.get_cmap('Spectral'))
    pl.legend(target)
    pl.xlabel('Principal Component 1', fontsize = 15)
    pl.ylabel('Principal Component 2', fontsize = 15)
    pl.title(heading, fontsize = 20)
    pl.show()


# In[ ]:


#Plotting Plots for clusters calcualted using HAC Algorithm and Ground Truth Plot
PCA_Plot(points, cluster_set, 'GMM PCA plot')
PCA_Plot(points, ground_set, 'Ground Truth PCA plot')


# In[ ]:




