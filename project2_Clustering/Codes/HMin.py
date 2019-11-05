#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import itertools
from sklearn.decomposition import PCA
import matplotlib.pyplot as pl
import numpy as np
from random import randint


# In[2]:


# User Input for Dataset & Number of Clusters
file_name = input("Enter file name: ")
clusters = int(input("Enter Number of Clusters:"))


# In[3]:


# Reading file into Dataframe
Data = pd.read_csv(file_name, sep='\t', lineterminator='\n', header=None)


# In[4]:


#Data.head


# In[5]:


#Extracting attributes and storing it
points = Data.iloc[:,2:]


# In[6]:


#points.head()


# In[7]:


#Creating array for all attributes 
points_arr=points.iloc[:,:].values


# In[8]:


#points_arr


# In[9]:


size = len(points.index)
#print(size)


# In[10]:


#Initializing Distance Matrix to store Euclidian Distance for each individual point
dist_matrix = np.zeros([size,size], dtype=float)


# In[11]:


#len(points_arr)


# In[12]:


#Function to calculate Euclidian Distance between 2 Points
def eucl_dist(p1, p2):
    return np.linalg.norm(p1-p2)


# In[13]:


#Storing Euclidian Distances calculated in Distance Matrix
for i in range(0,len(points_arr)):
    for j in range(i+1,len(points_arr)):
        d = eucl_dist(points_arr[i],points_arr[j])
        dist_matrix[i][j]=d
        dist_matrix[j][i]=d
print(dist_matrix)


# In[14]:


#Creating Dictionary to store initial clusters ( 386 points assigned to 386 clusters ranging from 1 to 386)
clust_dict = {}
for i in range (0,size):
    clust_dict[i+1]=str(i+1)


# In[15]:


#Function to return the index of the point which has the least distance stored in the distance matrix
def min_value(dist_matrix):
    #Retreive minimum value
    minval = np.min(dist_matrix[np.nonzero(dist_matrix)])
    #print(minval)
    #Retreive Index of Minimum value
    a=np.where(dist_matrix == minval)
    #[80 87 88 55]
    #[87 80 88 55]
    p1=a[1][0]
    p2=a[1][1]
    if p1==p2:
        p1=a[1][2]
        p2=a[1][3]
    #p1 = a[-2]
    #p2 = a[-1]
    return p1,p2


# In[16]:


ini_clusters = size
#clusters=5 #No. of clusters
#print(clusters,size)
#Run a loop till the number of clusters is not equal to users' input for number of clusters
while(ini_clusters!=clusters):
    print("No. of clusters:",ini_clusters)
    p1,p2=min_value(dist_matrix)
    print(p1,p2)
    #Updating Distance Matrix after merging points into a cluster (Points 3 and 6 merged to 3,6 by modifying point 3 and setting 0 for all values of point 6)
    for i in range(0,len(dist_matrix)):
        dist_matrix[i][p1]=min(dist_matrix[i][p1], dist_matrix[i][p2])
        dist_matrix[p1][i]=min(dist_matrix[p1][i], dist_matrix[p2][i])
        dist_matrix[i][p2]=0
        dist_matrix[p2][i]=0
    #Updating the cluster dictionary with merged points (3 and 6 merged to 3,6)   
    clust_dict[p1+1]=clust_dict[p1+1]+','+clust_dict[p2+1]
    #Deleting point that is merged(Point 6 in this case is deleted as its merged into 3,6)
    del clust_dict[p2+1]
    #print("val",clust_dict[p2+1])
    #print("len",len(clust_dict))
    #c=0
    #Reading Cluster Dictionary and printing all the values 
    #for key in clust_dict:
     #   s=clust_dict[key]
     #   a=s.split(',')
    #    c+=len(a)
    #print("No of vals:", c)
    ini_clusters=len(clust_dict)
   # print(ini_clusters)
        


# In[17]:


print(dist_matrix)


# In[18]:


#print(len(clust_dict))
#Printing all the clusters and its members
c=1
for key in clust_dict:
    print("Cluster",c)
    print("[",clust_dict[key],"]")
    c+=1
print('\n')
#print(clust_dict)


# In[19]:


#idx = np.argwhere(np.all(dist_matrix[..., :] == 0, axis=0))
#a2 = np.delete(dist_matrix, idx, axis=1)
#idy = np.argwhere(np.all(a2[..., :] == 0, axis=1))
#a3 = np.delete(a2, idy, axis=0)


# In[20]:


#print(a3)


# In[21]:


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


# In[22]:


#Creating Cluster IDs for each row
cluster_set=create_label(clust_dict)
#print(type(cluster_set[0]))


# In[23]:


#Extracting ground truth from the data set and storing it in an array
ground_set = Data.iloc[:,1:2]
#print(type(ground_set))
ground_set = ground_set.to_numpy()
#print(type(ground_set))


# In[24]:


# Creating Array of Ground Truth Clusters
P = np.zeros([size,size], dtype=int)
for i in range(0,size):
    for j in range(0,size):
        if ground_set[i]==ground_set[j]:
            P[i][j]=1
        else:
            P[i][j]=0


# In[25]:


# Creating Array of Clusters reported by Clustering Algorithm (Heirarchical)
C = np.zeros([size,size], dtype=int)
for i in range(0,size):
    for j in range(0,size):
        if cluster_set[i]==cluster_set[j]:
            C[i][j]=1
        else:
            C[i][j]=0


# In[26]:


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


# In[27]:


#Rand Index Calculation
rand = (M11+M00)/(M11+M00+M01+M10)
print("Rand Index:",rand)


# In[28]:


#Jaccard Coefficient Calculation
jaccard_coeff = (M11)/(M11+M01+M10)
print("Jaccard Coefficient:",jaccard_coeff)


# In[29]:


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


# In[30]:


#Plotting Plots for clusters calcualted using HAC Algorithm and Ground Truth Plot
PCA_Plot(points, cluster_set, 'HAC PCA plot')
PCA_Plot(points, ground_set, 'Ground Truth PCA plot')


# In[ ]:




