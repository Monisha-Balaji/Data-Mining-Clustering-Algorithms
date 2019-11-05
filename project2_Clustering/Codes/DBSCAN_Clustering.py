#!/usr/bin/env python
# coding: utf-8

# In[168]:


import pandas as pd
import itertools
from sklearn.decomposition import PCA
import matplotlib.pyplot as pl
import numpy as np
from random import randint


# In[169]:


file_name = input("Enter file name: ")
eps = float(input("Enter E-Neighborhood distance:"))
minpoints= int(input("Enter Number of MinPoints:"))


# In[170]:


# Reading file into Dataframe
Data = pd.read_csv(file_name, sep='\t', lineterminator='\n', header=None)
#Data


# In[171]:


#Extracting attributes and storing it
points = Data.iloc[:,2:]
#points


# In[172]:


#Creating array for all attributes 
points_arr=points.iloc[:,:].values
#points_arr


# In[173]:


size = len(points.index)
#print(size)


# In[174]:


cols = len(points_arr[0])
#print(cols)


# In[175]:


#Function to calculate Euclidian Distance between 2 Points
def eucl_dist(p1, p2):
    return np.linalg.norm(p1-p2)


# In[176]:


def distance_matrix(arr):
    #Initializing Distance Matrix to store Euclidian Distance for each individual point
    dist_matrix = np.zeros([size,size], dtype=float)
    for i in range(0,size):
        for j in range(0,size):
            #print("dist bet:", centroids[j])
            dist_matrix[i][j]=eucl_dist(arr[i],arr[j])
    return dist_matrix


# In[177]:


def regionquery(eps, dis):
    np = []
    np_index = []
    for i in range(len(dis)):
        if dis[i]<eps:
            np.append(points_arr[i])
            np_index.append(i)
    return np, np_index


# In[178]:


clus = {}


# In[179]:


def DBSCAN(points_arr, eps, MinPts):
    count=1
    visited = np.zeros(size, dtype=float)
    point_type = np.zeros(size, dtype=int)
    cluster_no = np.zeros(size, dtype=int)
    
    store = []
    neighborpts = []
    neighpts_index = []
    dist_matrix = distance_matrix(points_arr)
    for i in range(0,size):
        if visited[i]==0:
            visited[i]=1
            neighborpts, neighpts_index = regionquery(eps, dist_matrix[i])
            #print("neighboring points for:",i,"is--")
            #print(neighborpts)
            #print("indexes for",i,"----",neighpts_index)
            if len(neighborpts)<MinPts:
                #point_type[i] = -1
                cluster_no[i] = -1
            else:
                #print(i," has more than Minpts:",str(len(neighborpts)))
                clus[count]= str(i)
                cluster_no[i] = count
                expandcluster(eps, MinPts, neighborpts, neighpts_index, count,dist_matrix,visited,cluster_no,clus)
                store.append(neighborpts[:])
                count= count + 1
                #print("CLUSTER---",clus)
    return cluster_no
        
    
    


# In[180]:


def expandcluster(eps, MinPts, neighborpts, neighpts_index, count,dist_matrix,visited,cluster_no,clus):
    neighbors = []
    neighbors_in = []
    if len(neighpts_index)!= 0:
        for i in neighpts_index:
            if visited[i]==0:
                visited[i]=1
                neighbors, neighbors_in = regionquery(eps, dist_matrix[i])
                #print("For expand-index ",i," neighbors are---",neighbors_in)
                if len(neighbors)>=MinPts:
                    for j in neighbors_in:
                        #if j not in neighborpts:
                        #neighborpts.append(j)
                        neighpts_index.append(j)
                    #print("final neighbors--------",neighborpts)
            if cluster_no[i] == 0 or cluster_no[i]==-1:
                cluster_no[i]= count
                clus[count] = str(clus[count])+","+str(i)
    return


# In[181]:


result =DBSCAN(points_arr,eps,minpoints)


# In[182]:


print(result)


# In[192]:


#Assigning Cluster IDs to individual points(386)
list_set = set(result)
unique_list = (list(list_set))
clust_dict={}
clusters=len(unique_list)
print(clusters)
for i in unique_list:
    clust_dict[i]=""
for i in range(0,len(result)):
    x=result[i]
    if clust_dict[x] == '':
        clust_dict[x] = str(i+1)
    else:
        clust_dict[x] = clust_dict[x]+","+str(i+1) 
print(clust_dict)
print(clusters)


# In[184]:


#Extracting ground truth from the data set and storing it in an array
ground_set = Data.iloc[:,1:2]
#print(type(ground_set))
ground_set = ground_set.to_numpy()
#print(type(ground_set))


# In[185]:


# Creating Array of Ground Truth Clusters
P = np.zeros([size,size], dtype=int)
for i in range(0,size):
    for j in range(0,size):
        if ground_set[i]==ground_set[j]:
            P[i][j]=1
        else:
            P[i][j]=0


# In[186]:


# Creating Array of Clusters reported by Clustering Algorithm (Heirarchical)
C = np.zeros([size,size], dtype=int)
for i in range(0,size):
    for j in range(0,size):
        if result[i]==result[j]:
            C[i][j]=1
        else:
            C[i][j]=0


# In[187]:


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


# In[188]:


#Rand Index Calculation
rand = (M11+M00)/(M11+M00+M01+M10)
print("Rand Index:",rand)


# In[189]:


#Jaccard Coefficient Calculation
jaccard_coeff = (M11)/(M11+M01+M10)
print("Jaccard Coefficient:",jaccard_coeff)


# In[190]:


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


# In[191]:


#Plotting Plots for clusters calcualted using HAC Algorithm and Ground Truth Plot
PCA_Plot(points, result, 'DBSCAN PCA plot')
PCA_Plot(points, ground_set, 'Ground Truth PCA plot')


# In[ ]:




