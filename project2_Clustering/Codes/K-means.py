#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd
import itertools
from sklearn.decomposition import PCA
import matplotlib.pyplot as pl
import numpy as np
from random import randint


# In[77]:


# User Input for Dataset & Number of Clusters
file_name = input("Enter file name: ")
clusters = int(input("Enter Number of Clusters:"))
iterations= int(input("Enter Number of Iterations"))
choice = int(input("Enter 1 to randlomly assign initial clusters or 2 to manually assign initial clusters"))
if (choice==2):
    ini_clusters= input("Enter IDs for initial clusters")
    a=ini_clusters.split(" ")
    if (len(a)!=clusters):
        print("Invalid Input")


# In[78]:


# Reading file into Dataframe
Data = pd.read_csv(file_name, sep='\t', lineterminator='\n', header=None)


# In[79]:


#Extracting attributes and storing it
points = Data.iloc[:,2:]


# In[80]:


#Creating array for all attributes 
points_arr=points.iloc[:,:].values


# In[81]:


size = len(points.index)
#print(size)


# In[82]:


cols = len(points_arr[0])
#print(cols)


# In[83]:


#Function to calculate Euclidian Distance between 2 Points
def eucl_dist(p1, p2):
    return np.linalg.norm(p1-p2)


# In[84]:


if(choice==1):
    idx = np.random.randint(size, size=clusters)
    print(idx)
    print(points_arr[idx,:])
elif(choice == 2):
    idx=a


# In[85]:


#Creating Dictionary to store initial clusters ( 386 points assigned to 386 clusters ranging from 1 to 386)
clust_dict = {}
c=1
for i in idx:
    clust_dict[c]=i
    c+=1
print(clust_dict)


# In[86]:


centroids = np.zeros([clusters,cols], dtype = float)
i=0
for key in clust_dict:
    centroids[i]=points_arr[int(clust_dict[key])-1]
    i+=1
print(centroids)


# In[87]:


def distance_matrix(arr,centroids):
    #Initializing Distance Matrix to store Euclidian Distance for each individual point
    dist_matrix = np.zeros([size,clusters], dtype=float)
    for i in range(0,size):
        for j in range(0,clusters):
            #print("dist bet:", centroids[j])
            dist_matrix[i][j]=eucl_dist(arr[i],centroids[j])
    return dist_matrix


# In[93]:


def find_clusters(dist_matrix):
    clust_dict={}
    for c in range(1,clusters+1):
        clust_dict[c]=''
    print(clust_dict)
    for i in range(0,size):
        min_value=np.min(dist_matrix[i])
        #print(min_value)
        index=np.where(dist_matrix[i] == min_value)
        #print(index,dist_matrix[i])
        #if min_value!=0:
        clust = int(index[0])+1
        if clust_dict[clust] == '':
            clust_dict[clust] = str(i+1)
        else:
            clust_dict[clust] = clust_dict[clust]+","+str(i+1)
    print(clust_dict)
    return clust_dict


# In[94]:


def calculate_centroid(arr,clust_dict):
    centroids = np.zeros([clusters,cols], dtype = float)
    i=0
    for key in clust_dict:
        clust_vals = clust_dict[key]
        vals = clust_vals.split(",")
        temp = np.zeros([len(vals),cols], dtype = float)
        j=0
        for v in vals:
            temp[j]=arr[int(v)-1]
            j+=1
        centroids[i]=np.mean(temp, axis = 0)
        i+=1
    return centroids
    


# In[95]:


def check_centroid(arr1,arr2):
    return np.array_equal(arr1,arr2)


# In[96]:


no_iter=1
while(no_iter<=iterations):
    print("Iter:", no_iter)
    dist_matrix = distance_matrix(points_arr,centroids)
    clust_dict = find_clusters(dist_matrix)
    #print(centroids)
    c = calculate_centroid(points_arr,clust_dict)
    if(check_centroid(c,centroids)):
        print("Exiting as there is no change in centroids from previous iteration")
        break
    else:
        centroids = c
    no_iter+=1
print(clust_dict)


# In[28]:


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


# In[29]:


#Creating Cluster IDs for each row
cluster_set=create_label(clust_dict)
#print(type(cluster_set[0]))


# In[30]:


#Extracting ground truth from the data set and storing it in an array
ground_set = Data.iloc[:,1:2]
#print(type(ground_set))
ground_set = ground_set.to_numpy()
#print(type(ground_set))


# In[31]:


# Creating Array of Ground Truth Clusters
P = np.zeros([size,size], dtype=int)
for i in range(0,size):
    for j in range(0,size):
        if ground_set[i]==ground_set[j]:
            P[i][j]=1
        else:
            P[i][j]=0


# In[32]:


# Creating Array of Clusters reported by Clustering Algorithm (Heirarchical)
C = np.zeros([size,size], dtype=int)
for i in range(0,size):
    for j in range(0,size):
        if cluster_set[i]==cluster_set[j]:
            C[i][j]=1
        else:
            C[i][j]=0


# In[33]:


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


# In[34]:


#Rand Index Calculation
rand = (M11+M00)/(M11+M00+M01+M10)
print("Rand Index:",rand)


# In[35]:


#Jaccard Coefficient Calculation
jaccard_coeff = (M11)/(M11+M01+M10)
print("Jaccard Coefficient:",jaccard_coeff)


# In[36]:


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


# In[37]:


#Plotting Plots for clusters calcualted using HAC Algorithm and Ground Truth Plot
PCA_Plot(points, cluster_set, 'K-means PCA plot')
PCA_Plot(points, ground_set, 'Ground Truth PCA plot')


# In[ ]:




