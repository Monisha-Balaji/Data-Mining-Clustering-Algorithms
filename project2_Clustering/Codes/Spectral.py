#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#import dataframes as df
import sys
import matplotlib.pyplot as plt
import pylab as pl
np.set_printoptions(threshold=sys.maxsize)
import statistics
from random import randint
from sklearn.decomposition import PCA


# In[6]:


file_name = input("Enter file name: ")

df = pd.read_csv(file_name, delimiter="\t",header=None)
df.to_csv('a.csv', encoding='utf-8', index=False)
df1=df.iloc[:,2:]
print(df1.head(5))

row,col=df1.shape
print("row:",row)
print("col",col)


# In[9]:


# Compute nxn adjacency (similarity) matrix
sig=int(input("Enter sigma value: "))
A = np.zeros((row,row))
for i in range(row):
    for j in range(row):
        if i == j:
            A[i][j] = 0.0
        else:
            A[i][j] =np.exp(np.linalg.norm((df1.iloc[i,:]-df1.iloc[j,:])/sig**2))            


# In[ ]:


D=np.zeros((row,row))
for i in range(row):
    for j in range(row):
        if(i==j):
            D[i][j]=sum(A[j])
#D


# In[ ]:


L = D - A
B = np.linalg.inv(D) @ L
# Compute eigenvalues and eigenvectors of B
w, v = np.linalg.eig(B)


# In[ ]:


# Get reduced basis 
k = int(input("Enter Number of Clusters:"))
v = v[:,:k]


# In[ ]:


Y = np.zeros((row,k))
for i in range(row):
    Y[i,:] = v[i,:] * (1/ (sum(v[i,:] ** 2) ** (1/2)))


# In[ ]:


# Save the file in text format
np.savetxt("Y.txt", Y, delimiter=",")


# In[ ]:


# User Input for Dataset & Number of Clusters
file_name = pd.read_csv("Y.txt", delimiter="\t",header=None)
clusters = k
iterations= int(input("Enter Number of Iterations"))
choice = int(input("Enter 1 to randlomly assign initial clusters or 2 to manually assign initial clusters"))
if (choice==2):
    ini_clusters= input("Enter IDs for initial clusters")
    a=ini_clusters.split(" ")
    if (len(a)!=clusters):
        #the format to enter IDs would be like: 3 1 2 4
        print("Invalid Input")


# In[ ]:


# Reading file into Dataframe
Data = pd.read_csv("Y.txt", sep=',', lineterminator='\n', header=None)


# In[ ]:


#Extracting attributes and storing it
points = Data.iloc[:,:]


# In[ ]:


#Creating array for all attributes 
points_arr=points.iloc[:,:].values


# In[ ]:


size = len(points.index)
print(size)


# In[ ]:


cols = len(points_arr[0])
#print(cols)


# In[ ]:


#Function to calculate Euclidian Distance between 2 Points
def eucl_dist(p1, p2):
    return np.linalg.norm(p1-p2)


# In[ ]:


if(choice==1):
    idx = np.random.randint(size, size=clusters)
    print(idx)
    print(points_arr[idx,:])
elif(choice == 2):
    idx=a


# In[ ]:


#Creating Dictionary to store initial clusters ( 386 points assigned to 386 clusters ranging from 1 to 386)
clust_dict = {}
c=1
for i in idx:
    clust_dict[c]=i
    c+=1
print(clust_dict)


# In[ ]:


centroids = np.zeros([clusters,cols], dtype = float)
i=0
for key in clust_dict:
    centroids[i]=points_arr[int(clust_dict[key])-1]
    i+=1
print(centroids)


# In[ ]:


def distance_matrix(arr,centroids):
    #Initializing Distance Matrix to store Euclidian Distance for each individual point
    dist_matrix = np.zeros([size,clusters], dtype=float)
    for i in range(0,size):
        for j in range(0,clusters):
            #print("dist bet:", centroids[j])
            dist_matrix[i][j]=eucl_dist(arr[i],centroids[j])
    return dist_matrix


# In[ ]:


def find_clusters(dist_matrix):
    clust_dict={}
    for c in range(1,clusters+1):
        clust_dict[c]=''
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
    return clust_dict


# In[ ]:


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
    


# In[ ]:


def check_centroid(arr1,arr2):
    return np.array_equal(arr1,arr2)


# In[ ]:


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
#ground_set = Data.iloc[:,1:2]
ground_set = df.iloc[:,1:2]
print(type(ground_set))
ground_set = ground_set.to_numpy()
print(type(ground_set))


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
PCA_Plot(points, cluster_set, 'Spectral PCA plot')
PCA_Plot(points, ground_set, 'Ground Truth PCA plot')

