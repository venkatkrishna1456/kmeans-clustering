import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import random

# initilize 2d array
X = np.array([[2, 4],
                [1.7, 2.8],
                [7, 8],
                [8.6, 8],
                [3.4, 1.5],
                [9, 11]])

# euclidean distance for each points and center point
def euclidean_distance(x, centroid):
    distance = np.zeros((len(x), len(centroid)))
    for i in range(len(x)):
        for j in range(len(centroid)):
            distance[i,j] = np.sum((x[i] - centroids[j]) ** 2) ** 0.5
    return distance

# number of clusters is 2
num_clusters = 2

# initialize center value as random value.
centroids = np.array([[random.randint(0, 9), random.randint(0, 9)], [random.randint(0, 9), random.randint(0, 9)]])

# calculate the distance from 2 centers
distances = euclidean_distance(X, centroids)
# get index of smallest value for each point with 2 centers
points = np.array([np.argmin(i) for i in distances])

# repeat for 10 times
for i in range(15):
    # update center point
    centroids = []
    for cluster_id in range(num_clusters):
        temp_centroid = X[points==cluster_id].mean(axis=0) 
        centroids.append(temp_centroid)

    centroids = np.vstack(centroids)
     
    # calculate distance and set cluster id
    distances = euclidean_distance(X, centroids)
    points = np.array([np.argmin(i) for i in distances])

# plot clustered points
plt.scatter(X[points==0, 0], X[points==0, 1], color="red", s=150)
plt.scatter(X[points==1, 0], X[points==1, 1], color="yellow", s=150)
plt.scatter(centroids[:, 0], centroids[:, 1], color="black", s=150)
plt.show()