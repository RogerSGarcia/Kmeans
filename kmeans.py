# https://github.com/stuntgoat/kmeans
# https://github.com/mubaris/friendly-fortnight

from collections import defaultdict
from random import uniform
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

def get_key_from_value(dictionary, value):
    """
    Get the key given a value
    dictionary : dictionary to search
    value : value to match

    Returns
    k : the key associated with the value, else None
    """    
    for k,v in dictionary.items():
        if v == value:
            return k
    return None

def euclidean_distance(x,y):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

def distance(a, b):
    """
    """
    dimensions = len(a)
    
    _sum = 0
    for dimension in range(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return sqrt(_sum)

def point_avg(points):
    """
    """
    dimensions = len(points[0])

    new_center = []

    for dimension in range(dimensions):
        dim_sum = 0  # dimension sum
        for p in points:
            dim_sum += p[dimension]

        # average of each dimension
        new_center.append(dim_sum / float(len(points)))

    return new_center

def update_centers(data_set, assignments):
    """
    """
    new_means = {}
    centers = []
    
    print("updating centers...")
    for assignment, point in zip(assignments, data_set):
        if assignment not in new_means:
            new_means[assignment] = []
        new_means[assignment].append(point)
        
    for center, points in sorted(new_means.items()):
        centers.append(point_avg(points))

    return centers


def assign_points(data_points, centers):
    """
    """
    assignments = []
    for point in data_points:

        cluster_dist = {}
        for i in range(len(centers)):
            cluster_dist[i] = euclidean_distance(point, centers[i])

        cluster_shortest_dist = min(cluster_dist.values())
        cluster_index = get_key_from_value(cluster_dist, cluster_shortest_dist)

        assignments.append(cluster_index)

    return assignments

def k_means(dataset, initial_centriods):
    
    assignments = assign_points(dataset, initial_centriods)
    old_assignments = None
    assignments_count = 0
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
        assignments_count += 1

        plot_assignments(new_centers, list(zip(assignments, dataset)), assignments_count)

    return list(zip(assignments, dataset)), new_centers

def plot_assignments(new_centers, results, assignments_count):
    centers = np.array(new_centers)
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    fig, ax = plt.subplots()

    print("number of centers", len(centers))
    for i in range(len(centers)):
            points = np.array([X[j] for j in range(len(X)) if results[j][0] == i])
            print("shape of points", points.shape)
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i], label = "Cluster " + str(i + 1))
    ax.scatter(centers[:, 0], centers[:, 1], marker='o', s=200, c='#050505')
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    plt.title("K means clustering ( iter = " + str(assignments_count) +  ")" )
    plt.legend(loc='upper right', prop={"size":10}, markerscale = 4)
    # ax.legend(markerscale=5)
    save_path = os.path.join(path_to_save, "iter" + str(assignments_count))
    # plt.savefig(save_path)
    # plt.show()    
    plt.close()

#Ensure directory exist to save plots 
path_to_save = os.path.join(os.getcwd(), "saved_plots")
if not os.path.exists(path_to_save): os.makedirs(path_to_save)

# Importing the dataset
data = pd.read_csv('data/cluster_demo.csv')
print("Input Data and Shape")
print(data.shape)
print(data.head())
print(data.tail())

# Getting the values and plotting it
f1 = data['x'].values
f2 = data['y'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)
# plt.savefig(os.path.join(path_to_save, "iter0"))
# plt.show()

# Number of clusters
k = 3

#format dataset
dataset = list(zip(f1, f2))
ic_x = np.random.randint(np.min(dataset), np.max(dataset), size=k)
ic_y = np.random.randint(np.min(dataset), np.max(dataset), size=k)
initial_centriods = np.array(list(zip(ic_x, ic_y)), dtype=np.float32)
print("Initial Centroids")
print(initial_centriods)
plt.scatter(f1, f2, c='black', s=7)
plt.scatter(initial_centriods[:,0], initial_centriods[:, 1], marker='o', s=200, c='#050505')
# plt.savefig(os.path.join(path_to_save, "initial"))

#call function for kmeans

# results = list(k_means(points, 3))
results, centers = k_means(dataset, initial_centriods.tolist())

plt.show()

print("results...")
# print(results)

# centers = np.array(centers)
# colors = ['r', 'g', 'b', 'y', 'c', 'm']
# fig, ax = plt.subplots()
# for i in range(3):
#         points = np.array([X[j] for j in range(len(X)) if results[j][0] == i])
#         ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
# ax.scatter(centers[:, 0], centers[:, 1], marker='*', s=200, c='#050505')
# plt.show()