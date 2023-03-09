#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
  - Author: Zhiyu Long
  
  - Submission Date: 03/05/2023
  
  - Purpose: HW2
  
  - Statement of Academic Honesty:
  
  The following code represents my own work. I have neither received nor given 
  inappropriate assistance. I have not copied or modified code from any source 
  other than the course webpage or the course textbook. I recognize that any 
  unauthorized assistance or plagiarism will be handled in accordance with the 
  Northeastern University’s Academic Honesty Policy and the policies of this 
  course. I recognize that my work is based on an assignment created by Khoury 
  College of Computer Sciences at Northeastern University. Any publishing or 
  posting of source code for this project is strictly prohibited unless you have 
  written consent from Khoury College of Computer Sciences at Northeastern University.
'''
#################################################################
# Check running enviroment at first

import pkg_resources
import subprocess
import sys

# check python version
if sys.version_info[0] < 3:
    print("Your python version is ", sys.version)
    print("This script requires Python 3 or later")
    sys.exit(1)

# check if all needed packages are installed
def installIfNotExist(required):
  installed = {pkg.key for pkg in pkg_resources.working_set}
  missing = required - installed

  if missing:
    print(">> Installing the missing packages: ", missing)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + list(missing), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print("\n--Finished checking required packages.--\n")
  else:
    print("\n--Required packages already satisfied.--\n")


required = {"pillow", "numpy", "pandas"}
installIfNotExist(required)

# import all needed packages
from PIL import Image
import numpy as np
import pandas as pd

#################################################################
# K-Means implementation from HW1

def convertToNdarray(data):
    """
    Convert the data type to list
    """
    if isinstance(data, pd.DataFrame): 
        return data.values
    if isinstance(data, list):
        return np.array(data)
    return data


def getDist(x, y):
    """
    Calculate Euclidean distance between two data points.
    distance = √((x1-y1)² + (x2-y2)² + ... + (xn-yn)²
    """
    
    # sum of the squared errors
    sumErr = 0
    for i in range(len(x)):
        sumErr += np.power(x[i] - y[i], 2)
    # return square root of the sum
    return np.square(sumErr)


def getClusterMeans(data, assignation):
    """
    Calculate the mean of the data points for each cluster.
    """
    height, width, channels = data.shape

    new_assignation = assignation

    # convert data to a dataframe
    # using the assigned cluster as the index for each data point
    df = pd.DataFrame(data, index=new_assignation)
    
    # np array records the mean for each cluster
    means = df.groupby(df.index).mean().values
    
    return means


def assignDataPoints(dataPoints, centroids):
    """
    For each point in X, calculate the distance to each centroid.
    Assign each point to the closest centroid
    """
    n = len(dataPoints)
    k = len(centroids)
    
    # assigned cluster for each data point
    assignation = [0 for i in range(n)]
    # distance to centroid for each data point
    allDist = [0 for i in range(n)]
    
    # assign each point to the closest centroid
    for i in range(len(dataPoints)):
        
        # record distances to all centroids for the current data point
        distances = [0 for i in range(k)]
        
        # calculate the distance to each centroid
        for j in range(k):
            dist = getDist(centroids[j], dataPoints[i])
            distances[j] = dist
            
        # the nearest centroid
        closestCluster = np.argmin(distances)
        
        assignation[i] = closestCluster
        allDist[i] = distances[closestCluster]
        
    return np.array(assignation).astype(np.int32), np.array(allDist)


def coordinate_descent_kmeans(X, k, max_iter=100, tol=1e-4):
    """
    Coordinate descent algorithm for k-means clustering.

    Parameters:
        X (array-like): data matrix with shape (n_samples, n_features)
        k (int): number of clusters
        max_iter (int): maximum number of iterations (default: 100)
        tol (float): convergence tolerance (default: 1e-4)

    Returns:
        centroids (array-like): cluster centroids with shape (k, n_features)
        clusters (array-like): cluster assignments for each data point with shape (n_samples,)
    """
    
    # check matrix data type, convert to np array
    data = convertToNdarray(X)
    
    n = len(data)
    
    # select k random data points
    randRowsIdx = np.random.choice(np.arange(0, n-1), k, replace=False)
    # initial the random centroids
    centroids = data[randRowsIdx][:, :]
    
    # assigned cluster for each data point
    clusters = [0 for i in range(n)]
    
    # distance to centroid for each data point
    oldDist = np.zeros(n)
    
    # Re-calculate centroids in each loop
    for i in range(max_iter):
        
        # assign data points to the nearest centroids
        clusters, newDist = assignDataPoints(data, centroids)
        
        # check for empty clusters
        # clusters = checkEmptyClusters(centroids, clusters, k, data)

        # update centroids to the cluster means
        centroids = getClusterMeans(data, clusters)
        
        # check for convergence by comparing current and previous centroids
        if np.allclose(newDist, oldDist, atol=tol):
            break
        else:
            # update distances
            oldDist = newDist
    
    return centroids, clusters

#########################################################
# Apply K-Means to image segmentation

def loadImage(filename):
  # Open image file
  img = Image.open(filename)
  # Convert image to numpy array
  img_array = np.array(img)
  return img_array

def writeImage(img_array, filename):
  # convert the image array back to shape H*W*3
  output_img = img_array[:, :, :3]
  img = Image.fromarray(output_img)
  img.save(filename)

def getStdArrayWith5Features(img_array):
  # Get dimensions of the image
  height, width, channels = img_array.shape
  # new matrix with size H * W * 5
  new_img_array = np.zeros((height, width, channels+2), dtype=int)

  for i in range(height):
    for j in range(width):
      rgb = img_array[i, j]
      features = np.zeros((5,))
      # copy the rgb features
      features[:3] = rgb
      # add x and y coordinates as the additional features
      features[3:] = [i, j]
      # assign the updated features to the new matrix
      new_img_array[i, j] = features
  
  # standardize the values of each feature
  mean = np.mean(new_img_array, axis=0)
  std = np.std(new_img_array, axis=0)
  std[std == 0] = 1e-8  # replace 0 with a small constant to avoid division by zero
  
  std_img_array = (new_img_array - mean / std)
  return std_img_array

def generateSegImg(img_array, centroids, clusters):
  '''
  Use the cluster centers to generate the segmented image by 
  replacing each data point’s color values with the closest center.
  '''
  # Get dimensions
  height, width, channels = img_array.shape
  # new matrix for the segmented image
  seg_img_array = np.zeros((height, width, channels+2), dtype=int)
  # reshape the clusters
  cluster_map = clusters.reshape((height, width))

  for i in range(height):
    for j in range(width):
      hx = np.zeros((5,)) # segmented pixel

      # get the centroid
      k = cluster_map[i, j]
      centroid = centroids[k]
      # replace the color to centroid's color
      hx[:3] = centroid[:3]
      # keep the same x and y coordinates
      hx[3:] = [i, j]
      # assign the new pixel to the segmented matrix
      seg_img_array[i, j] = hx

  return seg_img_array


def checkEmptyClusters(centroids, clusters, k, data):
  emptyClusters = np.where(np.bincount(clusters, minlength=k) == 0)[0]
  if len(emptyClusters) > 0:
    for j in emptyClusters:
      # find the nearest cluster to the centroid of the empty cluster
      distToCentroids = np.linalg.norm(centroids - centroids[j], axis=1)
      nearestCluster = np.argmin(distToCentroids)
      
      # find the point in the nearest cluster with the maximum distance to its centroid
      distances = np.linalg.norm(data - centroids[nearestCluster], axis=1)
      farthestPoint = np.argmax(distances)
      
      # assign the empty cluster to the farthest point
      clusters[farthestPoint] = j
  return clusters


def main(K, inputFile, outputFile):
  # load image, get the 3d np array in size H * W * 3
  print(">>> Loading image from: ", inputFile)
  img_array = loadImage(inputFile)

  print(">>> Applying K-Means ")  
  # standardize and convert img_array with 5 features
  std_img_array = getStdArrayWith5Features(img_array)
  # apply K-Means clustering from HW1
  centroids, clusters = coordinate_descent_kmeans(std_img_array, K)
  
  # generate the segmented image
  print(">>> Generating  segmented image")  
  seg_img_array = generateSegImg(img_array, centroids, clusters)

  # wrtie the segmented image to file
  print(">>> Writing image into: ", outputFile)
  writeImage(seg_img_array, outputFile)


if __name__ == "__main__":
  argv = sys.argv
  
  if len(argv) == 4:
    K = int(argv[1]) # the number of clusters
    inputFile = argv[2]
    outputFile = argv[3]
    main(K, inputFile, outputFile)
  else:
    print("Invalid command: ", argv)
    

