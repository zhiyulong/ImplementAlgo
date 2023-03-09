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
    # convert data to a dataframe
    # using the assigned cluster as the index for each data point
    df = pd.DataFrame(data, index=assignation)
    
    # np array records the mean for each cluster
    means = df.groupby(df.index).mean().values
    for i in range(len(means)):
      cluster = df.groupby(df.index).get_group(i).values
      shortest = np.argmin(np.abs(cluster - means[i]).sum(axis=1))
      means[i] = cluster[shortest]

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
    clusters = np.zeros(n)
    dist = np.zeros(n)
    
    # Re-calculate centroids in each loop
    for i in range(max_iter):
      # assign data points to the nearest centroids
      clusters, newDist = assignDataPoints(data, centroids)

      # check for empty clusters
      clusters, newDist = checkEmptyClusters(data, centroids, clusters, newDist)

      # update centroids to the cluster means
      centroids = getClusterMeans(data, clusters)
      
      # check for convergence by comparing current and previous centroids
      if np.allclose(newDist, dist, atol=tol):
          break
      else:
          # update distances
          dist = newDist
    
    return centroids, clusters

#########################################################
# Apply K-Means to image segmentation

def checkEmptyClusters(data, centroids, clusters, dist):
  emptyClusters = np.where(np.bincount(clusters, minlength=len(centroids)) == 0)[0]
  if len(emptyClusters) > 0:
    for j in emptyClusters:
      # find the nearest cluster to the centroid of the empty cluster
      distToCentroids = np.linalg.norm(centroids - centroids[j], axis=1)
      nearestCluster = np.argmin(distToCentroids)
      
      # find the point in the nearest cluster with the maximum distance to its centroid
      distances = np.linalg.norm(data - centroids[nearestCluster], axis=1)
      farthestPoint = np.argmax(distances)
      
      # assign the empty cluster to the farthest point
      newDist = getDist(centroids[j], data[farthestPoint])
      dist[farthestPoint] = newDist
      clusters[farthestPoint] = j

  return clusters, dist


def loadImage(filename):
  # Open image file
  img = Image.open(filename).convert("RGB")
  # Convert image to numpy array
  img_array = np.array(img)
  return img_array

def writeImage(img_array, filename):
  img_array = img_array
  img = Image.fromarray(img_array, mode="RGB")
  img.save(filename)


def standardizeData(img_array):
  # standardize the values of each feature

  # the last feature is the y coordinate, the std of y coordinate will 
  # always be 0, because all the values of a variable are the same. 
  # to avoid this, flat the matrix.
  std_data = img_array.reshape(-1, img_array.shape[2])

  mean = np.mean(std_data, axis=0)
  std = np.std(std_data, axis=0)

  std_data = (std_data - mean) / std

  return std_data, std, mean

def getArrayWithMoreFeatures(img_array):
  # Get dimensions of the image
  height, width, channels = img_array.shape
  # new matrix with size H * W * (C+2)
  new_img_array = np.zeros((height, width, channels+2), dtype=int)

  for i in range(height):
    for j in range(width):
      rgb = img_array[i, j]
      features = np.zeros((channels+2,))
      # copy the rgb features
      features[:channels] = rgb
      # add x and y coordinates as the additional features
      features[channels:] = [i, j]
      # assign the updated features to the new matrix
      new_img_array[i, j] = features
  
  return new_img_array

def generateSegMatrix(std_matrix, centroids, clusters):
  '''
  Use the cluster centers to generate the segmented image by 
  replacing each data point’s color values with the closest center.
  '''
  # new matrix for the segmented image
  num, features = std_matrix.shape
  seg_matrix = np.zeros((num, features))

  for i in range(num):
      hx = np.zeros((features,)) # segmented pixel

      # get the centroid
      k = clusters[i]
      centroid = centroids[k]
      # replace the color to centroid's color
      hx[:features-2] = centroid[:features-2]
      # keep the same x and y coordinates
      hx[features-2:] = std_matrix[i][features-2:]
      # assign the new pixel to the segmented matrix
      seg_matrix[i] = hx

  return seg_matrix



def main(K, inputFile, outputFile):
  # load image, get the 3d np array in shape (H, W, C)
  print(">>> Loading image from: ", inputFile)
  img_array = loadImage(inputFile)

  print(">>> Applying K-Means ")  
  # standardize and convert img_array with C+2 features
  new_img_array = getArrayWithMoreFeatures(img_array)
  # standardize the values in the 3d array to a 2d matrix with shape (H*W, C+2)
  std_matrix, std, mean = standardizeData(new_img_array)
  std_matrix = std_matrix.astype(float)
  # apply K-Means clustering from HW1
  centroids, clusters = coordinate_descent_kmeans(std_matrix, K, max_iter=3)
  
  # generate the segmented image
  print(">>> Generating  segmented image")
  seg_matrix = generateSegMatrix(std_matrix, centroids, clusters)

  # de-standardize the matrix
  destd_seg_matrix = (seg_matrix * std) + mean
  # reshape matrix to 3d img array
  destd_img = destd_seg_matrix.reshape(new_img_array.shape)
  
  # convert the image array back to shape H*W*C
  output_img = destd_img[:, :, :img_array.shape[2]].astype(np.uint8)

  # wrtie the segmented image to file
  print(">>> Writing image into: ", outputFile)
  writeImage(output_img, outputFile)
  print(">>> Finish running program.")


if __name__ == "__main__":
  argv = sys.argv
  
  if len(argv) == 4:
    K = int(argv[1]) # the number of clusters
    inputFile = argv[2]
    outputFile = argv[3]
    main(K, inputFile, outputFile)
  else:
    print("Invalid command: ", argv)
    

