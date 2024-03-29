# Simplifying Image Compression with K-means Clustering
Introduction:
In today's digital age, images play a central role in our daily lives, from social media to websites and personal collections. However, storing and transmitting large image files can be resource intensive. Image compression helps overcome this challenge by reducing file size while preserving important visual information. In this blog post, we will explore one such technique called K-means clustering, which is commonly used for image stacking.
Understanding K-means Clustering:
K-means clustering is a popular algorithm used in unsupervised machine learning. It is mostly used for data clusters, where similar data points are grouped together. The algorithm works iteratively to find the focal points that represent the center of each cluster. These focal points are then used to provide data points to their neighboring units.
Image Compression with K-means:
Imagine a colorful picture with thousands of different colors. By using K-means clustering, we can reduce the number of colors in an image and preserve all its visual properties. This process involves selecting representative subsets of colors (centroids) and then replacing each pixel with the nearest point in the image.
Implementation Steps:
Finding closest centroids
def find_closest_centroids(X, centroids):

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
          dist = [] 
          for j in range(centroids.shape[0]):
              ij = np.linalg.norm(X[i] - centroids[j])
              dist.append(ij)

          idx[i] = np.argmin(dist)       
                      
    
    return idx

# Load data stored in arrays X, y from data folder (ex7data2.mat)
import os
from os.path import dirname, join as pjoin
import scipy.io as sio
data = sio.loadmat(os.path.join('data', 'ex7data2.mat'))
X = data['X']

print("First five elements of X are:\n", X[:5]) 
print('The shape of X is:', X.shape)

# Select an initial set of centroids (3 Centroids)
initial_centroids = np.array([[3,3], [6,2], [8,5]])

# Find closest centroids using initial_centroids
idx = find_closest_centroids(X, initial_centroids)

# Print closest centroids for the first three elements
print("First three elements in idx are:", idx[:3])

# UNIT TEST
from public_tests import *

find_closest_centroids_test(find_closest_centroids)
Computing centroid means
def compute_centroids(X, idx, K):
    
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    

    for i in range(K):   
          points = X[idx == i]  
          centroids[i] = np.mean(points, axis = 0)
    
    return centroids

K = 3
centroids = compute_centroids(X, idx, K)

print("The centroids are:", centroids)

# UNIT TEST
compute_centroids_test(compute_centroids)
Random initialization
def kMeans_init_centroids(X, K):
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids

# Set number of centroids and max number of iterations
K = 3
max_iters = 10

# Set initial centroids by picking random examples from the dataset
initial_centroids = kMeans_init_centroids(X, K)

# Run K-Means
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)
K-Means on image pixels
def kMeans_init_centroids(X, K):
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids

# Set number of centroids and max number of iterations
K = 3
max_iters = 10

# Set initial centroids by picking random examples from the dataset
initial_centroids = kMeans_init_centroids(X, K)

# Run K-Means
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)

# Preprocess the image
original_img = plt.imread('bird_small.png')
X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

# Select initial centroids
initial_centroids = kMeans_init_centroids(X_img, K)

# Run K-means algorithm
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)

# Plot the colors of the image and mark the centroids
plot_kMeans_RGB(X_img, centroids, idx, K)
Compress the image
# Assign pixels to centroids
idx = find_closest_centroids(X_img, centroids)

# Replace pixels with centroids
X_recovered = centroids[idx, :]

# Reconstruct the compressed image
X_recovered = np.reshape(X_recovered, original_img.shape)

# Display original image
fig, ax = plt.subplots(1,2, figsize=(16,16))
plt.axis('off')

ax[0].imshow(original_img)
ax[0].set_title('Original')
ax[0].set_axis_off()

# Display compressed image
ax[1].imshow(X_recovered)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()
Benefits of Image Compression:
Image compression offers many benefits, such as reduced file size, faster load time, better storage, and improved user experience With compression of images, we can significantly reducing the amount of storage space and bandwidth required for delivery. Smaller image files load faster across websites and digital platforms, giving users a smoother browsing experience. Additionally, compressed images consume less resources, making them ideal for large image backups or archives.
Conclusion:
In this blog post, we explored how to use K-means clustering to efficiently classify images. By reducing the number of colors in an image and preserving its visual fidelity, we can save significant amounts of money in storage space and bandwidth. Image embedding plays an important role in a variety of applications, from web development to digital photography, enhancing the user experience and optimizing product usage.
