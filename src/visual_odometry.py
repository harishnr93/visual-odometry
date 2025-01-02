"""
Date: 23.Nov.2024
Author: Harish Natarajan Ravi
Email: harrish.nr@gmail.com
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from m2bk import *
import features_algo

np.random.seed(1)
#np.set_printoptions(threshold=np.nan)

dataset_handler = DatasetHandler()

# accessing grayscale image
image = dataset_handler.images[0]

plt.figure(figsize=(8, 6), dpi=100)
plt.title("Grayscale Image")
plt.imshow(image, cmap='gray')

# accessing colour image
image_rgb = dataset_handler.images_rgb[0]

plt.figure(figsize=(8, 6), dpi=100)
plt.title("Colour Image")
plt.imshow(image_rgb)


i = 0
# accessing depth map
depth = dataset_handler.depth_maps[i]

plt.figure(figsize=(8, 6), dpi=100)
plt.title("Depth Map")
plt.imshow(depth, cmap='jet')

print("Depth map shape: {0}".format(depth.shape))

# depth Value
v, u = depth.shape
depth_val = depth[v-1, u-1]
print("Depth value of the very bottom-right pixel of depth map {0} is {1:0.3f}".format(i, depth_val))

# K matrix
k_matrix = dataset_handler.k

print("K - matrix:") 
print(k_matrix)

# Number of frames in the dataset
print("Number of frames in the dataset: {0}".format(dataset_handler.num_frames))

i = 30
image = dataset_handler.images[i]

plt.figure(figsize=(8, 6), dpi=100)
plt.title("Gray i = 30")
plt.imshow(image, cmap='gray')

i = 0
# Feature Extraction from one image
image = dataset_handler.images[i]
kp, des = features_algo.extract_features(image)
print("******** ORB *********")
print("Number of features detected in frame {0}: {1}".format(i, len(kp)))
print("Coordinates of the first keypoint in frame {0}: {1}".format(i, str(kp[0].pt)))

# Features Visualization 
i = 0
image = dataset_handler.images_rgb[i]
features_algo.visualize_features(image, kp)
#plt.show()

# Feature Extraction from image dataset
images = dataset_handler.images
kp_list, des_list = features_algo.extract_features_dataset(images, features_algo.extract_features)

i = 0
print("Number of features detected in frame {0}: {1}".format(i, len(kp_list[i])))
print("Coordinates of the first keypoint in frame {0}: {1}\n".format(i, str(kp_list[i][0].pt)))

# Note- The length of the returned by dataset_handler lists should be the same as the length of the image array
print("Length of images array: {0}".format(len(images)))

# Feature Matching
i = 0 
des1 = des_list[i]
des2 = des_list[i+1]

match = features_algo.match_features(des1, des2)
print("Number of features matched in frames {0} and {1}: {2}".format(i, i+1, len(match)))

#i = 0 
#des1 = des_list[i]
#des2 = des_list[i+1]
#match = features_algo.match_features(des1, des2)

# Filtered feature matching
dist_threshold = 0.6
filtered_match = features_algo.filter_matches_distance(match, dist_threshold)

print("Number of features matched in frames {0} and {1} after filtering by distance: {2}".format(i, i+1, len(filtered_match)))

# Visualize n first matches, set n to None to view all matches
# set filtering to True if using match filtering, otherwise set to False
n = 25
filtering = True

i = 30 
image1 = dataset_handler.images[i]
image2 = dataset_handler.images[i+1]

kp1 = kp_list[i]
kp2 = kp_list[i+1]

des1 = des_list[i]
des2 = des_list[i+1]

match = features_algo.match_features(des1, des2)

if filtering:
    dist_threshold = 0.6
    match = features_algo.filter_matches_distance(match, dist_threshold)

match = [m[0] for m in match]

image_matches = features_algo.visualize_matches(image1,kp1,image2,kp2,match[:n])  

# Matching Features in Each Subsequent Image Pair in the Dataset
matches = features_algo.match_features_dataset(des_list, features_algo.match_features)

i = 0
print("Number of features matched in frames {0} and {1}: {2}".format(i, i+1, len(matches[i])))

# Match filtering by thresholding the distance for each subsequent image pair in the dataset
dist_threshold = 0.6

filtered_matches = features_algo.filter_matches_dataset(features_algo.filter_matches_distance, matches, dist_threshold)

if len(filtered_matches) > 0:
    
    # This variable is set to True if you want to use filtered matches further
    is_main_filtered_m = False
    if is_main_filtered_m: 
        matches = filtered_matches

    i = 0
    print("Number of filtered matches in frames {0} and {1}: {2}".format(i, i+1, len(filtered_matches[i])))


# Trajectory Estimation

# Estimating Camera Motion between a Pair of Images

i = 0
match = matches[i]
kp1 = kp_list[i]
kp2 = kp_list[i+1]
k = dataset_handler.k
depth = dataset_handler.depth_maps[i]

rmat, tvec, image1_points, image2_points = features_algo.estimate_motion(match, kp1, kp2, k, depth1=depth)

print("Estimated rotation:\n {0}".format(rmat))
print("Estimated translation:\n {0}".format(tvec))

# Camera Movement Visualization

i=30
image1  = dataset_handler.images_rgb[i]
image2 = dataset_handler.images_rgb[i + 1]

image_move = visualize_camera_movement(image1, image1_points, image2, image2_points)
plt.figure(figsize=(16, 12), dpi=100)
plt.title("Camera movement visualization - (a)")
plt.imshow(image_move)


image_move = visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=True)
plt.figure(figsize=(16, 12), dpi=100)
plt.title("Camera movement visualization - (b)")
plt.imshow(image_move)

# Camera Trajectory Estimation

depth_maps = dataset_handler.depth_maps
trajectory = features_algo.estimate_trajectory(features_algo.estimate_motion, matches, kp_list, k, depth_maps=depth_maps)

i = 1
print("Camera location in point {0} is: \n {1}\n".format(i, trajectory[:, [i]]))

# Note- The length of the returned by trajectory should be the same as the length of the image array
print("Length of trajectory: {0}".format(trajectory.shape[1]))

# Visualizations and Output Graphs 

# Part 1. Features Extraction
images = dataset_handler.images
kp_list, des_list = features_algo.extract_features_dataset(images, features_algo.extract_features)


# Part II. Feature Matching
matches = features_algo.match_features_dataset(des_list, features_algo.match_features)

# Set to True if you want to use filtered matches or False otherwise
is_main_filtered_m = True
if is_main_filtered_m:
    dist_threshold = 0.75
    filtered_matches = features_algo.filter_matches_dataset(features_algo.filter_matches_distance, matches, dist_threshold)
    matches = filtered_matches

    
# Part III. Trajectory Estimation
depth_maps = dataset_handler.depth_maps
trajectory = features_algo.estimate_trajectory(features_algo.estimate_motion, matches, kp_list, k, depth_maps= depth_maps)


#!!! Make sure you don't modify the output in any way
# Print Submission Info
print("Trajectory X:\n {0}".format(trajectory[0,:].reshape((1,-1))))
print("Trajectory Y:\n {0}".format(trajectory[1,:].reshape((1,-1))))
print("Trajectory Z:\n {0}".format(trajectory[2,:].reshape((1,-1))))

visualize_trajectory(trajectory)

print("Done!!")
plt.show()