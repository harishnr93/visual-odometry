"""
Date: 23.Nov.2024
Author: Harish Natarajan Ravi
Email: harrish.nr@gmail.com
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from m2bk import *

def extract_features(image):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """ 
    #sift = cv2.SIFT_create()
    orb = cv2.ORB_create(nfeatures = 1500)
    kp, des = orb.detectAndCompute(image,None)
    
    return kp, des

def visualize_features(image, kp):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints

    Returns:
    """
    display = cv2.drawKeypoints(image, kp, None)
    plt.title("Features Visualization")
    plt.imshow(display)

def extract_features_dataset(images, extract_features_function):
    """
    Find keypoints and descriptors for each image in the dataset

    Arguments:
    images -- a list of grayscale images
    extract_features_function -- a function which finds features (keypoints and descriptors) for an image

    Returns:
    kp_list -- a list of keypoints for each image in images
    des_list -- a list of descriptors for each image in images
    
    """
    kp_list = []
    des_list = []
    
    for img in images:
        #kp, des = extract_features(img)
        kp, des = extract_features_function(img)
        kp_list.append(kp)
        des_list.append(des)
    
    return kp_list, des_list

def match_features(des1, des2):
    """
    Match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image

    Returns:
    match -- list of matched features from two images. Each match[i] is k or less matches for the same query descriptor
    """
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 16)
    search_params = dict(checks=50)   
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    match = flann.knnMatch(np.float32(des1),np.float32(des2),k=2)

    return match

def filter_matches_distance(match, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_match = []
    
    for m,n in match:
        distance_ratio = m.distance / n.distance
        if distance_ratio < dist_threshold:
            filtered_match.append([m,None])

    return filtered_match

def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
    """
    image_matches = cv2.drawMatches(image1,kp1,image2,kp2,match,None)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.title("Matches Visualization")
    plt.imshow(image_matches)

def match_features_dataset(des_list, match_features):
    """
    Match features for each subsequent image pair in the dataset

    Arguments:
    des_list -- a list of descriptors for each image in the dataset
    match_features -- a function which maches features between a pair of images

    Returns:
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
               
    """
    matches = []
    
    for i in range(len(des_list)-1):
        descriptor1 = des_list[i]
        descriptor2 = des_list[i+1]
        match = match_features(descriptor1, descriptor2)
        matches.append(match)
    
    return matches

def filter_matches_dataset(filter_matches_distance, matches, dist_threshold):
    """
    Filter matched features by distance for each subsequent image pair in the dataset

    Arguments:
    filter_matches_distance -- a function which filters matched features from two images by distance between the best matches
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_matches -- list of good matches for each subsequent image pair in the dataset. 
                        Each matches[i] is a list of good matches, satisfying the distance threshold
               
    """
    filtered_matches = []
    
    for i in range(len(matches)):
        new_match = filter_matches_distance(matches[i], dist_threshold)
        filtered_matches.append(new_match)
    
    return filtered_matches

def estimate_motion(match, kp1, kp2, k, depth1=None):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera calibration matrix 
    
    Optional arguments:
    depth1 -- a depth map of the first frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    rmat -- recovered 3x3 rotation numpy matrix
    tvec -- recovered 3x1 translation numpy vector
    image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system
    image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system
               
    """
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []
    
    # Extract matching points
    image1_points = [kp1[m.queryIdx].pt for m, _ in match]
    image2_points = [kp2[m.trainIdx].pt for m, _ in match]
    
    if depth1 is None:
        # Use Essential Matrix Decomposition
        pts1 = np.array(image1_points)
        pts2 = np.array(image2_points)
        
        # Find the Essential matrix and recover pose
        E, mask_match = cv2.findEssentialMat(pts1, pts2, k, method=cv2.RANSAC)
        _, rmat, tvec, _ = cv2.recoverPose(E, pts1, pts2, k)
    else:
        # Use PnP with RANSAC
        f, cu, cv = k[0, 0], k[0, 2], k[1, 2]
        objectPoints = []
        
        for x, y in image1_points:
            z = depth1[int(y), int(x)]
            if z > 0:  # Ensure valid depth values
                objectPoints.append([(x - cu) * z / f, (y - cv) * z / f, z])
        
        objectPoints = np.array(objectPoints, dtype=np.float32)
        image2_points = np.array(image2_points, dtype=np.float32)
        
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints, image2_points, k, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
        rmat, _ = cv2.Rodrigues(rvec)

    return rmat, tvec, image1_points, image2_points

def estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=[]):
    """
    Estimate complete camera trajectory from subsequent image pairs

    Arguments:
    estimate_motion -- a function which estimates camera motion from a pair of subsequent image frames
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
    kp_list -- a list of keypoints for each image in the dataset
    k -- camera calibration matrix 
    
    Optional arguments:
    depth_maps -- a list of depth maps for each frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    trajectory -- a 3xlen numpy array of the camera locations, where len is the lenght of the list of images and   
                  trajectory[:, i] is a 3x1 numpy vector, such as:
                  
                  trajectory[:, i][0] - is X coordinate of the i-th location
                  trajectory[:, i][1] - is Y coordinate of the i-th location
                  trajectory[:, i][2] - is Z coordinate of the i-th location
                  
                  * Consider that the origin of your trajectory cordinate system is located at the camera position 
                  when the first image (the one with index 0) was taken. The first camera location (index = 0) is geven 
                  at the initialization of this function

    """
#     trajectory = np.zeros((3, 1))

    trajectory = [np.array([0, 0, 0])]
    
    R = np.diag([1,1,1])
    T = np.zeros([3, 1])
    RT = np.hstack([R, T])
    RT = np.vstack([RT, np.zeros([1, 4])])
    RT[-1, -1] = 1
    
    for i in range(len(matches)):     
        match = matches[i]
        kp1 = kp_list[i]
        kp2 = kp_list[i+1]
        depth = depth_maps[i]
        
        rmat, tvec, _, _ = estimate_motion(match, kp1, kp2, k, depth1= None)
        rt_mtx = np.hstack([rmat, tvec])
        rt_mtx = np.vstack([rt_mtx, np.zeros([1, 4])])
        rt_mtx[-1, -1] = 1
        
        rt_mtx_inv = np.linalg.inv(rt_mtx)
        
        RT = np.dot(RT, rt_mtx_inv)
        new_trajectory = RT[:3, 3]
        trajectory.append(new_trajectory)
    
    trajectory = np.array(trajectory).T

    return trajectory