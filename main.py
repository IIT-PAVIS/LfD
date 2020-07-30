import pickle
import numpy as np
from matplotlib import pyplot as plt

from lfd import compute_estimates
from plotting import plot_est_and_gt_ellipses_on_images, plot_3D_scene

'''LfD - Localisation from Detections.

IIT - Italian Institute of Technology.
Pattern Analysis and Computer Vision (PAVIS) research line.

If you use this project for your research, please cite:
@inproceedings{rubino2017pami,  
title={3D Object Localisation from Multi-view Image Detections},  
author={Rubino, Cosimo and Crocco, Marco and Del Bue, Alessio},  
booktitle={Pattern Analysis and Machine Intelligence (TPAMI), 2017 IEEE Transactions on},  
year={2017},  
organization={IEEE}  

Ported to Python by Matteo Taiana.
'''

####################
# 0. Introduction. #
####################
# Conventions: variable names such as Ms_t indicate that there are multiple M matrices (Ms)
# which are transposed (_t) respect to the canonical orientation of such data.
# Prefixes specify if one variable refers to input data, estimates etc.: inputCs, estCs, etc.
#
# Variable names:
# C - Ellipse in dual form [3x3].
# Q - Quadric/Ellipsoid in dual form [4x4], in the World reference frame.
# K - Camera intrinsics [3x3].
# M - Pose matrix: transforms points from the World reference frame to the Camera reference frame [3x4].
# P - Projection matrix = K * M [3*4].
#
# A note on the visibility information: when false, it might mean that either the object is not visible in the image,
# or that the detector failed. For these cases the algorithm does not visualise the estimated, nor the GT ellipses.
#
# If one object is not detected in at least 3 frames, it is ignored. The values of the corresponding ellipsoid and
# ellipses are set to NaN, so the object is never visualised.


###########################################
# 1. Set the parameters for the algorithm #
#    and load the input data.             #
###########################################
# Select the dataset to be used.
# The name of the dataset defines the names of input and output directories.
dataset = 'Aldoma'  # Data used in the original (Matlab) implementation of Lfd, published with this paper:
                    # A. Aldoma, T. Faulhammer, and M. Vincze, “Automation of ground truth annotation for multi-view
                    # rgb-d object instance recognition datasets,” in IROS 2014.

# Select whether to save output images to files.
save_output_images = True

# Load the input data.
# Data association is implicitly defined in the data structures: each column of "visibility" corresponds to one object.
# The information in "bbs" is structured in a similar way, with four columns for each object.
bbs = np.load('Data/{:s}/InputData/bounding_boxes.npy'.format(dataset))  # Bounding boxes [X0, Y0, X1, Y1],
                                                                         # [X0, Y0] = top-left corner
                                                                         # [X1, Y1] = bottom-right corner
                                                                         # [n_frames x n_objects * 4].
K = np.load('Data/{:s}/InputData/intrinsics.npy'.format(dataset))  # Camera intrinsics [3x3].
Ms_t = np.load('Data/{:s}/InputData/camera_poses.npy'.format(dataset))  # Camera pose matrices, transposed and stacked.
                                                                        # [n_frames * 4 x 3].
                                                                        # Each matrix [3x4] transforms points from the
                                                                        # World reference frame to the Camera reference
                                                                        # frame.
visibility = np.load('Data/{:s}/InputData/visibility.npy'.format(dataset))  # Visibility information, indicates whether
                                                                            # a detection is available for a given
                                                                            # object, on a given frame.
                                                                            # [n_frames x n_objects].

# Compute the number of frames and the number of objects for the current dataset from the size of the visibility matrix.
n_frames = visibility.shape[0]
n_objects = visibility.shape[1]


######################################
# 2. Run the algorithm: estimate the #
#    object ellipsoids.              #
######################################
[inputCs, estCs, estQs] = compute_estimates(bbs, K, Ms_t, visibility)


#############################
# 3. Visualise the results. #
#############################
# Load the input images.
with open('Data/{:s}/InputData/images.bin'.format(dataset), 'rb') as handle:
    images = pickle.load(handle)

# Load Ground Truth ellipsoids.
gtQs = np.load('Data/{:s}/GroundTruthData/gt.npy'.format(dataset), allow_pickle=True)

# Plot ellipses on input images.
plot_est_and_gt_ellipses_on_images(K, Ms_t, estCs, gtQs, visibility, images, dataset, save_output_images)

# Plot ellipsoids and camera poses in 3D.
plot_3D_scene(estQs, gtQs, Ms_t, dataset, save_output_images)

# Visualise the plots that have been produced.
plt.show()
