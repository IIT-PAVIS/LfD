""" plotting - This module contains code used for visualising the results of LfD.

IIT - Italian Institute of Technology.
Pattern Analysis and Computer Vision (PAVIS) research line.

Ported to Python by Matteo Taiana.
"""


import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
import numpy as np

from lfd import dual_ellipse_to_parameters, project_ellipsoids, dual_quadric_to_ellipsoid_parameters


def plot_ellipse(C, colour):
    """Plots one ellipse on one existing figure.

    The input ellipse must be in dual form ([3x3] matrix).
    """
    # Only plot if the ellipse is valid: none of its element is NaN.
    if not((np.isnan(C)).any()):
        centre, axes, R = dual_ellipse_to_parameters(C)
        # Transform the rotation matrix into a rotation angle.
        # Remember: R = [cos(a), -sin(a); sin(a), cos(a)] -> a = atan2(R[1,0],R[0,0])
        angle_deg = np.rad2deg(math.atan2(R[1, 0], R[0, 0]))
        plot_axes = plt.gca()
        e = Ellipse(xy=centre, width=axes[0]*2, height=axes[1]*2,
                    angle=angle_deg, edgecolor=colour, linestyle='-',
                    linewidth=2, fill=False)
        plot_axes.add_artist(e)


def plot_est_and_gt_ellipses_on_images(K, Ms_t, estCs, gtQs, visibility, images, dataset, save_output_images):
    """Plot ellipses on images by projecting ellipsoids.

    Ground Truth ellipses are drawn in red, estimated ellipses are drawn in blue.
    If save_output_images is True, creates the output directory and stores the images there.
    """
    # Get the number of frames and the number of objects from the size of the visibility matrix.
    n_frames = visibility.shape[0]
    n_objects = visibility.shape[1]

    # Project GT ellipsoids onto the input images.
    Ps_t = np.transpose((np.dot(K, np.transpose(Ms_t))))
    if gtQs.shape[0] != 0:
        gt_ellipses = project_ellipsoids(Ps_t, gtQs, visibility)

    # Plot projection of estimated ellipsoids and projection of GT ellipsoids onto input images.
    for frame_id in range(n_frames):
        plt.figure(frame_id)
        plt.imshow(images[frame_id])
        # Plot the estimated ellipses.
        for obj_id in range(n_objects):
            # Only plot the ellipses if there is a label for this object in this image.
            if visibility[frame_id, obj_id]:
                estC = estCs[frame_id * 3:frame_id * 3 + 3, 3 * obj_id:3 * obj_id + 3]
                if gtQs.shape[0] != 0:
                    gtC = gt_ellipses[frame_id * 3:frame_id * 3 + 3, 3 * obj_id:3 * obj_id + 3]
                blue = (0, 0, 1)
                red = (1, 0, 0)
                # One more check: the object could be labelled in this image, but it might still not have a valid
                # estimate if it is visible in fewer than 3 frames in total.
                if not ((np.isnan(estC)).any()):
                    if gtQs.shape[0] != 0:
                        plot_ellipse(gtC, red)
                    plot_ellipse(estC, blue)
        plt.text(300, 50, 'Projection of GT   ellipsoids', {'color': 'r', 'fontsize': 12})
        plt.text(300, 70, 'Projection of Est. ellipsoids', {'color': 'b', 'fontsize': 12})
        if save_output_images:
            output_path = 'Output/{:s}/'.format(dataset)
            # Create output directory, in case it does not exist already.
            Path(output_path).mkdir(parents=True, exist_ok=True)
            plt.savefig('{:s}/projectedEllipsoids{:03d}.png'.format(output_path, frame_id), pad_inches=0.0)


def compute_ellipsoid_points(centre, axes, R):
    """Compute 3D points for plotting one ellipsoid."""
    size_side = 50  # Number of points for plotting one curve on the ellipsoid.

    # Compute the set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, size_side)
    v = np.linspace(0, np.pi, size_side)

    # Compute the Cartesian coordinates of the surface points of the ellipsoid aligned with the axes and
    # centred at the origin:
    # (this is the equation of an ellipsoid):
    x = axes[0] * np.outer(np.cos(u), np.sin(v))
    y = axes[1] * np.outer(np.sin(u), np.sin(v))
    z = axes[2] * np.outer(np.ones_like(u), np.cos(v))

    # Rotate the points according to R.
    x, y, z = np.tensordot(R, np.vstack((x, y, z)).reshape((3, size_side, size_side)), axes=1)

    # Apply the translation.
    x = x + centre[0]
    y = y + centre[1]
    z = z + centre[2]
    return x, y, z


def plot_ellipsoid(Q, colour, figure_axes):
    """Plot one 3D ellipsoid specified as a [4x4] matrix."""
    points = []
    [centre, axes, R] = dual_quadric_to_ellipsoid_parameters(Q)
    if centre is not None:
        [x, y, z] = compute_ellipsoid_points(centre, axes, R)
        figure_axes.plot_wireframe(x, y, z,  rstride=1, cstride=1,  color=colour, linewidth=0.5)
        n_points = x.shape[0]*x.shape[1]
        points = np.hstack((x.reshape(n_points, 1), y.reshape(n_points, 1), z.reshape(n_points, 1)))
    return points


def plot_camera(M, figure_axes):
    """Plot a pyramid to visualise the camera pose.

       The base of the pyramid points in the positive Z axis direction.
    """
    # Compute the points for the camera at the origin, aligned with the axes.
    base_width = 0.20  # Width (and height) of the pyramid base in meters
    x = np.array([0, 0, 0, 0, 0,  1,  1, -1, -1,  1])*base_width/2
    y = np.array([0, 0, 0, 0, 0,  1, -1, -1,  1,  1])*base_width/2
    z = np.array([0, 0, 0, 0, 0,  2,  2,  2,  2,  2])*base_width/2
    points = np.vstack((x, y, z))

    # Place the camera in the desired pose, which is the inverse of the pose specified by M:
    # M transforms the points from the world reference frame into the camera reference frame, so it also expresses
    # the pose of the world in the camera reference frame. We want the inverse of that: the pose of the camera
    # in the world reference frame.
    Mhom = np.vstack((M, (0, 0, 0, 1)))  # Cartesian to Homogeneous representation.
    Minv = np.linalg.inv(Mhom)  # Inverse transformation matrix (still Homogeneous representation).
    Minv /= Minv[3, 3]

    # Apply the rotation.
    R = Minv[0:3, 0:3]
    points = np.dot(R, points)

    # Apply the translation.
    t = Minv[0:3, 3]
    points = points + t.repeat(10).reshape(3, 10)

    x = points[0, :].reshape(2, 5)
    y = points[1, :].reshape(2, 5)
    z = points[2, :].reshape(2, 5)

    figure_axes.plot_wireframe(x, y, z,  rstride=1, cstride=1,  color=[0, 0, 0], linewidth=0.5)


def plot_3D_scene(estQs, gtQs, Ms_t, dataset, save_output_images):
    """Plot """
    fig = plt.figure(figsize=(8, 8))  # Open a new figure.
    figure_axes = fig.add_subplot(111, projection='3d')

    # Plot the GT ellipsoids in red.
    for ellipsoid_id in range(gtQs.shape[0]):
        plot_ellipsoid(gtQs[ellipsoid_id, :, :], [1, 0, 0], figure_axes)

    # Plot the estimated ellipsoids in blue.
    for ellipsoid_id in range(estQs.shape[0]):
        # Plot only if the data is valid
        if not ((np.isnan(estQs[ellipsoid_id, :, :])).any()):
            _ = plot_ellipsoid(estQs[ellipsoid_id, :, :], [0, 0, 1], figure_axes)

    # Plot the camera poses in black.
    for pose_id in range(Ms_t.shape[0]//4):
        plot_camera(Ms_t[pose_id*4:pose_id*4+4, :].transpose(), figure_axes)

    figure_axes.set_xlabel('X axis')
    figure_axes.set_ylabel('Y axis')
    figure_axes.set_zlabel('Z axis')

    red_patch = mpatches.Patch(color='red', label='GT')
    blue_patch = mpatches.Patch(color='blue', label='Estimates')
    plt.legend(handles=[red_patch, blue_patch])

    # This forces axes to be equal, but also forces the scene to be a cube.
    # MatPlotLib does not have an "axes equal" function.
    # figure_axes.auto_scale_xyz([-30, 40], [0, -70], [-35, 25])
    fig.show()

    if save_output_images:
        output_path = 'Output/{:s}/'.format(dataset)
        # Create output directory, in case it does not exist already.
        Path(output_path).mkdir(parents=True, exist_ok=True)
        plt.savefig('{:s}/ellipsoids.png'.format(output_path), pad_inches=0.0)
