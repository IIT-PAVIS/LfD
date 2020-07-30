import numpy as np
import math


def dual_quadric_to_ellipsoid_parameters(Q):
    """Computes centre, axes length and orientation of one ellipsoid.

   A [4x4] matrix can represent general quadrics. In spite of preconditioning, the estimated quadrics can still,
   in rare cases, represent something other than ellipsoids. This is corrected by forcing the lengths of the
   axes to be positive.

    :param Q: Ellipsoid/Quadric in dual form [4x4].

    :returns:
      - centre - Ellipsoid centre in Cartesian coordinates [3x1].
      - axes - Length of ellipsoid axes [3x1].
      - R - Orientation of the ellipsoid [3x3].
    """

    # Scale the ellipsoid to put it in the usual form, with Q[3,3] = -1.
    Q = Q / (-Q[3, 3])

    # Compute ellipsoid centred on origin.
    centre = -Q[:3, 3]
    T = np.vstack((np.array((1, 0, 0, -centre[0])), np.array((0, 1, 0, -centre[1])),
                   np.array((0, 0, 1, -centre[2])), np.array((0, 0, 0, 1))))
    Qcent = T.dot(Q).dot(T.transpose())

    # Compute axes and orientation.
    [D, V] = np.linalg.eig(Qcent[:3, :3])
    sort_ev = np.argsort(D)
    V = np.vstack((V[:, sort_ev[0]], V[:, sort_ev[1]], V[:, sort_ev[2]])).transpose()
    D.sort()
    if sum(D < 0) > 0:
        # Take the absolute value of eigenvalues (they can be negative because of numerical issues).
        for index in range(D.shape[0]):
            if D[index] < 0:
                D[index] *= -1

    a = np.sqrt(D[0])
    b = np.sqrt(D[1])
    c = np.sqrt(D[2])
    
    axes = np.array([a, b, c])
    R = V
    return centre, axes, R


def dual_ellipse_to_parameters(C):
    """Computes centre, axes length and orientation of one ellipse.

    :param C: Ellipse in dual form [3x3].

    :returns:
      - centre - Ellipse centre in Cartesian coordinates [2x1].
      - axes - Ellipse axes lengths [2x1].
      - R - Ellipse orientation [2x2].
    """
    if C[2, 2] > 0:
        C = C / -C[2, 2]

    centre = (-C[0:2, 2]).reshape(2, 1)

    T = np.vstack((np.hstack((np.eye(2), -centre)), np.array([0, 0, 1])))

    C_centre = T.dot(C).dot(T.transpose())
    C_centre = 0.5 * (C_centre + C_centre.transpose())
    D, V = np.linalg.eig(C_centre[0:2, 0:2])

    axes = np.sqrt(abs(D))
    R = V

    return centre, axes, R


def fit_one_ellipse_in_bb(bb):
    """Computes the ellipse inscribed in the bounding box that is provided.

    The axes of the ellipse will be aligned with the axes of the image.

    :param bb: Bounding box, in the format: [X0, Y0, X1, Y1].

    :returns C: Ellipse in dual form [3x3].
    """

    # Encode ellipse size (axes).
    width = abs(bb[2]-bb[0])/2  # Width of the bounding box.
    height = abs(bb[3]-bb[1])/2  # Height of the bounding box.
    Ccn = np.vstack((np.hstack((np.diag((1/width**2, 1/height**2)), np.zeros((2, 1)))), np.array((0, 0, -1))))

    # Encode ellipse location.
    centre = np.array(((bb[0]+bb[2])/2, (bb[1]+bb[3])/2))  # Bounding box centre.
    P = np.vstack((np.hstack((np.eye(2, 2), centre.reshape(2, 1))), np.array((0, 0, 1,))))
    Cinv = P.dot(np.linalg.inv(Ccn)).dot(P.transpose())

    # Force matrix to be symmetric: Cinv = (Cinv + Cinv') / 2
    Cinv = 0.5 * (Cinv + Cinv.transpose())

    # Scale ellipse so that element [2,2] is 1.
    C = Cinv / Cinv[2, 2]

    C = C * np.sign(C[0, 0]+C[1, 1])
    return C


def fit_ellipses_in_bbs(bbs, visibility):
    """Computes several ellipses, in dual form, each one inscribed in one bounding box.

    The axes of the ellipses will be aligned with the axes of the image.

    :param bbs: Detection bounding boxes. Format: [X0, Y0, X1, Y1], size: [n_frames x n_objects * 4].
    :param visibility:  Object visibility information: [n_frames x n_objects].

    :returns: Cs - Ellipses fitted to the bounding boxes, for each image and each object, in dual form.
                   Size: [n_frames*3 x n_objects*3]. Each ellipse is described by a [3x3] submatrix.
    """

    # Get the number of frames and the number of objects from the size of the visibility matrix.
    n_frames = visibility.shape[0]
    n_objects = visibility.shape[1]

    Cs = np.zeros([n_frames*3, n_objects*3])
    for frame in range(n_frames):
        for obj_id in range(n_objects):
            if visibility[frame, obj_id]:
                C = fit_one_ellipse_in_bb(bbs[frame, obj_id*4:(obj_id+1)*4])
                Cs[frame*3:(frame+1)*3, obj_id*3:(obj_id+1)*3] = C
    return Cs


def vector_to_symmetric_mat_4(vec):
    """Builds a symmetric 4x4 matrix using the elements specified in a vector.

    The elements are copied first to the first row, in order, then to the second
    row, starting from the element on the diagonal, and so on.
    """
    A = np.zeros((4, 4))
    # Fill the top-right triangular matrix
    A[0, 0] = vec[0]
    A[0, 1] = vec[1]
    A[0, 2] = vec[2]
    A[0, 3] = vec[3]
    A[1, 1] = vec[4]
    A[1, 2] = vec[5]
    A[1, 3] = vec[6]
    A[2, 2] = vec[7]
    A[2, 3] = vec[8]
    A[3, 3] = vec[9]
    # Fill the rest of the elements
    A[1, 0] = vec[1]
    A[2, 0] = vec[2]
    A[3, 0] = vec[3]
    A[2, 1] = vec[5]
    A[3, 1] = vec[6]
    A[3, 2] = vec[8]
    return A


def symmetric_mat_3_to_vector(C):
    """Serialises a symmetric 3x3 matrix.

    First, the elements of the first row are take in order.
    Then, the elements of the second row, starting from the one on the diagonal.
    Lastly, the third element of the third row is copied.
    """
    serialised = np.zeros(6)
    serialised[0] = C[0, 0]
    serialised[1] = C[0, 1]
    serialised[2] = C[0, 2]
    serialised[3] = C[1, 1]
    serialised[4] = C[1, 2]
    serialised[5] = C[2, 2]
    return serialised


def estimate_one_ellipsoid(Ps_t, Cs):
    """Estimates one ellipsoid given projection matrices and detection ellipses for one image sequence.

    Transformations are applied to precondition the numerical problem. Equations are then rearranged to
    form the linear system M * w = 0. The least squares solution is computed using SVD.

    :param Ps_t: Stacked and transposed projection matrices, only dor the frames in which the current object was
                 detected (n_views) [n_views * 4 x 3].
    :param Cs: Ellipses fitted to the input bounding boxes, for the current object, in dual form, only for the frames
               in which the current object was detected (n_views). Size: [n_views*3 x 3]. Each ellipse is described by
               a [3x3] submatrix.

    :returns adj_Q: The estimated ellipse, in dual form [4x4].
    """
    n_views = math.floor(Cs.shape[0]/3)  # Number of frames in which the current object was detected.

    M = np.zeros((6*n_views, 10+n_views))  # Matrix which represents the linear system to be solved.
                                           # This has nothing to do with the movement matrix, the name has been
                                           # kept like this for consistency with the name used in the paper.

    # Compute the B matrices and stack them into M.
    for index in range(n_views):
        # Get centre and axes of current ellipse.
        [centre, axes, _] = dual_ellipse_to_parameters(Cs[3 * index:3 * index + 3, :])

        # Compute T, a transformation used to precondition the ellipse: centre the ellipse and scale the axes.
        div_f = np.linalg.norm(axes)  # Distance of point (A,B) from origin.
        T   = np.linalg.inv((np.vstack((np.hstack((np.eye(2)*div_f, centre)), np.array([0, 0, 1])))))
        T_t = T.transpose()

        # Compute P_fr, applying T to the projection matrix.
        P_fr = np.dot(Ps_t[4*index:4*index+4, :], T_t)

        # Compute the coefficients for the linear system based on the current P_fr.
        B = compute_B(P_fr)

        # Apply T to the ellipse.
        C_t = np.dot(np.dot(T, Cs[3*index:3*index+3, :]), T_t)

        # Transform the ellipse to vector form.
        C_tv = symmetric_mat_3_to_vector(C_t)
        C_tv /= -C_tv[5]

        # Write the obtained coefficients to the correct slice of M.
        M[6*index:6*index+6, 0:10] = B
        M[6*index:6*index+6, 10+index] = -C_tv

    _, _, V = np.linalg.svd(M)
    w = V[-1, :]  # V is transposed respect to Matlab, so we take the last row.

    # Take the 10 elements of the result and build the ellipsoid matrix (dual form).
    # 10 elements are sufficient because the matrix is symmetric.
    Qadjv = w[0:10]
    adj_Q = vector_to_symmetric_mat_4(Qadjv)

    return adj_Q


def compute_B(P_fr):
    """Rearranges the parameters so that it is possible to estimate the ellipsoid by solving a linear system.

       Please refer to the paper for details.
    """
    B = np.zeros((6, 10))

    # Vectorise P, one row after the other (C order = row-major).
    vec_p = np.reshape(P_fr, (12, 1), order='C')

    r = vec_p[0:9]
    t = vec_p[9:12]

    # Fill B.
    B[0, :] = (r[0] ** 2,
               2 * r[0] * r[3],
               2 * r[0] * r[6],
               2 * r[0] * t[0],
               r[3] ** 2,
               2 * r[3] * r[6],
               2 * r[3] * t[0],
               r[6] ** 2,
               2 * r[6] * t[0],
               t[0] ** 2)

    B[1, :] = (r[1] * r[0],
               r[1] * r[3] + r[4] * r[0],
               r[1] * r[6] + r[7] * r[0],
               t[1] * r[0] + r[1] * t[0],
               r[4] * r[3],
               r[4] * r[6] + r[7] * r[3],
               r[4] * t[0] + t[1] * r[3],
               r[7] * r[6],
               t[1] * r[6] + r[7] * t[0],
               t[1] * t[0])

    B[2, :] = (r[2] * r[0],
               r[2] * r[3] + r[5] * r[0],
               r[2] * r[6] + r[8] * r[0],
               t[2] * r[0] + r[2] * t[0],
               r[5] * r[3],
               r[5] * r[6] + r[8] * r[3],
               r[5] * t[0] + t[2] * r[3],
               r[8] * r[6],
               t[2] * r[6] + r[8] * t[0],
               t[2] * t[0])

    B[3, :] = (r[1] ** 2,
               2 * r[1] * r[4],
               2 * r[1] * r[7],
               2 * r[1] * t[1],
               r[4] ** 2,
               2 * r[4] * r[7],
               2 * r[4] * t[1],
               r[7] ** 2,
               2 * r[7] * t[1],
               t[1] ** 2)

    B[4, :] = (r[2] * r[1],
               r[2] * r[4] + r[5] * r[1],
               r[2] * r[7] + r[8] * r[1],
               t[2] * r[1] + r[2] * t[1],
               r[5] * r[4],
               r[5] * r[7] + r[8] * r[4],
               r[5] * t[1] + t[2] * r[4],
               r[8] * r[7],
               t[2] * r[7] + r[8] * t[1],
               t[2] * t[1])

    B[5, :] = (r[2] ** 2,
               2 * r[2] * r[5],
               2 * r[2] * r[8],
               2 * r[2] * t[2],
               r[5] ** 2,
               2 * r[5] * r[8],
               2 * r[5] * t[2],
               r[8] ** 2,
               2 * r[8] * t[2],
               t[2] ** 2)

    return B


def estimate_ellipsoids(Ps_t, input_ellipsoids_centres, inputCs, visibility):
    """Estimates one ellipsoid per object given projection matrices and detection ellipses for one image sequence.

    Input consists of one projection matrix per image frame, plus detection ellipses (with data association) for each
    object. If a previous estimate for the location of the ellipsoids is available, it can be provided.

    :param Ps_t: Stacked and transposed projection matrices, [n_frames * 4 x 3].
    :param input_ellipsoids_centres: If an initial estimate is available for the location of the ellipsoids, it can be
                                     provided via this parameter. It has to be filled with zeros (in Homogenous
                                     format), otherwise. Represented in Homogeneous coordinates [4 x n_objects].
    :param inputCs: Ellipses fitted to the input bounding boxes, for each image and each object, in dual form.
                    Size: [n_frames*3 x n_objects*3]. Each ellipse is described by a [3x3] submatrix.
    :param visibility: : Object visibility information: [n_frames x n_objects].

    :returns: Estimated ellipsoids, in dual form [n_objects x 4 x 4].
    """

    n_objects = visibility.shape[1]
    # Initialise output structure.
    estQs = np.zeros((n_objects, 4, 4))

    for obj in range(n_objects):
        # If there are at least 3 detections for this object, then proceed with the estimation.
        if sum(visibility[:, obj]) >= 3:
            # Select only the C's [3x3] for the current object, for the frames in which it was detected.
            # Create a mask with True in the desired locations.
            row_selector = np.kron(visibility[:, obj], np.ones(3).reshape(1, 3))
            row_selector = np.array(row_selector, dtype=bool)[0]
            # Apply the mask.
            selectedCs = inputCs[row_selector, obj*3:obj*3+3]

            # Select the corresponding projection matrices.
            # Create the mask.
            row_selector = np.kron(visibility[:, obj], np.ones(4).reshape(1, 4))
            row_selector = np.array(row_selector, dtype=bool)[0]
            # Apply the mask.
            selectedPs_t = Ps_t[row_selector, :]  # Selected Projection matrices, transposed.

            # Compute the translation matrix due to the centre of the current ellipsoid.
            translM = np.eye(4)
            translM[0:3, 3] = input_ellipsoids_centres[0:3, obj]

            # Loop over the frames in which the current object is present,
            # apply the translation matrix to each projection matrix, for numerical preconditioning.
            for instance_id in range(math.floor(selectedPs_t.shape[0]/4)):
                first = np.hstack((np.eye(3), np.zeros((3, 1))))
                second = np.vstack((selectedPs_t[instance_id*4:instance_id*4+4, :].transpose(), np.array((0, 0, 0, 1))))
                selectedPs_t[instance_id*4:instance_id*4+4, :] = np.dot(np.dot(first, second), translM).transpose()

            # Estimate the parameters of the current ellipsoid.
            estQ = estimate_one_ellipsoid(selectedPs_t, selectedCs)

            # Re-apply the translation which had been removed.
            estQ = np.dot(translM, np.dot(estQ, translM.transpose()))

            # Force the estQ matrix to be symmetric: estQ = (estQ + estQ') / 2
            estQ = 0.5 * (estQ + estQ.transpose())

            # Scale the ellipsoid to put it in standard form, with element[3,3] set to -1.
            estQ /= -estQ[3, 3]

            # Store the results for the current ellipsoid into the output data structure.
            estQs[obj, :, :] = estQ

        else:  # In case there are fewer than 3 detections for this object, output NaN values for the estimate.
            estQs[obj, :, :] = np.nan

    return estQs


def project_ellipsoids(Ps_t, estQs, visibility):
    """Project the ellipsoids onto the image, producing ellipses.

    :param Ps_t: Stacked and transposed projection matrices, [n_frames * 4 x 3].
    :param estQs: Estimated ellipsoids, in dual form [n_objects x 4 x 4].
    :param visibility: Object visibility information: [n_frames x n_objects].

    :returns Cs: Ellipses in dual form [n_frames * 3 x n_objects * 3].
    """

    # Get the number of frames and the number of objects from the size of the visibility matrix.
    n_frames = visibility.shape[0]
    n_objects = visibility.shape[1]

    Cs = np.zeros([n_frames*3, n_objects*3])
    for frame in range(n_frames):
        for obj in range(n_objects):
            # If the estimate is valid (not NaN) and the object is visible in this frame:
            if not((np.isnan(estQs[obj, :, :])).any()) and visibility[frame, obj]:
                # Transform the ellipsoid to the camera reference frame and project them.
                P = Ps_t[frame*4:frame*4+4, :].transpose()
                Ctemp = np.dot(np.dot(P, estQs[obj, :, :]), P.transpose())
                # Scale the ellipse to put it in standard form, with element [2,2] set to 1.
                Ctemp /= Ctemp[2, 2]
                # Write the result in the output structure.
                Cs[frame*3:frame*3+3, obj*3:obj*3+3] = Ctemp
            else:  # Propagate the NaN's of the ellipsoid to the ellipse.
                Cs[frame*3:frame*3+3, obj*3:obj*3+3] = np.nan
    return Cs


def compute_estimates(bbs, K, Ms_t, visibility):
    """Estimate one ellipsoid per object given detection bounding boxes and camera parameters.

    :param bbs: Detection bounding boxes. Format: [X0, Y0, X1, Y1], size: [n_frames x n_objects * 4].
    :param K: Camera intrinsics, [3x3].
    :param Ms_t: Camera pose matrices, transposed and stacked, [n_frames * 4 x 3]. Each submatrix transforms points from
                 the World reference frame to the Camera reference frame [3x4].
    :param visibility:  Object visibility information: [n_frames x n_objects].

    :returns:
        - inputCs - Ellipses fitted to the input bounding boxes, for each image and each object, in dual form.
                    Size: [n_frames*3 x n_objects*3]. Each ellipse is described by a [3x3] submatrix.
        - estCs - Ellipses resulting from the projection of the estimated ellipsoids, in dual form.
                  Size: [n_frames*3 x n_objects*3]. Each ellipse is described by a [3x3] submatrix.
        - estQs_second_step - Estimated ellipsoids, one per detected object, in dual form. [n_objects x 4 x 4].
    """

    # Get the number of frames and the number of objects from the size of the visibility matrix.
    n_frames = visibility.shape[0]
    n_objects = visibility.shape[1]

    # Compute the stacked and transposed projection matrices.
    # Ps_t = (K*Ms_t')'   [n_frames * 4 x 3].
    Ps_t = np.transpose((np.dot(K, np.transpose(Ms_t))))

    # Compute ellipses inscribed in the detection bounding boxes.
    inputCs = fit_ellipses_in_bbs(bbs, visibility)

    # Set the initial ellipsoids centres to the origin.
    input_ellipsoids_centres = np.dot(np.array(([0], [0], [0], [1])), (np.ones((1, n_objects))))

    # Perform the first round of estimation.
    estQs_first_step = estimate_ellipsoids(Ps_t, input_ellipsoids_centres, inputCs, visibility)

    # Extract the centres of the current estimates for the ellipsoids.
    first_step_ellipsoids_centres = input_ellipsoids_centres
    for object_id in range(n_objects):
        first_step_ellipsoids_centres[0:3, object_id] = estQs_first_step[object_id, 0:3, 3]

    # Perform the second round of estimation, exploiting the centres estimated at the previous step.
    estQs_second_step = estimate_ellipsoids(Ps_t, first_step_ellipsoids_centres, inputCs, visibility)

    # Project the estimated ellipsoids onto the images.
    estCs = project_ellipsoids(Ps_t, estQs_second_step, visibility)

    return inputCs, estCs, estQs_second_step
