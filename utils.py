import numpy as np

import math

import open3d as o3d

def orthogonalize_rotation_matrix(R):
    """
    Orthogonalize the given rotation matrix to ensure it is a valid rotation matrix.
    Ensure the z-direction remains correct.
    
    Parameters:
    R (np.array): 3x3 rotation matrix

    Returns:
    np.array: The closest valid 3x3 rotation matrix
    """
    # Extract the columns of the rotation matrix
    x = R[:, 0]
    y = R[:, 1]
    z = R[:, 2]

    # Ensure z is a unit vector
    z = z / np.linalg.norm(z)

    # Re-orthogonalize x and y with respect to z
    x = x - np.dot(x, z) * z
    x = x / np.linalg.norm(x)

    y = np.cross(z, x)
    
    # Construct the orthogonalized rotation matrix
    R_orthogonalized = np.column_stack((x, y, z))

    return R_orthogonalized

def rotation_matrix_to_euler_angles_xyz(R):
    """
    Convert a rotation matrix to Euler angles (XYZ order).
    
    Parameters:
    R (np.array): 3x3 rotation matrix

    Returns:
    tuple: Euler angles (alpha, beta, gamma) in radians
    """
    # Check if the matrix is a valid rotation matrix
    if not (np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-4, rtol=1e-4) and np.isclose(np.linalg.det(R), 1.0)):
        raise ValueError("The provided matrix is not a valid rotation matrix")

    # Calculate cy, which is used to detect gimbal lock
    cy = np.sqrt(R[0, 0] ** 2 + R[0, 1] ** 2)

    # Check for singularity (gimbal lock)
    singular = cy < 1e-6

    if not singular:
        # Calculate each Euler angle in XYZ order
        x = np.arctan2(R[2, 1], R[2, 2])  # Roll
        y = np.arctan2(-R[2, 0], R[2, 2])  # Pitch
        z = np.arctan2(R[1, 0], R[0, 0])  # Yaw
    else:
        # Handle singularity case
        x = np.arctan2(R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], R[2, 2])
        z = 0

    return np.array([x, y, z])  # XYZ order

def axangle2mat(axis, angle, is_normalized=False):
    ''' Rotation matrix for rotation angle `angle` around `axis`
    Parameters
    ----------
    axis : 3 element sequence
       vector specifying axis for rotation.
    angle : scalar
       angle of rotation in radians.
    is_normalized : bool, optional
       True if `axis` is already normalized (has norm of 1).  Default False.
    Returns
    -------
    mat : array shape (3,3)
       rotation matrix for specified rotation
    Notes
    -----
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    '''
    x, y, z = axis
    if not is_normalized:
        n = math.sqrt(x*x + y*y + z*z)
        x = x/n
        y = y/n
        z = z/n
    c = math.cos(angle); s = math.sin(angle); C = 1-c
    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC
    return np.array([
            [ x*xC+c,   xyC-zs,   zxC+ys ],
            [ xyC+zs,   y*yC+c,   yzC-xs ],
            [ zxC-ys,   yzC+xs,   z*zC+c ]])

def rxryrz2mat(rxryrz):

    x, y, z = rxryrz
    n = math.sqrt(x*x + y*y + z*z)
    x = x/n
    y = y/n
    z = z/n
    angle = n
    axis = rxryrz/n

    return axangle2mat(axis,angle)

def mat2axangle(mat, unit_thresh=1e-5):
        """Return axis, angle and point from (3, 3) matrix `mat`
        Parameters
        ----------
        mat : array-like shape (3, 3)
            Rotation matrix
        unit_thresh : float, optional
            Tolerable difference from 1 when testing for unit eigenvalues to
            confirm `mat` is a rotation matrix.
        Returns
        -------
        axis : array shape (3,)
           vector giving axis of rotation
        angle : scalar
           angle of rotation in radians.
        Examples
        --------
        # >>> direc = np.random.random(3) - 0.5
        # >>> angle = (np.random.random() - 0.5) * (2*math.pi)
        # >>> R0 = axangle2mat(direc, angle)
        # >>> direc, angle = mat2axangle(R0)
        # >>> R1 = axangle2mat(direc, angle)
        # >>> np.allclose(R0, R1)
        True
        Notes
        -----
        http://en.wikipedia.org/wiki/Rotation_matrix#Axis_of_a_rotation
        """
        M = np.asarray(mat, dtype=np.float32)
        # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
        L, W = np.linalg.eig(M.T)
        i = np.where(np.abs(L - 1.0) < unit_thresh)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        direction = np.real(W[:, i[-1]]).squeeze()
        # rotation angle depending on direction
        cosa = (np.trace(M) - 1.0) / 2.0
        if abs(direction[2]) > 1e-8:
            sina = (M[1, 0] + (cosa - 1.0) * direction[0] * direction[1]) / direction[2]
        elif abs(direction[1]) > 1e-8:
            sina = (M[0, 2] + (cosa - 1.0) * direction[0] * direction[2]) / direction[1]
        else:
            sina = (M[2, 1] + (cosa - 1.0) * direction[1] * direction[2]) / direction[0]
        angle = math.atan2(sina, cosa)
        
        return direction, angle

def mat2rxryrz(mat, unit_thresh=1e-5):
    direction, angle = mat2axangle(mat, unit_thresh)
    rxryrz = direction[:3] * angle
    return rxryrz

def ensure_vector_continuity(current_vector, new_vector):
    """
    Ensure continuity of rotation vectors by selecting the direction closest to the current vector.
    
    Parameters:
    current_vector (np.array): The current rotation vector (3-element array)
    new_vector (np.array): The new rotation vector (3-element array)
    
    Returns:
    np.array: A 3-element array representing the continuous rotation vector
    """
    if np.dot(current_vector, new_vector) < 0:
        return -new_vector
    return new_vector

def o3d_left_multiply_transform(mesh, M):
    """
    Apply a transformation matrix to all vertices of the mesh using left multiplication.
    
    Parameters:
    mesh (o3d.geometry.TriangleMesh): The mesh whose vertices will be transformed.
    M (np.array): The 4x4 transformation matrix to apply.
    
    Returns:
    None
    """
    # Get the vertices of the mesh
    vertices = np.asarray(mesh.vertices)
    
    # Convert to homogeneous coordinates
    ones = np.ones((vertices.shape[0], 1))
    vertices_homogeneous = np.hstack([vertices, ones])
    
    # Apply the transformation matrix using left multiplication
    transformed_vertices_homogeneous = M @ vertices_homogeneous.T
    
    # Convert back to 3D coordinates
    transformed_vertices = transformed_vertices_homogeneous[:3, :].T
    
    # Update the mesh vertices
    mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)

def limit_orientation_change(current_orientation, target_orientation, max_change=0.1):
    """
    Limit the change in orientation between two steps.
    
    Parameters:
    current_orientation (np.array): Current orientation as a rotation vector (3-element array)
    target_orientation (np.array): Target orientation as a rotation vector (3-element array)
    max_change (float): Maximum allowed change in orientation per step (in radians)
    
    Returns:
    np.array: New orientation as a rotation vector (3-element array)
    """
    # Compute the difference in orientation

    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm
    
    delta_orientation = target_orientation - current_orientation
    
    # Compute the angle of the rotation vector (delta_orientation)
    angle = np.linalg.norm(delta_orientation)
    
    if angle > max_change:
        # Limit the change to max_change
        delta_orientation = normalize(delta_orientation) * max_change
    
    # Compute the new orientation
    new_orientation = current_orientation + delta_orientation
    
    return new_orientation