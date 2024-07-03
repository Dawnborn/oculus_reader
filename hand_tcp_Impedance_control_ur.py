import os
import numpy as np
# from oculus_reader.reader import OculusReader
from oculus_reader.reader_hand import OculusHandReader
import time
import math

import rtde_control
import rtde_receive

import open3d as o3d

from kalmanfilter import KalmanFilter

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

def urpose2homomatrix(ur_pose):
    """
    This function calculates the homogeneous transformation matrix from the given pose.
    
    Parameters:
    ur_pose (np.array): A 6-element array containing px, py, pz, rx, ry, rz

    Returns:
    np.array: A 4x4 homogeneous transformation matrix
    """
    px, py, pz, rx, ry, rz = ur_pose
    
    angle = np.sqrt(rx**2 + ry**2 + rz**2)

    kx = rx / angle
    ky = ry / angle
    kz = rz / angle

    # Initialize the 4x4 homogeneous transformation matrix with zeros
    homo_matrix = np.zeros((4, 4))

    # Fill in the rotation part of the matrix
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    one_minus_cos = 1 - cos_angle

    homo_matrix[0, 0] = cos_angle + kx * kx * one_minus_cos
    homo_matrix[0, 1] = kx * ky * one_minus_cos - kz * sin_angle
    homo_matrix[0, 2] = ky * sin_angle + kx * kz * one_minus_cos
    homo_matrix[0, 3] = px

    homo_matrix[1, 0] = kz * sin_angle + kx * ky * one_minus_cos
    homo_matrix[1, 1] = cos_angle + ky * ky * one_minus_cos
    homo_matrix[1, 2] = -kx * sin_angle + ky * kz * one_minus_cos
    homo_matrix[1, 3] = py

    homo_matrix[2, 0] = -ky * sin_angle + kx * kz * one_minus_cos
    homo_matrix[2, 1] = kx * sin_angle + ky * kz * one_minus_cos
    homo_matrix[2, 2] = cos_angle + kz * kz * one_minus_cos
    homo_matrix[2, 3] = pz

    # The last row of the homogeneous transformation matrix
    homo_matrix[3, 0] = 0
    homo_matrix[3, 1] = 0
    homo_matrix[3, 2] = 0
    homo_matrix[3, 3] = 1

    return homo_matrix

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

def test():
    # success = diana.set_cartesian_impedance(stiffness=(100, 100, 100, 100, 100, 100), damping=[0.5])
    # print(f"Set impedance:{success}")
    # success = diana.change_control_mode("cartesian_impedance")
    # print(f"change control mode:{success}")
    rtde_c = rtde_control.RTDEControlInterface("10.3.15.95")
    rtde_r = rtde_receive.RTDEReceiveInterface("10.3.15.95")
    init_q = rtde_r.getActualQ()  # [0.26260805130004883, -1.2464478772929688, -1.9548711776733398, -1.4632833761027833, 1.7008233070373535, -4.365320030842916]

    joints_pos_home = np.asarray(
        [0.26260805130004883, -1.2464478772929688, -1.9548711776733398, -1.4632833761027833, 1.7008233070373535,
         -4.365320030842916])
    # success = diana.move_joints(joints_pos_home,wait=True, vel=0.5)


    tcp_current_pose = rtde_r.getActualTCPPose()  # xyz rx ry rz

    rot_vr2bot = np.array([1,0,0,0,0,-1,0,1,0]).reshape((3, 3))  # 眼镜与机器人基座的坐标系转换
    # rot_tcp = np.array([-1,0,0, 0,1,0, 0,0,-1]).reshape((3,3))
    # rot_tcp = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape((3, 3))  # tcp与手柄的转换关系
    rot_tcp = np.array([0,1,0, 0,0,1, 1,0,0]).reshape((3,3)) #ur tcp与手的坐标关系
    rot_tcp = np.eye(3)

    h_vr2bot = np.eye(4)
    h_vr2bot[:3,:3] = rot_vr2bot # 硬编码实现眼镜到机器人base的转换
    h_vr2bot[0,3] = -1

    # scale = 1
    scale = 4
    # scale = 6 # servo
    # scale = 12
    # scale = 24
    # scale = 36
    # scale = 42
    # scale = 1

    TIME_STEP = 0.05
    VIS=True
    MOVE_ROBOT=False
    if MOVE_ROBOT:
        import pdb
        input("Warning: You are trying to move the real robot!!! Press Enter to continue...")

    USE_KALMAN_POS=True
    USE_KALMAN_ORIENT=False

    ORIENTATION_CLIP=True

    if VIS:
        # 创建两个坐标系
        mesh_frame_hand = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        mesh_frame_vrbase = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        mesh_frame_robotbase = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        mesh_frame_tcp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        mesh_frame_tcp_target = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])

        # 创建一个可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh_frame_hand)
        vis.add_geometry(mesh_frame_vrbase)
        vis.add_geometry(mesh_frame_robotbase)
        vis.add_geometry(mesh_frame_tcp)
        vis.add_geometry(mesh_frame_tcp_target)

    input("please put on the VR device to wake it up before continue")
    oculus_reader = OculusHandReader(reinstall=False)

    # Initialize Kalman Filters for position and orientation
    kf_position = KalmanFilter(dim_x=3, dim_z=3)
    kf_position.set_measurement_matrix(np.eye(3))
    kf_position.set_process_noise_covariance(0.01 * np.eye(3))
    kf_position.set_measurement_noise_covariance(0.2 * np.eye(3))

    kf_orientation = KalmanFilter(dim_x=3, dim_z=3)
    kf_orientation.set_measurement_matrix(np.eye(3))
    kf_orientation.set_process_noise_covariance(0.01 * np.eye(3))
    kf_orientation.set_measurement_noise_covariance(0.2 * np.eye(3))
    kf_orientation.set_initial_state(x=np.array([3, -0.4, 0.6]))

    dot_translation_pre = None
    rightGrip_pre = False
    while (True):
        time.sleep(TIME_STEP)
        ret = oculus_reader.get_transformations_and_buttons()
        # print(ret)
        try:
            rotmat_right = ret[0] # 4*4
            tmp = rotmat_right[:3,3]
            # print("hand translation:{}".format(rotmat_right[:3,3]))
            print("rotmat_right:\n{}".format(rotmat_right))
        except:
            print("oculus_reader.get_transformations_and_buttons failed!!!")
            continue

        buttons = ret[1]

        rightGrip = (buttons == "1")
        # print("right pinch:{}".format(rightGrip))

        dot_translation = rotmat_right[:3, -1]
        # Kalman filter update for position
        if USE_KALMAN_POS:
            kf_position.predict()
            kf_position.update(dot_translation)
            dot_translation_filtered = kf_position.x
            # print("dot_translation_filtered: {}".format(dot_translation_filtered))
            rotmat_right[:3, -1] = dot_translation_filtered
            dot_translation = rotmat_right[:3, -1]

        dot_translation_relative = np.zeros(3)

        tcp_current_pose = rtde_r.getActualTCPPose()  # xyz rx ry rz
        tcp_current_pose = np.array(tcp_current_pose)

        h_tcp2bot = np.eye(4)
        h_tcp2bot[:3,3] = tcp_current_pose[:3]
        # rot = rotation_vector_to_matrix([tcp_current_pose[3],tcp_current_pose[4],tcp_current_pose[5]])
        # rot = orthogonalize_rotation_matrix(rot)

        angle = math.sqrt(tcp_current_pose[3]**2 + tcp_current_pose[4]**2 + tcp_current_pose[5]**2)
        axis = tcp_current_pose[3:]
        rot = axangle2mat(axis=axis,angle=angle)
        h_tcp2bot[:3,:3] = rot

        # h_tcp2bot = urpose2homomatrix(tcp_current_pose)
        
        R_bot_dot = np.dot(rot_vr2bot, orthogonalize_rotation_matrix(rotmat_right[:3, :3]))
        # orientation = rotation_matrix_to_vector(R_bot_dot @ rot_tcp)
        R_tmp = R_bot_dot @ rot_tcp
        h_tmp = np.eye(4)
        h_tmp[:3,:3] = R_tmp
        orientation = np.asarray([3,0,0])
        # orientation = rotation_matrix_to_euler_angles_xyz(R_tmp)
        # orientation = ensure_vector_continuity(tcp_current_pose[3:], orientation)
        # orientation = rotation_matrix_to_vector(R_tmp)
        orientation = mat2rxryrz(h_tmp[:3,:3])

        orientation_filtered = orientation
        if USE_KALMAN_ORIENT:
            print("orientation before kalman and clip: {}".format(orientation))
            kf_orientation.predict()
            kf_orientation.update(orientation)
            orientation_filtered = kf_orientation.x
            

        if ORIENTATION_CLIP:
            orientation_standard = np.asarray([3,0,0])
            orientation_filtered_clipped = limit_orientation_change(orientation_standard, orientation_filtered, max_change=0.8)
            print("orientation after kalman and clip: {}".format(orientation_filtered_clipped))
            orientation_filtered = orientation_filtered_clipped
        

        if VIS:
            # 重置变换
            mesh_frame_robotbase.vertices = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6).vertices
            vis.update_geometry(mesh_frame_robotbase)

            mesh_frame_tcp.vertices = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5).vertices
            mesh_frame_tcp.transform(h_tcp2bot)
            vis.update_geometry(mesh_frame_tcp)

            mesh_frame_hand.vertices = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2).vertices
            mesh_frame_hand.transform(rotmat_right)
            # mesh_frame_hand.transform(h_vr2bot)
            o3d_left_multiply_transform(mesh_frame_hand,h_vr2bot)
            vis.update_geometry(mesh_frame_hand)

            # 同样重置和更新vrbase的变换
            mesh_frame_vrbase.vertices = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3).vertices
            mesh_frame_vrbase.transform(h_vr2bot)
            vis.update_geometry(mesh_frame_vrbase)

            vis.poll_events()
            vis.update_renderer()

        if (rightGrip and rightGrip_pre):
            print(rotmat_right)
            if dot_translation_pre is None:
                dot_translation_pre = dot_translation
            else:
                dot_translation_relative = (dot_translation - dot_translation_pre)
                dot_translation_relative = np.dot(rot_vr2bot, dot_translation_relative)
                dot_translation_relative_scaled = dot_translation_relative * scale
                print("dot_translation_relative (scaled): {}".format(dot_translation_relative_scaled))
                
                print("tcp_current_translation: {}".format(tcp_current_pose[:3]))
                print("tcp_current_orientation: {}".format(tcp_current_pose[3:]))
                dot_translation_relative_scaled = np.clip(dot_translation_relative_scaled, -0.1, 0.1)
                tcp_next_translation = (tcp_current_pose[:3] + dot_translation_relative_scaled)
                print("tcp_next_translation: {}".format(tcp_next_translation))

                # orientation = np.array([0,0,0])
                # tcp_pos_target = np.concatenate([tcp_next_translation, orientation], axis=-1)
                # tcp_pos_target = np.concatenate([tcp_current_pose[:3], orientation_filtered], axis=-1)
                tcp_pos_target = np.concatenate([tcp_current_pose[:3], orientation_filtered], axis=-1)

                if VIS:
                    mesh_frame_tcp_target.vertices = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4).vertices
                    
                    # h_tmp[:3,:3] = rxryrz2mat(tcp_pos_target[3:])
                    # h_tmp[:3,:3] = rxryrz2mat(orientation)
                    # h_tmp[:3,:3] = rxryrz2mat(orientation_filtered)
                    
                    h_tmp[:3,3] = tcp_pos_target[:3]
                    mesh_frame_tcp_target.transform(h_tmp)
                    vis.update_geometry(mesh_frame_tcp_target)
                    vis.poll_events()
                    vis.update_renderer()

                if MOVE_ROBOT:
                    vel = 0.05
                    acc = 0.1
                    dt = TIME_STEP
                    lookahead_time = 0.09
                    gain = 100
                    success = rtde_c.servoL(tcp_pos_target.tolist(), vel, acc, dt, lookahead_time, gain)
                    print(f"Move joints success: {success}")

                dot_translation_pre = dot_translation
        else:
            dot_translation_pre = None
        
        rightGrip_pre = rightGrip
        continue


if __name__ == "__main__":
    test()
