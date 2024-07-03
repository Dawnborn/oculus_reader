import numpy as np
import open3d as o3d

import rtde_control
import rtde_receive

def homo_matrix_prepare5(ur_pose):
    """
    Convert a pose in the form of a 6-element array to a 4x4 homogeneous transformation matrix.
    
    Parameters:
    ur_pose (np.array): A 6-element array containing px, py, pz, rx, ry, rz

    Returns:
    np.array: A 4x4 homogeneous transformation matrix
    """
    px, py, pz, rx, ry, rz = ur_pose
    
    angle = np.sqrt(rx**2 + ry**2 + rz**2)
    
    if angle == 0:
        return np.eye(4)
    
    kx = rx / angle
    ky = ry / angle
    kz = rz / angle

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    one_minus_cos = 1 - cos_angle

    # Initialize the homogeneous transformation matrix
    homo_matrix = np.zeros((4, 4))

    # Fill in the rotation part of the matrix
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

    homo_matrix[3, 0] = 0
    homo_matrix[3, 1] = 0
    homo_matrix[3, 2] = 0
    homo_matrix[3, 3] = 1

    return homo_matrix

if __name__ == "__main__":
    rtde_c = rtde_control.RTDEControlInterface("10.3.15.95")
    rtde_r = rtde_receive.RTDEReceiveInterface("10.3.15.95")
    init_q = rtde_r.getActualQ()  # [0.26260805130004883, -1.2464478772929688, -1.9548711776733398, -1.4632833761027833, 1.7008233070373535, -4.365320030842916]

    mesh_frame_robotbase = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    mesh_frame_tcp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    mesh_frame_tcp_target = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])

    # 创建一个可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh_frame_robotbase)
    vis.add_geometry(mesh_frame_tcp)
    vis.add_geometry(mesh_frame_tcp_target)

    tcp_current_pose = rtde_r.getActualTCPPose()  # xyz rx ry rz