import os
import numpy as np
from oculus_reader.reader_hand import OculusHandReader
import time
import math
import rtde_control
import rtde_receive
import open3d as o3d
from kalmanfilter import KalmanFilter
from LivePlot import LivePlotApp  # Import the class
from utils import axangle2mat, orthogonalize_rotation_matrix, mat2rxryrz, limit_orientation_change, o3d_left_multiply_transform

def test():
    rtde_c = rtde_control.RTDEControlInterface("10.3.15.95")
    rtde_r = rtde_receive.RTDEReceiveInterface("10.3.15.95")
    init_q = rtde_r.getActualQ() # [0.4281730651855469, -1.8017007313170375, -2.3542089462280273, 0.9464456278034667, -5.328896347676412, -0.42126590410341436]

    joints_pos_home = np.asarray(
        [0.4281730651855469, -1.8017007313170375, -2.3542089462280273, 0.9464456278034667, -5.328896347676412, -0.42126590410341436])


    tcp_current_pose = rtde_r.getActualTCPPose()
    rot_vr2bot = np.array([1,0,0,0,0,-1,0,1,0]).reshape((3, 3))
    rot_tcp = np.array([0,1,0, 0,0,1, 1,0,0]).reshape((3,3))
    # rot_tcp = np.eye(3)
    rot_tcp = np.array([[  0.0000000,  0.0000000,  1.0000000],
   [1.0000000,  0.0000000, -0.0000000],
  [-0.0000000,  1.0000000,  0.0000000 ]])

    h_vr2bot = np.eye(4)
    h_vr2bot[:3,:3] = rot_vr2bot
    h_vr2bot[0,3] = -1

    scale = 4
    TIME_STEP = 0.05
    VIS=True
    VIS_URVECTOR=True
    MOVE_ROBOT=True
    if MOVE_ROBOT:
        input("Warning: You are trying to move the real robot!!! Press Enter to continue...")
        rtde_c.moveJ(joints_pos_home, 1.05, 1.4, True)

    USE_KALMAN_POS=True
    USE_KALMAN_ORIENT=False
    ORIENTATION_CLIP=False

    if VIS:
        mesh_frame_hand = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        mesh_frame_vrbase = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        mesh_frame_robotbase = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        mesh_frame_tcp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        mesh_frame_tcp_target = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh_frame_hand)
        vis.add_geometry(mesh_frame_vrbase)
        vis.add_geometry(mesh_frame_robotbase)
        vis.add_geometry(mesh_frame_tcp)
        vis.add_geometry(mesh_frame_tcp_target)

    input("please put on the VR device to wake it up before continue")
    oculus_reader = OculusHandReader(reinstall=True,tag="xyzxyzw",APK_path="oculus_reader/APK_stable/posesample.apk")

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

    # Initialize the LivePlotApp for TCP position visualization
    if VIS_URVECTOR:
        live_plot = LivePlotApp(vector_length=3, max_points=100,timestep=TIME_STEP)

    while True:
        time.sleep(TIME_STEP)
        ret = oculus_reader.get_transformations_and_buttons()
        try:
            rotmat_right = ret[0]
            tmp = rotmat_right[:3,3]
        except:
            print("oculus_reader.get_transformations_and_buttons failed!!!")
            continue

        buttons = ret[1]
        rightGrip = (buttons == "1")

        dot_translation = rotmat_right[:3, -1]
        if USE_KALMAN_POS:
            kf_position.predict()
            kf_position.update(dot_translation)
            dot_translation_filtered = kf_position.x
            rotmat_right[:3, -1] = dot_translation_filtered
            dot_translation = rotmat_right[:3, -1]

        dot_translation_relative = np.zeros(3)

        tcp_current_pose = rtde_r.getActualTCPPose()
        tcp_current_pose = np.array(tcp_current_pose)

        h_tcp2bot = np.eye(4)
        h_tcp2bot[:3,3] = tcp_current_pose[:3]

        angle = math.sqrt(tcp_current_pose[3]**2 + tcp_current_pose[4]**2 + tcp_current_pose[5]**2)
        axis = tcp_current_pose[3:]
        rot = axangle2mat(axis=axis,angle=angle)
        h_tcp2bot[:3,:3] = rot

        R_bot_dot = np.dot(rot_vr2bot, orthogonalize_rotation_matrix(rotmat_right[:3, :3]))
        R_tmp = R_bot_dot @ rot_tcp
        h_tmp = np.eye(4)
        h_tmp[:3,:3] = R_tmp
        orientation = np.asarray([3,0,0])
        orientation = mat2rxryrz(h_tmp[:3,:3])

        orientation_filtered = orientation
        if USE_KALMAN_ORIENT:
            kf_orientation.predict()
            kf_orientation.update(orientation)
            orientation_filtered = kf_orientation.x

        if ORIENTATION_CLIP:
            orientation_standard = np.asarray([3,0,0])
            orientation_filtered_clipped = limit_orientation_change(orientation_standard, orientation_filtered, max_change=0.8)
            orientation_filtered = orientation_filtered_clipped

        if VIS:
            mesh_frame_robotbase.vertices = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6).vertices
            vis.update_geometry(mesh_frame_robotbase)

            mesh_frame_tcp.vertices = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5).vertices
            mesh_frame_tcp.transform(h_tcp2bot)
            vis.update_geometry(mesh_frame_tcp)

            mesh_frame_hand.vertices = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2).vertices
            mesh_frame_hand.transform(rotmat_right)
            o3d_left_multiply_transform(mesh_frame_hand,h_vr2bot)
            vis.update_geometry(mesh_frame_hand)

            mesh_frame_vrbase.vertices = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3).vertices
            mesh_frame_vrbase.transform(h_vr2bot)
            vis.update_geometry(mesh_frame_vrbase)

            vis.poll_events()
            vis.update_renderer()

        if rightGrip and rightGrip_pre:
            if dot_translation_pre is None:
                dot_translation_pre = dot_translation
            else:
                dot_translation_relative = (dot_translation - dot_translation_pre)
                dot_translation_relative = np.dot(rot_vr2bot, dot_translation_relative)
                dot_translation_relative_scaled = dot_translation_relative * scale
                dot_translation_relative_scaled = np.clip(dot_translation_relative_scaled, -0.1, 0.1)
                tcp_next_translation = (tcp_current_pose[:3] + dot_translation_relative_scaled)

                # tcp_pos_target = np.concatenate([tcp_current_pose[:3], orientation_filtered], axis=-1)
                tcp_pos_target = np.concatenate([tcp_next_translation, orientation_filtered], axis=-1)

                # Update the live plot with the new TCP position
                if VIS_URVECTOR:
                    live_plot.update(orientation_filtered.tolist())

                if VIS:
                    mesh_frame_tcp_target.vertices = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4).vertices
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

if __name__ == "__main__":
    test()
