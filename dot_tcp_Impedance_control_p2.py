import numpy as np
import math
import time
import yaml

from oculus_reader.reader import OculusReader

import open3d as o3d

import argparse

from diana_control_p2 import DianaControl

from kalmanfilter import KalmanFilter
from utils import mat2rxryrz, o3d_left_multiply_transform, axangle2mat, orthogonalize_rotation_matrix

IP_ADDRESS_LEFT = "192.168.100.85"
IP_ADDRESS_RIGHT = "192.168.100.63"

MOVE_ROBOT = True
VIS = True

# DianaApi.initSrv((IP_ADDRESS_RIGHT, 0, 0, 0, 0, 0))

# tcp_pose = np.zeros(6)
# DianaApi.getJointPos(tcp_pose,IP_ADDRESS_RIGHT)

# print(tcp_pose)
# import pdb
# pdb.set_trace()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description='parameters.')

    # 添加配置文件路径参数
    parser.add_argument('--config', type=str, default="config/dot_tcp_diana_v3.yaml", help='Path to the configuration file')

    args = parser.parse_args()

    # 加载配置文件
    config = load_config(args.config)

    scale = config['scale'] # servo
    threshold = config['threshold']

    TIME_STEP=config['time_step']
    USE_KALMAN=config['use_kalman']

    process_noise_covariance=config['process_noise_covariance']
    measurement_noise_covariance=config['measurement_noise_covariance']

    clip_threshold=config['clip_threshold']

    ahead_time=config['ahead_time']
    gain=config['gain']

    # ip_addr = config['ip_addr']
    ip_addr = IP_ADDRESS_RIGHT
    
    diana = DianaControl(ip_address=ip_addr)
    # diana.set_log_level('DEBUG')
    
    tcp_pos = diana.get_tcp_pos()
    print("TCP pos: ", tcp_pos)

    success = diana.change_control_mode("position")
    print(f"change control mode:{success}")

    joints_pos_home = np.asarray([32, -33, -11, 133, 3.9, -67.6, -25.9])/180*np.pi
    # joints_pos_home = np.asarray([ 0.80264351,  0.21575201, -1.45688636,  3.99719464,  1.78378008,  1.01457256,  0.])
    joints_pos_home = np.array([ 0.80254764, 0.21581193, -1.45675453, 3.65444579, 1.78207832, 1.01561518, 0.])

    success = diana.move_joints(joints_pos_home,wait=True, vel=0.5)

    rot_vr2bot = np.array([-1,0,0, 0,0,1, 0,1,0]).reshape((3,3)) # 眼镜与机器人基座的坐标系转换
    # rot_tcp = np.array([-1,0,0, 0,1,0, 0,0,-1]).reshape((3,3))
    rot_tcp = np.array([1,0,0, 0,-1,0, 0,0,-1]).reshape((3,3)) # tcp与手柄的转换关系

    h_vr2bot = np.eye(4)
    h_vr2bot[:3,:3] = rot_vr2bot
    h_vr2bot[0,3] = -1
    
    input("please put on the VR device to wake it up before continue")
    oculus_reader = OculusReader()

    kf_position = KalmanFilter(dim_x=3, dim_z=3)
    kf_position.set_measurement_matrix(np.eye(3))
    kf_position.set_process_noise_covariance(process_noise_covariance * np.eye(3))
    kf_position.set_measurement_noise_covariance(measurement_noise_covariance * np.eye(3))

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

    dot_translation_pre = None
    while(True):
        time.sleep(TIME_STEP)
        ret = oculus_reader.get_transformations_and_buttons()
        try:
            rotmat_right = ret[0]['r']
            assert(np.linalg.det(rotmat_right[:3,:3])>0.8)
                
        except:
            print("oculus_reader.get_transformations_and_buttons failed!!!")
            continue

        buttons = ret[1]

        rightGrip = buttons['rightGrip'][0]
        rightTrig = buttons['rightTrig'][0]

        if(buttons['A']):
 
            success = diana.move_joints(joints_pos_home,wait=True, vel=0.5, acc=0.5)
            print(f"Move joints home success: {success}")
            continue

        if(buttons['B']):
            success = diana.move_joints(joints_pos_home,wait=True, vel=0.5, acc=0.5)
            print(f"Move joints home comfortable success: {success}")
            continue

        tcp_current_pose = diana.get_tcp_pos()[:6]

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

        if USE_KALMAN:
            kf_position.predict()
            kf_position.update(tcp_current_pose[:3])
            tcp_current_pose[:3] = kf_position.x

        dot_translation = rotmat_right[:3,-1]
        dot_translation_relative = np.zeros(3)

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

        if(rightGrip>0.5):
            print(rotmat_right)
            if dot_translation_pre is None:
                dot_translation_pre = dot_translation
            else:
                dot_translation_relative = (dot_translation - dot_translation_pre)
                dot_translation_relative = np.dot(rot_vr2bot, dot_translation_relative)
                dot_translation_relative_scaled = dot_translation_relative*scale
                print("dot_translation_relative (unscaled): {}".format(dot_translation_relative))
                print("dot_translation_relative (scaled): {}".format(dot_translation_relative_scaled))
                print("tcp_current_translation: {}".format(tcp_current_pose[:3]))
                dot_translation_relative_scaled = np.clip(dot_translation_relative_scaled, -clip_threshold, clip_threshold)
                tcp_next_translation = tcp_current_pose[:3] + dot_translation_relative_scaled
                print("tcp_next_translation: {}".format(tcp_next_translation))

                translation_norm_raw = np.linalg.norm(dot_translation_relative)

                orientation = np.array([157,0,0])
                tcp_pos_target = np.concatenate([tcp_next_translation,orientation],axis=-1)

                if VIS:
                    mesh_frame_tcp_target.vertices = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4).vertices
                    h_tmp[:3,3] = tcp_pos_target[:3]
                    mesh_frame_tcp_target.transform(h_tmp)
                    vis.update_geometry(mesh_frame_tcp_target)
                    vis.poll_events()
                    vis.update_renderer()

                if MOVE_ROBOT:
                    if translation_norm_raw < threshold:
                    # if True:
                        print("small movement")
                        success = diana.move_tcp_servoL(tcp_pos=tcp_pos_target, ah_t=ahead_time, t=TIME_STEP, gain=gain, realiable=True)
                        # def servoL(tcp_pose, t=0.01, ah_t=0.01, gain=300, scale=1000, ipAddress=''):
                        # success2 = DianaApi.servoL(tcp_pos=tcp_pos_target, ah_t=ahead_time, t=TIME_STEP, gain=gain, ipAddress=IP_ADDRESS_RIGHT)
                    else:
                        # success2 = DianaApi.servoL(tcp_pos=tcp_pos_target, ah_t=ahead_time, t=TIME_STEP, gain=gain, ipAddress=IP_ADDRESS_RIGHT)
                        success = diana.move_tcp_servoL(tcp_pos=tcp_pos_target, ah_t=ahead_time, t=TIME_STEP, gain=gain, realiable=True)
                    print(f"Move joints success: {success}")

                dot_translation_pre = dot_translation
        else:
            dot_translation_pre = None
        
        continue

if __name__ == "__main__":
    main()