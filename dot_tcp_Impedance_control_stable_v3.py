import os
os.sys.path.append("/home/jh/Documents/ws_diana_yunlong/diana_robot_driver_dev-main")
from diana_contol import DianaControl
import numpy as np
from oculus_reader.reader import OculusReader
import time

from pyRobotiqGripper import RobotiqGripper
from utils import mat2rxryrz

from kalmanfilter import KalmanFilter

import argparse
import yaml

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

    ip_addr = config['ip_addr']


    gripper = RobotiqGripper()
    gripper.activate()
    
    diana = DianaControl(ip_address=ip_addr)
    # diana.set_log_level('DEBUG')
    
    tcp_pos = diana.get_tcp_pos()
    print("TCP pos: ", tcp_pos)

    success = diana.change_control_mode("position")
    print(f"change control mode:{success}")

    joints_pos_home = np.asarray([32, -33, -11, 133, 3.9, -67.6, -25.9])/180*np.pi
    
    success = diana.move_joints(joints_pos_home,wait=True, vel=0.5)

    rot_vr2bot = np.array([-1,0,0, 0,0,1, 0,1,0]).reshape((3,3)) # 眼镜与机器人基座的坐标系转换
    # rot_tcp = np.array([-1,0,0, 0,1,0, 0,0,-1]).reshape((3,3))
    rot_tcp = np.array([1,0,0, 0,-1,0, 0,0,-1]).reshape((3,3)) # tcp与手柄的转换关系
    
    input("please put on the VR device to wake it up before continue")
    oculus_reader = OculusReader()

    kf_position = KalmanFilter(dim_x=3, dim_z=3)
    kf_position.set_measurement_matrix(np.eye(3))
    kf_position.set_process_noise_covariance(process_noise_covariance * np.eye(3))
    kf_position.set_measurement_noise_covariance(measurement_noise_covariance * np.eye(3))

    dot_translation_pre = None
    while(True):
        time.sleep(TIME_STEP)
        ret = oculus_reader.get_transformations_and_buttons()
        try:
            rotmat_right = ret[0]['r']
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

        tcp_current_pose = diana.get_tcp_pos()

        if USE_KALMAN:
            kf_position.predict()
            kf_position.update(tcp_current_pose[:3])
            tcp_current_pose[:3] = kf_position.x

        dot_translation = rotmat_right[:3,-1]
        dot_translation_relative = np.zeros(3)
        gripper.goTo(int(rightTrig*255),force=50)
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
                
                R_bot_dot = np.dot(rot_vr2bot,rotmat_right[:3,:3])
                # orientation = rotation_matrix_to_euler_angles(R_bot_dot@rot_tcp)
                orientation = mat2rxryrz(R_bot_dot@rot_tcp)

                orientation_pre = tcp_current_pose[3:6]

                tcp_pos_target = np.concatenate([tcp_next_translation,orientation],axis=-1)

                if translation_norm_raw < threshold:
                # if True:
                    print("small movement")
                    success = diana.move_tcp_servoL(tcp_pos=tcp_pos_target, ah_t=ahead_time, t=TIME_STEP, gain=gain, realiable=True)
                else:
                    success = diana.move_tcp_servoL(tcp_pos=tcp_pos_target, ah_t=ahead_time, t=TIME_STEP, gain=gain, realiable=True)
                print(f"Move joints success: {success}")

                dot_translation_pre = dot_translation
        else:
            dot_translation_pre = None
        
        continue

if __name__ == "__main__":
    main()