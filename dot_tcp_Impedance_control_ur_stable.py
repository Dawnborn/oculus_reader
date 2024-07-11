import os
import numpy as np
from oculus_reader.reader import OculusReader
import time

import rtde_control
import rtde_receive


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

    rot_vr2bot = np.array([-1, 0, 0, 0, 0, 1, 0, 1, 0]).reshape((3, 3))  # 眼镜与机器人基座的坐标系转换
    # rot_tcp = np.array([-1,0,0, 0,1,0, 0,0,-1]).reshape((3,3))
    rot_tcp = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape((3, 3))  # tcp与手柄的转换关系

    # scale = 1
    scale = 4
    # scale = 6 # servo
    # scale = 12
    # scale = 24
    # scale = 36
    # scale = 42
    # scale = 1

    vel_scale = 8
    # vel_scale = 3

    wait = True
    TIME_STEP = 0.02

    input("please put on the VR device to wake it up before continue")
    oculus_reader = OculusReader()

    dot_translation_pre = None
    while (True):
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

        if (buttons['A']):
            # tcp_pos_target = np.concatenate([home_position,home_orientation],axis=-1)
            # print("tcp_pos_target:{}".format(tcp_pos_target))
            # success = diana.move_tcp(
            #     tcp_pos=tcp_pos_target,
            #     vel=0.5,
            #     acc=0.5,
            #     wait=True
            # )
            success = rtde_c.moveJ(joints_pos_home.tolist(), asynchronous=False, speed=0.5, acceleration=0.5)

            print(f"Move joints home success: {success}")
            continue

        dot_translation = rotmat_right[:3, -1]
        dot_translation_relative = np.zeros(3)

        if (rightGrip > 0.5):
            print(rotmat_right)
            if dot_translation_pre is None:
                dot_translation_pre = dot_translation
            else:
                dot_translation_relative = (dot_translation - dot_translation_pre)
                dot_translation_relative = np.dot(rot_vr2bot, dot_translation_relative)
                dot_translation_relative_scaled = dot_translation_relative * scale
                print("dot_translation_relative (scaled): {}".format(dot_translation_relative_scaled))
                tcp_current_pose = rtde_r.getActualTCPPose()  # xyz rx ry rz
                print("tcp_current_translation: {}".format(tcp_current_pose[:3]))
                dot_translation_relative_scaled = np.clip(dot_translation_relative_scaled, -0.1, 0.1)
                tcp_next_translation = tcp_current_pose[:3] + dot_translation_relative_scaled
                print("tcp_next_translation: {}".format(tcp_next_translation))

                vel_raw = np.linalg.norm(dot_translation_relative)
                print("\n vel_raw: {} m/s \n".format(vel_raw))
                # if vel_raw < 0.001:
                #     print("not moving!")
                #     continue

                vel_raw *= vel_scale

                R_bot_dot = np.dot(rot_vr2bot, rotmat_right[:3, :3])
                orientation = rotation_matrix_to_euler_angles_xyz(R_bot_dot @ rot_tcp)
                print("orientation_dot {}".format(orientation / np.pi * 180))
                # orientation = np.array([-180,0,0])/180*np.pi # xyz

                orientation_pre = tcp_current_pose[3:6]
                orientation_diff = np.linalg.norm(orientation_pre - orientation)
                print("orientation_diff: {}".format(orientation_diff))
                vel_raw += orientation_diff

                vel = vel_raw
                # if vel_raw < 0.001:
                #     print("not moving!")
                #     continue

                if vel < 0.3:
                    vel = 0.3
                if vel > 0.5:
                    vel = 0.5
                print("\n vel: {} m/s \n".format(vel))
                print("\n orientation :{} \n".format(orientation))

                tcp_pos_target = np.concatenate([tcp_next_translation, orientation], axis=-1)
                # tcp_pos_target = np.concatenate([tcp_current_pose[:3], orientation], axis=-1)

                # success = diana.move_tcp(
                #     tcp_pos=tcp_pos_target,
                #     vel=vel,
                #     acc=2,
                #     wait=wait
                # )
                # success = rtde_c.servoL(pose=tcp_pos_target.tolist(), lookahead_time=0.09, time=TIME_STEP, gain=100.,
                #                         speed=0.5, acceleration=0.5)
                success = rtde_c.servoL(tcp_pos_target.tolist(), 0.5, 0.5, TIME_STEP,
                                        0.09, 100.0)
                print(f"Move joints success: {success}")

                dot_translation_pre = dot_translation
        else:
            dot_translation_pre = None

        continue


if __name__ == "__main__":
    test()
