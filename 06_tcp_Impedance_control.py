import os
os.sys.path.append("/home/jh/Documents/ws_diana_yunlong/diana_robot_driver_dev-main")
from diana_contol import DianaControl, rotation_matrix_to_euler_angles
import numpy as np
from oculus_reader.reader import OculusReader
import time

'''
    The configuration of impedance is according to the TCP axis, you still can control the robot for both tcp and joints.
    And trajectory seems will end by fixed time, or fixed number of the waypoints, it will not check if the goal if reached.
'''
def test():
    ip_addr = "192.168.10.75"
    diana = DianaControl(ip_address=ip_addr)
    # diana.set_log_level('DEBUG')
    
    # time.sleep(2)
    # get the current joint positions, velocities and torques
    # joints_pos = diana.get_joint_pos()
    # joints_vel = diana.get_joint_vel()
    # joints_torque = diana.get_joint_torque()
    # print("Joints pos: ", joints_pos)
    # print("Joints vel: ", joints_vel)
    # print("Joints torque: ", joints_torque)
    # get the current TCP position
    tcp_pos = diana.get_tcp_pos()
    print("TCP pos: ", tcp_pos)

    success = diana.set_cartesian_impedance(stiffness=(100, 100, 100, 100, 100, 100), damping=[0.5])
    print(f"Set impedance:{success}")
    success = diana.change_control_mode("cartesian_impedance")
    print(f"change control mode:{success}")

    # position = np.array([400,-400,200])/1000
    home_position = np.array([-200,400,800])/1000
    # oritation = np.array([10,-6,-50])/180*np.pi
    home_oritation = np.array([180,10,0])/180*np.pi
    tcp_pos_target = np.concatenate([home_position,home_oritation],axis=-1)
    print("moving to home pose...........")
    success = diana.move_tcp(
        tcp_pos=tcp_pos_target,
        vel=0.3,
        acc=1,
        wait=True
    )

    rot_vr2bot = np.array([-1,0,0, 0,0,1, 0,1,0]).reshape((3,3)) # 眼镜与机器人基座的坐标系转换
    rot_tcp = np.array([-1,0,0, 0,1,0, 0,0,-1]).reshape((3,3))
    scale = 1
    vel_scale = 10
    # vel_scale = 6

    wait = True

    input("please put on the VR device to wake it up before continue")
    oculus_reader = OculusReader()

    dot_translation_pre = None
    while(True):
        time.sleep(0.1)
        ret = oculus_reader.get_transformations_and_buttons()
        try:
            rotmat_right = ret[0]['r']
        except:
            print("oculus_reader.get_transformations_and_buttons failed!!!")
            continue

        buttons = ret[1]

        rightGrip = buttons['rightGrip'][0]

        if(buttons['A']):
            tcp_pos_target = np.concatenate([home_position,home_oritation],axis=-1)
            success = diana.move_tcp(
                tcp_pos=tcp_pos_target,
                vel=0.5,
                acc=1,
                wait=True
            )
            print(f"Move joints home success: {success}")
            continue

        if(buttons['B']):
            print("quiting....")
            diana.stop()
            print("return")
            return

        dot_translation = rotmat_right[:3,-1]
        dot_translation_relative = np.zeros(3)
        if(rightGrip>0.5):
            print(rotmat_right)
            if dot_translation_pre is None:
                dot_translation_pre = dot_translation
            else:
                dot_translation_relative = (dot_translation - dot_translation_pre)
                dot_translation_relative = np.dot(rot_vr2bot, dot_translation_relative)
                print("dot_translation_relative: {}".format(dot_translation_relative))
                tcp_current_pose = diana.get_tcp_pos()
                tcp_next_translation = tcp_current_pose[:3] + dot_translation_relative*scale
                print("tcp_next_translation: {}".format(tcp_next_translation))

                vel_raw = np.linalg.norm(dot_translation_relative)
                print("\n vel_raw: {} m/s \n".format(vel_raw))
                if vel_raw < 0.001:
                    print("not moving!")
                    continue

                vel_raw *= vel_scale

                
                R_bot_dot = np.dot(rot_vr2bot,rotmat_right[:3,:3])
                oritation = rotation_matrix_to_euler_angles(rot_tcp@R_bot_dot)
                oritation = np.array([-180,0,0])/180*np.pi # xyz

                oritation_pre = tcp_current_pose[3:6]
                oritation_diff = np.max(oritation_pre - oritation)
                vel_raw += oritation_diff

                vel = vel_raw if vel_raw < 0.5 else 0.5
                vel = vel_raw if vel_raw > 0.1 else 0.1
                print("\n vel: {} m/s \n".format(vel))
                print("\n oritation :{} \n".format(oritation))
                tcp_pos_target = np.concatenate([tcp_next_translation,oritation],axis=-1)
                success = diana.move_tcp(
                    tcp_pos=tcp_pos_target,
                    vel=vel,
                    acc=2,
                    wait=wait
                )
                print(f"Move joints success: {success}")
                dot_translation_pre = dot_translation
        else:
            dot_translation_pre = None
        
        continue

if __name__ == "__main__":
    test()