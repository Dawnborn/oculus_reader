import os
os.sys.path.append("/home/jh/Documents/ws_diana_yunlong/diana_robot_driver_dev-main")
from diana_contol import DianaControl, rotation_matrix_to_euler_angles
import numpy as np
from oculus_reader.reader import OculusReader
import time

from pyRobotiqGripper import RobotiqGripper

'''
    The configuration of impedance is according to the TCP axis, you still can control the robot for both tcp and joints.
    And trajectory seems will end by fixed time, or fixed number of the waypoints, it will not check if the goal if reached.
'''
def test():

    gripper = RobotiqGripper()
    gripper.activate()

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

    # success = diana.set_cartesian_impedance(stiffness=(100, 100, 100, 100, 100, 100), damping=[0.5])
    # print(f"Set impedance:{success}")
    # success = diana.change_control_mode("cartesian_impedance")
    # print(f"change control mode:{success}")
    success = diana.change_control_mode("position")
    print(f"change control mode:{success}")

    # home_position = np.array([-200,400,800])/1000
    # home_orientation = np.array([0,10,0])/180*np.pi
    # tcp_pos_target = np.concatenate([home_position,home_orientation],axis=-1)
    # print("moving to home pose {}...........".format(tcp_pos_target))
    # success = diana.move_tcp(
    #     tcp_pos=tcp_pos_target,
    #     vel=0.2,
    #     acc=1,
    #     wait=True
    # )

    joints_pos_home = np.asarray([0.20555325057509052, -0.011840602165761105, -0.30533428759153086, 1.6415630854522976, 0.0369351961063038, -1.3390572623605348, 0.15645387821129342])
    joints_pos_home = np.asarray([-1.1570767963258144, 0.11702577592257235, -1.0692687497175277, 1.8052795916426385, 0.17278837773090672, -1.200028260946719, -1.4878777506126735])
    # success = diana.move_joints(joints_pos_home,wait=True, vel=0.5)

    rot_vr2bot = np.array([-1,0,0, 0,0,1, 0,1,0]).reshape((3,3)) # 眼镜与机器人基座的坐标系转换
    # rot_tcp = np.array([-1,0,0, 0,1,0, 0,0,-1]).reshape((3,3))
    rot_tcp = np.array([1,0,0, 0,-1,0, 0,0,-1]).reshape((3,3)) # tcp与手柄的转换关系
    
    scale = 1
    scale = 2
    scale = 4
    # scale = 6 # servo
    # scale = 12
    # scale = 24
    # scale = 36
    # scale = 42

    vel_scale = 8
    # vel_scale = 3
    
    wait = True
    TIME_STEP=1./50

    input("please put on the VR device to wake it up before continue")
    oculus_reader = OculusReader()

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
            # tcp_pos_target = np.concatenate([home_position,home_orientation],axis=-1)
            # print("tcp_pos_target:{}".format(tcp_pos_target))
            # success = diana.move_tcp(
            #     tcp_pos=tcp_pos_target,
            #     vel=0.5,
            #     acc=0.5,
            #     wait=True
            # )
            success = diana.move_joints(joints_pos_home,wait=True, vel=0.5, acc=0.5)
            print(f"Move joints home success: {success}")
            continue

        if(buttons['B']):
            print("quiting....")
            diana.stop()
            print("return")
            return

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
                print("dot_translation_relative: {}".format(dot_translation_relative))
                tcp_current_pose = diana.get_tcp_pos()
                print("tcp_current_translation: {}".format(tcp_current_pose[:3]))
                tcp_next_translation = tcp_current_pose[:3] + dot_translation_relative*scale
                print("tcp_next_translation: {}".format(tcp_next_translation))

                vel_raw = np.linalg.norm(dot_translation_relative)
                print("\n vel_raw: {} m/s \n".format(vel_raw))
                # if vel_raw < 0.001:
                #     print("not moving!")
                #     continue

                vel_raw *= vel_scale

                
                R_bot_dot = np.dot(rot_vr2bot,rotmat_right[:3,:3])
                orientation = rotation_matrix_to_euler_angles(R_bot_dot@rot_tcp)
                print("orientation_dot {}".format(orientation/np.pi*180))
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
                if vel>0.5:
                    vel = 0.5
                print("\n vel: {} m/s \n".format(vel))
                print("\n orientation :{} \n".format(orientation))

                tcp_pos_target = np.concatenate([tcp_next_translation,orientation],axis=-1)

                # success = diana.move_tcp(
                #     tcp_pos=tcp_pos_target,
                #     vel=vel,
                #     acc=2,
                #     wait=wait
                # )

                # success = diana.move_tcp_servoL(tcp_pos=tcp_pos_target, t=TIME_STEP, gain=600, realiable=True)
                # print(f"Move joints success: {success}")
                
                joints = diana.inverse_kinematics_ext(tcp_pos_target)
                if not (joints is None):
                    success = diana.move_tcp_servoJ(joints=joints, time=TIME_STEP, gain=200, look_ahead_time=0.2)
                    print(f"Move joints success: {success}")
                else:
                    print("inverse fail!!!")

                dot_translation_pre = dot_translation
        else:
            dot_translation_pre = None
        
        continue

if __name__ == "__main__":
    test()