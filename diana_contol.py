import logging
import os
import signal
import sys
from ctypes import cdll
from pathlib import Path

import numpy as np

# ======================================================================================================================

# Determine the absolute path to the directory containing agile_robot.py
current_directory = Path(os.path.abspath(os.path.dirname(__file__)))
# Build the path to DianaApi.py
diana_api_path = current_directory / "DianaApi-x86_64-x64-linux/DianaApi/bin"
# Append the path to the system path for Python imports
sys.path.append(str(diana_api_path))

libs_to_load = ["libxml2.so", "libBasicSdk.so", "libToolSdk.so", "libGenericAlgorithm.so", "libToolSdk.so",
                ]  # List all libs you want to load
for lib in libs_to_load:
    lib_path = diana_api_path / lib
    cdll.LoadLibrary(str(lib_path))

os.environ['LD_LIBRARY_PATH'] = str(diana_api_path) + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import DianaApi

def to_tuple(data):
    if isinstance(data, list):
        data = tuple(data)
    if isinstance(data, np.ndarray):
        data = tuple(data.tolist())
    return data

def to_list(data):
    if isinstance(data, np.ndarray):
        data = data.tolist()
    return data

def homogeneous2Pose(homogeneous):
    homogeneous = to_tuple(homogeneous)
    pose = [0]*6
    success = DianaApi.homogeneous2Pose(homogeneous,pose)
    if success:
        return pose
    else:
        return None


def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Euler angles (ZYX order).
    
    Parameters:
    R (np.array): 3x3 rotation matrix

    Returns:
    tuple: Euler angles (alpha, beta, gamma) in radians
    """
    # Check if the matrix is a valid rotation matrix
    if not (np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-4, rtol=1e-4) and np.isclose(np.linalg.det(R), 1.0)):
        raise ValueError("The provided matrix is not a valid rotation matrix")

    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])  # ZYX order

class DianaControl():
    def __init__(self, ip_address):
        self.robot_api = DianaApi
        self.ip_address = ip_address
        self.logger = logging.getLogger(__name__)
        self.tcp_transform = np.eye(4)
        self.tcp_pos = [0.0] * 7
        self.tcp_vel = [0.0] * 7
        self.tcp_acc = [0.0] * 7
        self.joints_pos = [0.0] * 7
        self.joints_vel = [0.0] * 7
        self.joints_torque = [0.0] * 7
        self.path_ids = []
        self.free_drive_on = False
        self.control_mode = "position"
        self.init()

    def reset_control_mode(self):
        self.control_mode = "position"
        self.change_control_mode("position")

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C! Safety terminate the code!')
        self.close_free_drive()
        self.stop()
        self.reset_control_mode()
        sys.exit(0)

    def set_log_level(self, level='INFO'):
        """
        Set the log level for the logger
        :param level: one in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        :return:
        """
        assert level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        log_level = getattr(logging, level)
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            encoding='utf-8',
            level=log_level,
            datefmt='%Y-%m-%d %H:%M:%S')

    def init(self):
        netInfo = (self.ip_address, 0, 0, 0, 0, 0)
        self.robot_api.initSrv(netInfo)
        signal.signal(signal.SIGINT, self.signal_handler)
        self.free_drive_on = False

    def get_joint_pos(self):
        """
        Get the current joint positions of the robot
        :return: tuple of joint positions in radians
        """
        success = self.robot_api.getJointPos(self.joints_pos, self.ip_address)
        if success:
            return self.joints_pos
        else:
            self.logger.error("Failed to get joint positions.")
            return None

    def get_joint_vel(self):
        """
        Get the current joint velocities of the robot
        :return: tuple of joint angle velocities in radians per second
        """
        success = self.robot_api.getJointAngularVel(self.joints_vel, self.ip_address)
        if success:
            return self.joints_vel
        else:
            self.logger.error("Failed to get joint velocities.")
            return None

    def get_joint_torque(self):
        """
        Get the current joint accelerations of the robot
        :return: tuple of joint accelerations in radians per second squared
        """
        success = self.robot_api.getJointTorque(self.joints_torque, self.ip_address)
        if success:
            return self.joints_torque
        else:
            self.logger.error("Failed to get joint accelerations.")
            return None

    def move_joints(self, joints, vel=0.1, acc=0.1, zv_shaper_order=0, zv_shaper_frequency=0, zv_shaper_damping_ratio=0, wait=True):
        """
        Move the robot to the specified joint positions
        :param joints: the joint positions to move to, in radian
        :param vel: radians per second
        :param acc: radians per second squared
        :param zv_shaper_order:
        :param zv_shaper_frequency:
        :param zv_shaper_damping_ratio:
        :param wait: if True, wait for the move to complete before returning
        :return:
        """
        joints = to_tuple(joints)
        success = self.robot_api.moveJToTarget(joints, vel, acc, zv_shaper_order, zv_shaper_frequency, zv_shaper_damping_ratio, self.ip_address)
        if wait:
            self.robot_api.wait_move()
        if not success:
            self.logger.error("Failed to move the robot to the specified joints. Please check if the joint value is radian.")
        return success

    def forward_kinematics(self, joints:tuple):
        """
        Compute the forward kinematics of the robot
        :param joints: the joint positions to compute the forward kinematics for, in radians
        :return: the TCP position in meters
        """
        tcp_pos = [0.0] * 7
        success = self.robot_api.forward(joints, tcp_pos, self.ip_address)
        if success:
            return tcp_pos
        else:
            self.logger.error("Failed to compute forward kinematics.")
            return None

    def inverse_kinematics(self, tcp_pos, ref_joints=None):
        """
        Compute the backward kinematics of the robot
        :param tcp_pos: the TCP position to compute the backward kinematics for, in meters
        :return: the joint positions in radians
        """
        joints = [0.0] * 7
        if ref_joints is None:
            success = self.robot_api.inverse(tcp_pos, joints, self.ip_address)
        else:
            success = self.robot_api.inverse_ext(ref_joints, tcp_pos, joints, ipAddress=self.ip_address)
        if success:
            return joints
        else:
            self.logger.error("Failed to compute backward kinematics. Please check if position in meter, and orientation in radians")
            return None

    def set_push_period(self, period=10):
        """
        Set the period of the push
        :param period: the period of the push in ms
        :return:
        """
        self.robot_api.setPushPeriod(period, self.ip_address)

    def get_tcp_pos(self):
        """
        Get the current TCP position of the robot
        :return: tuple of TCP position in meters, (x, y, z, rx, ry, rz)
        """
        success = self.robot_api.getTcpPos(self.tcp_pos, self.ip_address)
        if success:
            return self.tcp_pos
        else:
            self.logger.error("Failed to get TCP position.")
            return None

    def get_tcp_force(self):
        """
        Get the current TCP force of the robot according to the current TCP pose
        :return: TCP force, (fx, fy, fz, mx, my, mz) the force and torque of each axis
        """
        tcp_force = [0.0] * 6
        self.robot_api.getTcpForce(tcp_force, self.ip_address)
        return tcp_force

    def move_tcp(self, tcp_pos, vel, acc, zv_shaper_order=0, zv_shaper_frequency=0, zv_shaper_damping_ratio=0, wait=True):
        """
        Move the robot to the specified TCP position
        :param wait: if True, wait for the move to complete before returning
        :param tcp_pos: the TCP position to move to, in meters
        :param vel: meters per second
        :param acc: meters per second squared
        :param zv_shaper_order:
        :param zv_shaper_frequency:
        :param zv_shaper_damping_ratio:
        :return:
        """
        tcp_pos = to_tuple(tcp_pos)
        success = self.robot_api.moveJToPose(tcp_pos, vel, acc, zv_shaper_order, zv_shaper_frequency, zv_shaper_damping_ratio, self.ip_address)
        if wait:
            self.robot_api.wait_move()
        if not success:
            self.logger.error("Failed to move to the TCP position.")
        return success
    
    def move_tcp_speed(self,speeds,acc, t):
        speeds = to_tuple(speeds)
        acc = to_tuple(acc)
        success = self.robot_api.speedL(speeds,acc,t,self.ip_address)
        return success

    def move_tcp_servoL(self, tcp_pos, t=0.01, ah_t=0.2, scale=1.0, gain=300, realiable=True):
        """
        默认参数对于100hz 0.001m下比较平滑

        pose：基坐标系下的目标位姿数组的首地址，数组长度为 6。前三个元素单位：m；后三
        个元素单位：rad，注意，后三个角度需要是轴角。 
        time：运动时间。单位：s。 
        ah_time: look_ahead_time：时间（S），范围（0.03-0.2）大参数使轨迹更平滑。 
        gain：目标位置的比例放大器，范围（100,2000）。  
        scale：平滑比例系数。范围（0.0~1.0）。  
        active_tcp：需要移动的工具中心点对应的位姿向量（基于法兰坐标系），大小为 6 的数
        组，为空时将移动系统当前工具的中心点至pose。（注意：此处active_tcp 作为工具） 
        strIpAddress：可选参数，需要控制机械臂的IP 地址字符串，不填仅当只连接一台机械臂
        时生效。
        """
        tcp_pos = to_tuple(tcp_pos)
        success = self.robot_api.servoL_ex(tcp_pose=tcp_pos, t=t, ah_t=ah_t, gain=gain, scale=scale, ipAddress=self.ip_address,realiable=realiable)
        # self.robot_api.wait_move()
        return success
    
    def move_tcp_servoJ(self, joints, time=0.01, look_ahead_time=0.03, gain=300, reliable=False):
        joints = to_list(joints)
        success = self.robot_api.servoJ(joints_pos=joints, t=time, ah_t=look_ahead_time, gain=gain, ipAddress=self.ip_address)
        return success

    def open_free_drive(self, mode="normal"):
        """
        Open the free drive mode.
            E_DISABLE_FREEDRIVING = 0  # Disable free driving
            E_NORMAL_FREEDRIVING = 1  # Normal free driving
            E_FORCE_FREEDRIVING = 2  # Force free driving

        :return:
        """

        if mode == "normal":
            self.robot_api.freeDriving(self.robot_api.freedriving_mode_e.E_NORMAL_FREEDRIVING, self.ip_address)
            self.free_drive_on = True
        elif mode == "force":
            self.robot_api.freeDriving(self.robot_api.freedriving_mode_e.E_FORCE_FREEDRIVING, self.ip_address)
            self.free_drive_on = True
        else:
            self.logger.warning("Invalid free drive mode. Please choose from the following modes: ['normal', 'force']")

    def close_free_drive(self):
        if self.free_drive_on:
            self.robot_api.freeDriving(self.robot_api.freedriving_mode_e.E_DISABLE_FREEDRIVING, self.ip_address)
        else:
            print("Free drive is already off.")

    def release_brake(self):
        """
        Release the brake of the robot
        :return:
        """
        self.robot_api.releaseBrake(self.ip_address)

    def hold_brake(self):
        """
        Hold the brake of the robot
        :return:
        """
        self.robot_api.holdBrake(self.ip_address)

    def set_joint_impedance(self, stiffness=(3000, 3000, 3000, 1000, 500, 1000, 1000), damping=[0.5]):
        """
        Set the joint impedance of the robot
        :param stiffness: the stiffness of the joint
        :param damping: the damping of the joint
        :param inertia: the inertia of the joint
        :return:
        """
        success = self.robot_api.setJointImpeda(stiffness, damping, self.ip_address)
        if not success:
            self.logger.error("Failed to set joint impedance.")
        return success

    def change_control_mode(self, mode="position"):
        if mode == "position":
            success = self.robot_api.changeControlMode(self.robot_api.mode_e.T_MODE_POSITION,self.ip_address)
        elif mode == "joint_impedance":
            success = self.robot_api.changeControlMode(self.robot_api.mode_e.T_MODE_JOINT_IMPEDANCE,self.ip_address)
        elif mode == "cartesian_impedance":
            success = self.robot_api.changeControlMode(self.robot_api.mode_e.T_MODE_CART_IMPEDANCE,self.ip_address)
        else:
            print("Invalid control mode. Options could be 'position', 'joint_impedance', 'cartesian_impedance'.")
            return False
        if success:
            self.control_mode = mode
        return success

    def get_joint_impedance(self):
        """
        Get the joint impedance of the robot
        :return: tuple of stiffness and damping
        """
        stiffness = [0] * 7
        damping = [0] * 7
        success = self.robot_api.getJointImpeda(stiffness, damping, self.ip_address)
        if success:
            return stiffness, damping
        else:
            self.logger.error("Failed to get joint impedance.")
            return None

    def set_cartesian_impedance(self, stiffness=(100, 100, 100, 100, 100, 100), damping=0):
        """
        Set the cartesian impedance of the robot
        :param stiffness: the stiffness of the joint
        :param damping: the damping of the joint
        :param inertia: the inertia of the joint
        :return:
        """
        success = self.robot_api.setCartImpeda(stiffness, damping, self.ip_address)
        if not success:
            self.logger.error("Failed to set cartesian impedance.")
        return success

    def get_cartesian_impedance(self):
        """
        Get the cartesian impedance of the robot
        :return: tuple of stiffness and damping
        """
        stiffness = [0] * 6
        damping = [0] * 6
        success = self.robot_api.getCartImpeda(stiffness, damping, self.ip_address)
        if success:
            return stiffness, damping
        else:
            self.logger.error("Failed to get cartesian impedance.")
            return None

    def create_path(self, path_type=0):
        """
        Create a path for the robot

        :param path_type: the type of the path
                        moveJ = 1  # move joint
                        moveL = 2  # move link
        :return: path_id
        """
        if path_type not in [1, 2]:
            self.logger.warning("Invalid path type. Please choose from the following types: [1, 2]")
            # print all the available types
            print("moveJ = 1  # move joint")
            print("moveL = 2  # move link")
            return
        success, path_id = self.robot_api.createPath(path_type, self.ip_address)
        if not success:
            self.logger.error("Failed to create path.")
            return None
        self.path_ids.append(path_id)
        return path_id

    def add_point_to_path_joint(self, path_id, joints, vel_percent=0.2, acc_percent=0.2, blendradius_percent=0.3):
        """
        Add a joint point to path
        :param path_id: the path id
        :param joints: the joint positions to move to, in radians
        :param vel_percent: the velocity percentage
        :param acc_percent: the acceleration percentage
        :param blendradius_percent: the blend radius percentage
        :return:
        """
        assert path_id in self.path_ids, f"Path ID:{path_id} not found."
        success = self.robot_api.addMoveJ(path_id, joints, vel_percent, acc_percent, blendradius_percent, self.ip_address)
        if not success:
            self.logger.error("Failed to add joint point to path.")
        return success

    def add_point_to_path_link(self, path_id, joints, vel=0.1, acc=0.1, blendradius_percent=0.3):
        """
        Add a link point to path
        :param path_id: the path id
        :param joints: the joint positions to move to, in radians
        :param vel: the velocity
        :param acc: the acceleration
        :param blendradius_percent: the blend radius percentage
        :return:
        """
        assert path_id in self.path_ids, f"Path ID:{path_id} not found."
        success = self.robot_api.addMoveL(path_id, joints, vel, acc, blendradius_percent, self.ip_address)
        if not success:
            self.logger.error("Failed to add link point to path.")
        return success

    def execute_path(self, path_id, wait=True):
        """
        Execute the path
        :param path_id: the path id
        :param wait: if True, wait for the move to complete before returning
        :return:
        """
        assert path_id in self.path_ids, f"Path ID:{path_id} not found."
        success = self.robot_api.runPath(path_id, self.ip_address)
        if not success:
            self.logger.error("Failed to execute path.")
        return success

    def destroy_path(self, path_id):
        """
        Destroy the path
        :param path_id: the path id
        :return:
        """
        assert path_id in self.path_ids, f"Path ID:{path_id} not found."
        success = self.robot_api.destroyPath(path_id, self.ip_address)
        if not success:
            self.logger.error("Failed to destroy path.")
        self.path_ids.remove(path_id)
        return success

    def save_config(self):
        """
        Save the current configuration to the robot, including the joint impedance, cartesian impedance, etc.
        When the robot is restarted, the configuration will be loaded.
        :return:
        """
        success = self.robot_api.saveEnvironment(self.ip_address)
        if not success:
            self.logger.error("Failed to save configuration.")
        return success

    def set_waypoint_with_name(self, waypoint_name, tcp_pos, joints):
        """
        Set the waypoint
        :param waypoint_name: the name of the waypoint
        :param tcp_pos: the TCP position, position and axis angle
        :param joints: the joint positions
        :return:
        """
        success = self.robot_api.setWayPoint(waypoint_name, tcp_pos, joints, self.ip_address)
        if not success:
            self.logger.error("Failed to set waypoint.")
        return success

    def get_waypoint_with_name(self, waypoint_name):
        """
        Get the waypoint
        :param waypoint_name: the name of the waypoint
        :return: the TCP position and joint positions
        """
        tcp_pos = [0.0] * 7
        joints = [0.0] * 7
        success = self.robot_api.getWayPoint(waypoint_name, tcp_pos, joints, self.ip_address)
        if success:
            return tcp_pos, joints
        else:
            self.logger.error("Failed to get waypoint.")
            return None
    def delete_waypoint_with_name(self, waypoint_name):
        """
        Delete the waypoint
        :param waypoint_name: the name of the waypoint
        :return:
        """
        success = self.robot_api.deleteWayPoint(waypoint_name, self.ip_address)
        if not success:
            self.logger.error("Failed to delete waypoint.")
        return success

    def get_tcp_transform(self):
        """
        Get the current TCP position of the robot
        :return: the homogeneous transformation matrix of the TCP position
        """

        success = self.robot_api.getDefaultActiveTcp(self.tcp_transform, self.ip_address)
        if success:
            return self.tcp_transform
        else:
            self.logger.error("Failed to get TCP position.")
            return None

    def get_tcp_pose(self):
        """
        Get the current TCP position of the robot
        :return: the pose of the TCP, (x, y, z, rx, ry, rz)
        """
        success = self.robot_api.getDefaultActiveTcpPose(self.tcp_pos, self.ip_address)
        if success:
            return self.tcp_pos
        else:
            self.logger.error("Failed to get TCP position.")
            return None

    def get_tcp_payload(self):
        """
        Get the current TCP payload of the robot
        :return: the TCP payload, (mass:[0], center of mass:[1-3], tensor:[4-9])
        """
        payLoad=[0.0]*10
        success = self.robot_api.getActiveTcpPayload(payLoad,self.ip_address)
        if success:
            return payLoad
        else:
            self.logger.error("Failed to get TCP payload.")
            return None

    def stop(self):
        """
        Stop the robot
        :return:
        """
        print("diana_control.stop...")
        self.robot_api.stop(self.ip_address)

    def __del__(self):
        # # stop the robot
        # self.robot_api.stop(self.ip_address)  # stop the robot with max deceleration
        # # hold the brake
        # self.robot_api.holdBrake(self.ip_address)
        # destroy the server
        self.reset_control_mode()
        self.robot_api.destroySrv(self.ip_address)


