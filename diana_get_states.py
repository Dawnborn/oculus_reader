import os
os.sys.path.append("/home/jh/Documents/ws_diana_yunlong/diana_robot_driver_dev-main")
from diana_contol import DianaControl

def test():
    # initialize the DianaControl object with proper IP address
    ip_addr = "192.168.10.75"
    diana = DianaControl(ip_address=ip_addr)
    # set the log level to DEBUG
    # diana.set_log_level('DEBUG')

    # get the current joint positions, velocities and torques
    joints_pos = diana.get_joint_pos()
    joints_vel = diana.get_joint_vel()
    joints_torque = diana.get_joint_torque()
    print("Joints pos: ", joints_pos)
    print("Joints vel: ", joints_vel)
    print("Joints torque: ", joints_torque)

    # get the current TCP position
    tcp_pos = diana.get_tcp_pos()
    print("TCP pos: ", tcp_pos)

    # get the current TCP force
    tcp_force = diana.get_tcp_force()
    print("TCP force: ", tcp_force)

if __name__ == '__main__':
    test()
