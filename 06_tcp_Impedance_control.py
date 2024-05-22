from diana_contol import DianaControl
import numpy as np
'''
    The configuration of impedance is according to the TCP axis, you still can control the robot for both tcp and joints.
    And trajectory seems will end by fixed time, or fixed number of the waypoints, it will not check if the goal if reached.
'''
def test():
    ip_addr = "192.168.10.75"
    diana = DianaControl(ip_address=ip_addr)
    # diana.set_log_level('DEBUG')

    success = diana.set_cartesian_impedance(stiffness=(100, 100, 100, 100, 100, 100), damping=[0.5])
    print(f"Set impedance:{success}")
    success = diana.change_control_mode("cartesian_impedance")
    print(f"change control mode:{success}")

    tcp_pos_current = diana.get_tcp_pose()
    print(tcp_pos_current)
    input("current pos")

    position = np.array([400,-400,200])/1000
    oritation = np.array([10,-6,-50])/180*np.pi
    tcp_pos_target = np.concatenate([position,oritation],axis=-1)
    success = diana.move_tcp(
        tcp_pos=tcp_pos_target,
        vel=0.1,
        acc=0.1,
        wait=True
    )
    print(f"Move joints success: {success}")

if __name__ == "__main__":
    test()