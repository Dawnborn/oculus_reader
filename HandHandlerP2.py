import numpy as np
import time
import pyHandApi

SOFT_STIFF_FACTOR = 2
HARD_STIFF_FACTOR = 6 # max is 10

IP_ADDRESS_LEFT = "192.168.100.85"
IP_ADDRESS_RIGHT = "192.168.100.63"

ip_address = IP_ADDRESS_LEFT

SOFT_STIFF_FACTOR = 2
HARD_STIFF_FACTOR = 6 # max is 10


tire_grasp_open = np.array([
      0.966806, 0.00210926, 0.00651952, 0.00239688, 
      0.000287626, 0.00373914, 0.00402676, 
      0.000958753, 0.00364326, 0.00278038, 
      0.00412264, 0.0001, 0.00210926, 
      0.00364326, 0.000767002, 0.00134225
    ])
 
tire_grasp_close = np.array([
      0.966902, 0.000383501, 0.136718, 0.142183, 
      -0.0272286, 0.0187916, 0.29635, 
      -0.0255028, 0.14017, 0.524917, 
      -0.0730569, 0.13691, 0.521178, 
      -0.0175452, 0.0313512, 0.222814,
    ])

hand_fully_open = np.radians([54.0] + [0.0, 0.0, 0.0] + [0.0, 0.0, 0.0] * 4)

cube_hand_open = np.radians([54.0] + [0.0, 2.0, 2.0] + [-5.0, 20.0, 10.0] * 2 + [0.0, 60.0, 35.0] * 2)
cube_hand_close = np.radians([54.0] + [0.0, 7.0, 15.0] + [-5.0, 35.0, 25.0] * 2 + [0.0, 60.0, 35.0] * 2)

hand_grasp_open = np.radians([54.0] + [-6.0, 3.5, 0.0] + [-5.0, 8.0, 4.0] * 2 + [0.0, 7.0, 6.0] * 2)
hand_grasp_close = np.radians([54.0] + [-6.0, 8.0, 8.0] + [-5.0, 26.0, 15.0] * 2 + [0.0, 16.0, 23.0] * 2)
hand_grasp_more_close = np.radians([54] + [-6.0, 11.0, 7.0] + [-5.0, 32.0, 15.0] * 2 + [0.0, 23.0 , 23.0] * 2)

modes = [
#    open-Position   | close-Position
    [hand_grasp_open, hand_grasp_more_close],
    [cube_hand_open,   cube_hand_close], 
    ]

if __name__ == "__main__":

    hand = pyHandApi.HandApi(ip_address)

    time.sleep(2)
    flag = True
    while(flag):
        time.sleep(0.5)
        try:
            ret = hand.connect()
            flag = False
        except:
            print("connect error")
            continue

    time.sleep(2)


    hand.clearError()
    hand.clearError()
    hand.clearError()
    hand.clearError()
    hand.clearError()
    hand.clearError()


    hand.setHandMode('impedance')

    hand.setJoints(joints=modes[0][0],velocity=5, acceleration=5)
    time.sleep(2)

    hand.setJoints(joints=modes[0][1],velocity=5, acceleration=5)



    hand.clearError()
    del hand