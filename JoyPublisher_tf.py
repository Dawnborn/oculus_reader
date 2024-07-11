import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from oculus_reader.reader import OculusReader
from scipy.spatial.transform import Rotation
import time

def ret2xyzquat(ret):
    try:
        rotmat_right = ret[0]['r']
        buttons = ret[1]
        rightGrip = buttons['rightGrip'][0]

        R = Rotation.from_matrix(rotmat_right[:3,:3])
        quat = R.as_quat(scalar_first=False)
        x, y, z = rotmat_right[:3, 3]

        r = [x, y, z, quat[0], quat[1], quat[2], quat[3]]
        r.append(rightGrip)
        
    except:
        r = [-1.0] * 8
    
    return r

class MinimalPublisher(Node):

    def __init__(self, timer_period=0.5):
        super().__init__('minimal_publisher')
        self.timer_period = timer_period  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.oculus_reader = OculusReader()

    def timer_callback(self):
        ret = self.oculus_reader.get_transformations_and_buttons()
        ans = ret2xyzquat(ret)

        if ans[0] != -1.0:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'world'
            t.child_frame_id = 'oculus_frame'

            t.transform.translation.x = ans[0]
            t.transform.translation.y = ans[1]
            t.transform.translation.z = ans[2]
            t.transform.rotation.x = ans[3]
            t.transform.rotation.y = ans[4]
            t.transform.rotation.z = ans[5]
            t.transform.rotation.w = ans[6]

            self.tf_broadcaster.sendTransform(t)
            self.get_logger().info('Publishing TF: %s' % str(ans))


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher(timer_period=0.02)

    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
