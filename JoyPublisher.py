import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from oculus_reader.reader import OculusReader

def ret2string(ret):

    try:
        rotmat_right = ret[0]['r']
        buttons = ret[1]
        rightGrip = buttons['rightGrip'][0]

        r = rotmat_right.reshape(1,-1)[0].tolist()
        r.append(rightGrip)
        r = [str(i) for i in r]
        r = ",".join(r)
    except:
        r = "-1"
    
    return r

class MinimalPublisher(Node):

    def __init__(self, topic="topic", timer_period=0.5):
        super().__init__('minimal_publisher')
        self.topic = topic
        self.publisher_ = self.create_publisher(String, self.topic, 50)
        self.timer_period = timer_period  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.i = 0

        self.oculus_reader = OculusReader()

    def timer_callback(self):
        
        msg = String()

        ret = self.oculus_reader.get_transformations_and_buttons()
        ans = ret2string(ret)

        msg.data = ans
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)


def main(args=None):

    
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher(topic="joypose")

    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
