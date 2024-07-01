import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self, topic="topic", timer_period=0.5):
        super().__init__('minimal_publisher')
        self.topic = topic
        self.publisher_ = self.create_publisher(String, self.topic, 50)
        self.timer_period = timer_period  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.i = 0

        self.msg2send = None

    def set_msg2send(self,msg2send):
        self.msg2send = msg2send

    def timer_callback(self):
        msg = String()
        if self.msg2send:
            msg.data = 'Hello, ROS 2: %d' % self.i
            self.publisher_.publish(msg)
            self.get_logger().info('Publishing: "%s"' % msg.data)
            self.i += 1

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
