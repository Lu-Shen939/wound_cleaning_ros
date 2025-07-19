# test_start_cleaning.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
import time

class TestStartCleaning(Node):
    def __init__(self):
        super().__init__('test_start_cleaning')
        self.publisher = self.create_publisher(Empty, '/start_cleaning', 10)
        
        
        self.get_logger().info('Waiting 5 seconds before sending start command...')
        self.timer = self.create_timer(5.0, self.send_start_command)

    def send_start_command(self):
        msg = Empty()
        self.publisher.publish(msg)
        self.get_logger().info('Sent start cleaning command')
        
        
        self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = TestStartCleaning()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()