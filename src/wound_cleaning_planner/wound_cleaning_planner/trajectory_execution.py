import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from frankx_execution import MotionPlanningEnhancement
from frankx import Robot
import numpy as np

# Encoding Mapping
SEGMENT_TYPE_DICT = {
    0: 'approach',
    1: 'cleaning',
    2: 'retreat',
    3: 'return_to_center'
}


class CleaningExecutionNode(Node):
    def __init__(self):
        super().__init__('cleaning_execution_node')
        self.get_logger().info("✅ Cleaning Execution Node started")

        # Initialize the FrankaX robot
        robot_ip = self.declare_parameter("robot_ip", "192.168.1.118").value
        self.robot = Robot(robot_ip)
        self.robot.set_default_behavior()
        self.robot.recover_from_errors()
        self.robot.set_dynamic_rel(0.15)

        # Initialize trajectory planner and executor
        self.executor = MotionPlanningEnhancement(self.robot)

        # Subscribe to cleaning path results
        self.create_subscription(
            Float32MultiArray,
            '/cleaning_result',
            self.cleaning_result_callback,
            10
        )

    def cleaning_result_callback(self, msg: Float32MultiArray):
        self.get_logger().info("📥 Received /cleaning_result message")
        try:
            data = np.array(msg.data, dtype=np.float32)
            if len(data) % 7 != 0:
                self.get_logger().error("⚠️ Received data length not divisible by 7 (3 pos + 3 normal + 1 type)")
                return

            data = data.reshape(-1, 7)
            positions = data[:, 0:3]
            normals = data[:, 3:6]
            segment_ids = data[:, 6].astype(int)
            segment_types = [SEGMENT_TYPE_DICT.get(i, 'cleaning') for i in segment_ids]

            path_result = {
                'path_3d': positions,
                'orientations': normals,
                'segment_types': segment_types
            }

            # Calling FrankX optimization function
            self.get_logger().info("🛠 Optimizing trajectory...")
            optimized_result = self.executor.optimize_trajectory_with_constraints(path_result)

            if optimized_result is None:
                self.get_logger().error("❌ Trajectory optimization failed")
                return

            # Execution Track
            self.get_logger().info("🤖 Executing optimized trajectory...")
            exec_result = self.executor.execute_trajectory_interface(optimized_result)

            if exec_result:
                self.get_logger().info(f"✅ Trajectory executed. Duration: {exec_result['total_duration']:.2f}s")
            else:
                self.get_logger().warn("⚠️ Execution interface returned None")

        except Exception as e:
            self.get_logger().error(f"❗ Exception during execution: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = CleaningExecutionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
