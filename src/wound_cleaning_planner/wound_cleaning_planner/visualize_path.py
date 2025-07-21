import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class SimplifiedPathVisualizer(Node):
    def __init__(self):
        super().__init__('simplified_path_visualizer')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/cleaning_path_raw',
            self.path_callback,
            10)
        self.get_logger().info('Visualizer ready. Waiting for /cleaning_path_raw...')

    def path_callback(self, msg):
        data = msg.data
        if len(data) % 8 != 0:
            self.get_logger().error("Data length is not a multiple of 8 (x, y, z, qx, qy, qz, qw, segment_type)")
            return

        num_points = len(data) // 8
        points = np.array(data, dtype=np.float32).reshape((num_points, 8))
        self.visualize(points)

    def visualize(self, points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        seg_types = points[:, 7].astype(int)

    
        seg_color_map = {
            0: 'orange',   # approach
            1: 'red',      # cleaning
            2: 'blue',     # retreat
            3: 'purple'    # return_to_center
        }

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('3D Cleaning Path Visualization')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        for i in range(len(points) - 1):
            seg_type = seg_types[i]
            color = seg_color_map.get(seg_type, 'gray')
            ax.plot(
                [x[i], x[i+1]],
                [y[i], y[i+1]],
                [z[i], z[i+1]],
                color=color, linewidth=2
            )

        ax.scatter(x[0], y[0], z[0], c='green', s=80, marker='o', label='Start')
        ax.scatter(x[-1], y[-1], z[-1], c='black', s=80, marker='x', label='End')

        ax.legend()
        plt.tight_layout()
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = SimplifiedPathVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
