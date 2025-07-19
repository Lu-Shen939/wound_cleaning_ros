# simplified_visualize_path.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class SimplifiedPathVisualizer(Node):
    def __init__(self):
        super().__init__('simplified_path_visualizer')
        
        # 订阅话题
        self.path_sub = self.create_subscription(
            PoseArray, '/cleaning_path', self.path_callback, 10)
        self.times_sub = self.create_subscription(
            Float32MultiArray, '/path_times', self.times_callback, 10)
        self.velocities_sub = self.create_subscription(
            Float32MultiArray, '/path_velocities', self.velocities_callback, 10)
        
        # 数据存储
        self.current_path = None
        self.current_times = None
        self.current_velocities = None
        
        self.get_logger().info('Simplified path visualizer started')

    def path_callback(self, msg):
        self.get_logger().info(f'Received path with {len(msg.poses)} poses')
        self.current_path = msg
        self.visualize_if_ready()

    def times_callback(self, msg):
        self.current_times = msg.data
        self.visualize_if_ready()

    def velocities_callback(self, msg):
        self.current_velocities = msg.data
        self.visualize_if_ready()

    def visualize_if_ready(self):
        if (self.current_path is not None and 
            self.current_times is not None and 
            self.current_velocities is not None):
            self.visualize_path()

    def classify_segments(self):
        """根据速度对路径段进行分类"""
        if not self.current_velocities:
            return ['unknown'] * (len(self.current_path.poses) - 1)
        
        segment_types = []
        for velocity in self.current_velocities:
            if velocity < 0.01:  # 清洁速度阈值
                segment_types.append('cleaning')
            elif velocity < 0.02:  # 接近速度阈值
                segment_types.append('approach')
            else:
                segment_types.append('transition')
        return segment_types

    def visualize_path(self):
        try:
            # 提取坐标数据
            x_coords = [pose.position.x for pose in self.current_path.poses]
            y_coords = [pose.position.y for pose in self.current_path.poses]
            z_coords = [pose.position.z for pose in self.current_path.poses]
            
            # 分类路径段
            segment_types = self.classify_segments()
            
            # 创建2x2的子图布局
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. 3D路径视图
            ax1 = fig.add_subplot(221, projection='3d')
            self.plot_3d_path(ax1, x_coords, y_coords, z_coords, segment_types)
            
            # 2. 俯视图
            self.plot_top_view(ax2, x_coords, y_coords, segment_types)
            
            # 3. 速度剖面图
            self.plot_velocity_profile(ax3)
            
            # 4. 基本统计信息
            self.plot_statistics(ax4, x_coords, y_coords, z_coords, segment_types)
            
            plt.tight_layout()
            plt.show()
            plt.close(fig)
            
            # 打印基本统计信息
            self.print_statistics(x_coords, y_coords, z_coords, segment_types)
            
        except Exception as e:
            self.get_logger().error(f'Error visualizing path: {str(e)}')

    def plot_3d_path(self, ax, x_coords, y_coords, z_coords, segment_types):
        """3D路径图，根据段类型着色"""
        colors = {'cleaning': 'red', 'approach': 'orange', 'transition': 'blue', 'unknown': 'gray'}
        
        # 绘制连接线
        for i in range(len(x_coords) - 1):
            segment_type = segment_types[i] if i < len(segment_types) else 'unknown'
            color = colors[segment_type]
            ax.plot([x_coords[i], x_coords[i+1]], 
                   [y_coords[i], y_coords[i+1]], 
                   [z_coords[i], z_coords[i+1]], 
                   color=color, linewidth=2)
        
        # 标记起始点和结束点
        ax.scatter(x_coords[0], y_coords[0], z_coords[0], 
                  c='green', s=100, marker='o', label='Start')
        ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], 
                  c='red', s=100, marker='s', label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Cleaning Path')
        ax.legend()

    def plot_top_view(self, ax, x_coords, y_coords, segment_types):
        """俯视图"""
        colors = {'cleaning': 'red', 'approach': 'orange', 'transition': 'blue', 'unknown': 'gray'}
        
        # 绘制路径线
        for i in range(len(x_coords) - 1):
            segment_type = segment_types[i] if i < len(segment_types) else 'unknown'
            color = colors[segment_type]
            ax.plot([x_coords[i], x_coords[i+1]], 
                   [y_coords[i], y_coords[i+1]], 
                   color=color, linewidth=2)
        
        # 标记中心点
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        ax.scatter(center_x, center_y, c='black', s=100, marker='x', linewidth=3, label='Center')
        
        # 标记起始点
        ax.scatter(x_coords[0], y_coords[0], c='green', s=80, marker='o', label='Start')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Top View')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()

    def plot_velocity_profile(self, ax):
        """速度剖面图"""
        if not self.current_velocities:
            ax.text(0.5, 0.5, 'No velocity data', ha='center', va='center', transform=ax.transAxes)
            return
        
        velocities = self.current_velocities
        segment_indices = range(len(velocities))
        
        # 根据速度类型着色
        colors = []
        for v in velocities:
            if v < 0.01:
                colors.append('red')     # 清洁
            elif v < 0.02:
                colors.append('orange')  # 接近
            else:
                colors.append('blue')    # 过渡
        
        ax.bar(segment_indices, velocities, color=colors, alpha=0.7)
        
        # 添加阈值线
        ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Cleaning')
        ax.axhline(y=0.02, color='blue', linestyle='--', alpha=0.5, label='Transition')
        
        ax.set_xlabel('Segment Index')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Velocity Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_statistics(self, ax, x_coords, y_coords, z_coords, segment_types):
        """显示基本统计信息"""
        # 计算统计数据
        total_points = len(x_coords)
        cleaning_segments = segment_types.count('cleaning')
        transition_segments = segment_types.count('transition')
        approach_segments = segment_types.count('approach')
        
        # 计算总距离
        total_distance = 0.0
        for i in range(1, len(x_coords)):
            dist = np.sqrt((x_coords[i]-x_coords[i-1])**2 + 
                          (y_coords[i]-y_coords[i-1])**2 + 
                          (z_coords[i]-z_coords[i-1])**2)
            total_distance += dist
        
        total_time = max(self.current_times) if self.current_times else 0
        avg_velocity = np.mean(self.current_velocities) if self.current_velocities else 0
        
        # 显示统计信息
        stats_text = f"""Path Statistics:
        
Total Points: {total_points}
Total Distance: {total_distance:.3f} m
Total Time: {total_time:.1f} s
Avg Velocity: {avg_velocity:.4f} m/s

Segment Types:
• Cleaning: {cleaning_segments}
• Approach: {approach_segments}  
• Transition: {transition_segments}

Workspace:
X: {min(x_coords):.3f} to {max(x_coords):.3f} m
Y: {min(y_coords):.3f} to {max(y_coords):.3f} m
Z: {min(z_coords):.3f} to {max(z_coords):.3f} m"""
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9, fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Statistics')

    def print_statistics(self, x_coords, y_coords, z_coords, segment_types):
        """打印基本统计信息"""
        total_points = len(x_coords)
        
        # 计算总距离
        total_distance = 0.0
        cleaning_distance = 0.0
        
        for i in range(1, len(x_coords)):
            dist = np.sqrt((x_coords[i]-x_coords[i-1])**2 + 
                          (y_coords[i]-y_coords[i-1])**2 + 
                          (z_coords[i]-z_coords[i-1])**2)
            total_distance += dist
            
            if i-1 < len(segment_types) and segment_types[i-1] == 'cleaning':
                cleaning_distance += dist
        
        cleaning_segments = segment_types.count('cleaning')
        total_time = max(self.current_times) if self.current_times else 0
        avg_velocity = np.mean(self.current_velocities) if self.current_velocities else 0
        
        self.get_logger().info("=== PATH SUMMARY ===")
        self.get_logger().info(f"Points: {total_points}, Distance: {total_distance:.3f}m")
        self.get_logger().info(f"Cleaning distance: {cleaning_distance:.3f}m ({cleaning_segments} segments)")
        self.get_logger().info(f"Time: {total_time:.1f}s, Avg speed: {avg_velocity:.4f}m/s")
        self.get_logger().info("===================")

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