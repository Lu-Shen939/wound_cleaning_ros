import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from std_msgs.msg import Empty, Float32MultiArray
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge
import time
from geometry_msgs.msg import PointStamped  
from tf2_ros import TransformException  
import traceback
import tf2_geometry_msgs
import os
from datetime import datetime



class VelocityProfile:
      
    def __init__(self, cleaning_speed=0.005, transition_speed=0.02, 
                 retreat_speed=0.015, approach_speed=0.01,
                 max_acceleration=0.01, min_segment_time=0.1):
        
        self.cleaning_speed = cleaning_speed
        self.transition_speed = transition_speed
        self.retreat_speed = retreat_speed  
        self.approach_speed = approach_speed  
        self.max_acceleration = max_acceleration
        self.min_segment_time = min_segment_time
    
    def calculate_path_timing(self, path_3d, segment_types):
        if len(path_3d) < 2:
            return [0.0], [0.0]
        
        timestamps = [0.0]
        velocities = []
        current_time = 0.0
        
        for i in range(len(path_3d) - 1):
            segment_distance = np.linalg.norm(path_3d[i+1] - path_3d[i])
            
            if i < len(segment_types):
                segment_type = segment_types[i]
            else:
                segment_type = 'cleaning'
            
            # 根据段类型设置目标速度
            if segment_type == 'approach':
                target_speed = self.approach_speed
            elif segment_type == 'cleaning':
                target_speed = self.cleaning_speed
            elif segment_type == 'retreat':  
                target_speed = self.retreat_speed
            elif segment_type == 'back_to_center':  
                target_speed = self.transition_speed
            else:  # 其他过渡段
                target_speed = self.transition_speed
            
            if segment_distance > 0:
                segment_time = max(segment_distance / target_speed, self.min_segment_time)
                actual_velocity = segment_distance / segment_time
            else:
                segment_time = self.min_segment_time
                actual_velocity = 0.0
            
            current_time += segment_time
            timestamps.append(current_time)
            velocities.append(actual_velocity)
        
        return timestamps, velocities
    
    def apply_smooth_acceleration(self, timestamps, velocities):
        
        if len(velocities) < 2:
            return timestamps, velocities
        
        smoothed_timestamps = [timestamps[0]]
        smoothed_velocities = []
        current_time = timestamps[0]
        current_velocity = 0.0
        
        for i, target_velocity in enumerate(velocities):
            
            velocity_change = target_velocity - current_velocity
            
            if abs(velocity_change) > 0:
                accel_time = abs(velocity_change) / self.max_acceleration
                
            
                if accel_time > 0.1: 
                
                    accel_end_time = current_time + accel_time
                    smoothed_timestamps.append(accel_end_time)
                    current_time = accel_end_time
                    current_velocity = target_velocity
            
            
            if i + 1 < len(timestamps):
                remaining_time = timestamps[i + 1] - current_time
                if remaining_time > 0:
                    current_time += remaining_time
                    smoothed_timestamps.append(current_time)
            
            smoothed_velocities.append(current_velocity)
        
        return smoothed_timestamps, smoothed_velocities


class WoundCleaningPlanner:
    def __init__(self, tool_diameter=0.01, coverage_overlap=0.2, edge_extension=0.005):
        self.tool_diameter = tool_diameter
        self.coverage_overlap = coverage_overlap
        self.edge_extension = edge_extension
        self.camera_matrix = None
        self.dist_coeffs = None
        self.velocity_planner = VelocityProfile(
            cleaning_speed=0.003,    
            transition_speed=0.015,  
            max_acceleration=0.008,  
            min_segment_time=0.2     
        )

    def set_camera_params(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def find_wound_center(self, mask):
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        main_contour = max(contours, key=cv2.contourArea)
        
        M = cv2.moments(main_contour)
        if M["m00"] == 0:
            return None
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return np.array([cx, cy]), main_contour
    
    
    def pixel_to_3d(self, pixel_point, depth_value):
    
        if self.camera_matrix is None:
            raise ValueError("Camera matrix not set")
            
        x, y = pixel_point[0], pixel_point[1]
        
        x_cam = (x - self.camera_matrix[0, 2]) * depth_value / self.camera_matrix[0, 0]
        y_cam = (y - self.camera_matrix[1, 2]) * depth_value / self.camera_matrix[1, 1]
        z_cam = depth_value
        
        return np.array([x_cam, y_cam, z_cam])

    def get_depth_at_pixel(self, depth_map, pixel_point, radius=5):
        
        x, y = int(pixel_point[0]), int(pixel_point[1])
        
        if x < 0 or x >= depth_map.shape[1] or y < 0 or y >= depth_map.shape[0]:
            return None
            
        # first try to extract directly
        if depth_map[y, x] > 0:
            return depth_map[y, x]
        
        # Expand the search scope

        for r in range(1, radius + 1):
            x_min = max(0, x - r)
            x_max = min(depth_map.shape[1], x + r + 1)
            y_min = max(0, y - r)
            y_max = min(depth_map.shape[0], y + r + 1)
            
            local_patch = depth_map[y_min:y_max, x_min:x_max]
            valid_depths = local_patch[local_patch > 0]
            
            if len(valid_depths) > 0:
                return np.median(valid_depths)
                
        return None

    def find_ray_intersection(self, center_2d, ray_direction, mask_binary):
        
        h, w = mask_binary.shape
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        
        max_distance = max(w, h)
        step_size = 0.5
        num_steps = int(max_distance / step_size)
        
        last_valid_point = None
        
        for i in range(num_steps):
            distance = i * step_size
            current_point = center_2d + ray_direction * distance
            
            x, y = int(round(current_point[0])), int(round(current_point[1]))
            
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            
            if mask_binary[y, x] > 0:
                last_valid_point = current_point.copy()
            else:
                if last_valid_point is not None:
                    return last_valid_point
                else:
                    return None
        
        return last_valid_point
    
    def estimate_3d_distance(self, center_2d, edge_2d, depth_map):
  
        center_depth = self.get_depth_at_pixel(depth_map, center_2d)
        edge_depth = self.get_depth_at_pixel(depth_map, edge_2d)
        
        if center_depth is None or edge_depth is None:
            
            return np.linalg.norm(edge_2d - center_2d) * 0.001  
        
        center_3d = self.pixel_to_3d(center_2d, center_depth)
        edge_3d = self.pixel_to_3d(edge_2d, edge_depth)
        
        distance_3d = np.linalg.norm(edge_3d - center_3d)
        
        return distance_3d

    def generate_ray_path_points(self, center_2d, edge_2d, depth_map):
       
        ray_2d_points = []
        ray_3d_points = []
        
        # Generate 2d path points
        ray_length_3d = self.estimate_3d_distance(center_2d, edge_2d, depth_map)
        
        
        point_spacing = self.tool_diameter * 0.8  
        num_points = max(3, int(ray_length_3d / point_spacing))
        

        for i in range(num_points):
            ratio = i / (num_points - 1) if num_points > 1 else 0
            point_2d = center_2d + (edge_2d - center_2d) * ratio
            ray_2d_points.append(point_2d)
        
        # Convert to 3d points
        center_depth = self.get_depth_at_pixel(depth_map, center_2d)
        if center_depth is None or center_depth <= 0:
            print(f"Warning: Invalid center depth at {center_2d}")
            return None
        
        segment_types = []    
        for i, point_2d in enumerate(ray_2d_points):
            depth = self.get_depth_at_pixel(depth_map, point_2d)
            
            if depth is not None and depth > 0:
                # In-depth rationality check
                if abs(depth - center_depth) < 0.1:  
                    point_3d = self.pixel_to_3d(point_2d, depth)
                    ray_3d_points.append(point_3d)
                else:
                    # Use interpolation depth
                    if i == 0:
                        interpolated_depth = center_depth
                    else:
                        interpolated_depth = center_depth + (depth - center_depth) * i / len(ray_2d_points)
                    
                    if interpolated_depth > 0:
                        point_3d = self.pixel_to_3d(point_2d, interpolated_depth)
                        ray_3d_points.append(point_3d)

                if i == 0:
                    segment_types.append('approach')
                elif i == len(ray_2d_points) - 1:
                    segment_types.append('approach')  
                else:
                    segment_types.append('cleaning')
                        
        return (np.array(ray_3d_points) if ray_3d_points else None, segment_types)
    

    def rotation_matrix_to_quaternion(self, R):
        
        q = np.empty(4)
        trace = np.trace(R)
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            q[0] = 0.25 * S  # w
            q[1] = (R[2, 1] - R[1, 2]) / S  # x
            q[2] = (R[0, 2] - R[2, 0]) / S  # y
            q[3] = (R[1, 0] - R[0, 1]) / S  # z
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[0] = (R[2, 1] - R[1, 2]) / S
            q[1] = 0.25 * S
            q[2] = (R[0, 1] + R[1, 0]) / S
            q[3] = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[0] = (R[0, 2] - R[2, 0]) / S
            q[1] = (R[0, 1] + R[1, 0]) / S
            q[2] = 0.25 * S
            q[3] = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[0] = (R[1, 0] - R[0, 1]) / S
            q[1] = (R[0, 2] + R[2, 0]) / S
            q[2] = (R[1, 2] + R[2, 1]) / S
            q[3] = 0.25 * S
        
        return q

    def generate_tool_orientation(self, pixel_point, normal):
        
        # Use the incoming pixel point calculation vector 
        z_axis = -normal / np.linalg.norm(normal)
        
        # Generate vertical x-axis
        if abs(z_axis[2]) > 0.99:
            x_axis = np.array([1, 0, 0])
        else:
            x_axis = np.array([z_axis[1], -z_axis[0], 0])
            x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Calculate the y-axis
        y_axis = np.cross(z_axis, x_axis)
        
        rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
        
        return self.rotation_matrix_to_quaternion(rot_matrix)
    

    def generate_cleaning_path(self, mask, depth_map, num_rays=24, retreat_height=0.005):
   
        print(f"Starting path generation with mask shape: {mask.shape}, depth shape: {depth_map.shape}")
    
        center_result = self.find_wound_center(mask)
        if center_result is None:
            print("No wound center found")
            return None
            
        center_2d_depth, main_contour = center_result
        print(f"Wound center found at: {center_2d_depth}")
        
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        mask_scaled = cv2.resize(mask_binary, (depth_map.shape[1], depth_map.shape[0]))
        
        contour_area = cv2.contourArea(main_contour)
        center_depth = self.get_depth_at_pixel(depth_map, center_2d_depth)

        if center_depth is None or center_depth <= 0:
            print("Invalid center depth, cannot convert pixel units")
            return None
        
        fx = self.camera_matrix[0, 0]
        pixel_size_m = center_depth / fx
        
        equivalent_radius_pixels = np.sqrt(contour_area / np.pi)
        equivalent_radius_m = equivalent_radius_pixels * pixel_size_m
        circumference_m = 2 * np.pi * equivalent_radius_m
        
        effective_width = self.tool_diameter * (1 - self.coverage_overlap)
        num_rays = max(8, int(circumference_m / effective_width))
    
        print(f"Using {num_rays} rays for area {contour_area}")
        

        angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)

        all_path_points_3d = []
        all_orientations = []
        all_segment_types = []
        
        center_3d = self.pixel_to_3d(center_2d_depth, center_depth)
        center_normal = self.calculate_surface_normal(depth_map, center_2d_depth)
        
        if center_normal[2] < 0:
            center_normal = -center_normal
        center_normal = center_normal / np.linalg.norm(center_normal)  # 归一化
        
        center_orientation = self.generate_tool_orientation(center_2d_depth, center_normal)
        
        valid_rays = 0
        
        for ray_idx, angle in enumerate(angles):
            ray_direction = np.array([np.cos(angle), np.sin(angle)])
            
           
            edge_2d_scaled = self.find_ray_intersection(center_2d_depth, ray_direction, mask_scaled)
            
            if edge_2d_scaled is not None:
              
                ray_result = self.generate_ray_path_points(
                    center_2d_depth, edge_2d_scaled, depth_map
                )
                
                if ray_result[0] is not None and len(ray_result[0]) > 0:
                    ray_points_3d, ray_segment_types = ray_result
                    valid_rays += 1

                    
                    ray_path, ray_orientations, ray_types = self._create_complete_ray_path(
                        ray_points_3d, 
                        center_2d_depth, 
                        edge_2d_scaled, 
                        center_3d,
                        center_normal,
                        center_orientation,
                        depth_map, 
                        retreat_height,
                        is_last_ray=(ray_idx == len(angles) - 1)
                    )
                    
                  
                    all_path_points_3d.extend(ray_path)
                    all_orientations.extend(ray_orientations)
                    all_segment_types.extend(ray_types)
        
        print(f"Generated {len(all_path_points_3d)} points from {valid_rays} valid rays")
        
        if len(all_path_points_3d) == 0:
            print("No valid path points generated")
            return None
        
       
        timestamps, velocities = self.velocity_planner.calculate_path_timing(
            np.array(all_path_points_3d), all_segment_types
        )
        
    
        smooth_timestamps, smooth_velocities = self.velocity_planner.apply_smooth_acceleration(
            timestamps, velocities
        )
            
        return {
            'path_3d': np.array(all_path_points_3d),
            'orientations': all_orientations,
            'num_rays': valid_rays,
            'timestamps': smooth_timestamps,
            'velocities': smooth_velocities,
            'segment_types': all_segment_types
        }

    def _create_complete_ray_path(self, ray_points_3d, center_2d, edge_2d, center_3d, 
                            center_normal, center_orientation, depth_map, 
                            retreat_height, is_last_ray=False):
        ray_path = []
        ray_orientations = []
        ray_types = []
        
        for i, point_3d in enumerate(ray_points_3d):
            ray_path.append(point_3d)
            
            ratio = i / (len(ray_points_3d) - 1) if len(ray_points_3d) > 1 else 0
            pixel_point = center_2d + (edge_2d - center_2d) * ratio
            normal = self.calculate_surface_normal(depth_map, pixel_point)
            
            if normal[2] < 0:
                normal = -normal
            normal = normal / np.linalg.norm(normal)
            
            orientation = self.generate_tool_orientation(pixel_point, normal)
            ray_orientations.append(orientation)
            
            if i == 0:
                ray_types.append('approach')
            elif i == len(ray_points_3d) - 1:
                ray_types.append('cleaning')
            else:
                ray_types.append('cleaning')
        
        
        vertical_retreat = np.array([0, 0, 1])
        
        edge_point_raised = ray_points_3d[-1] + vertical_retreat * retreat_height
        ray_path.append(edge_point_raised)
        ray_orientations.append(ray_orientations[-1])
        ray_types.append('retreat')
        
        center_raised = center_3d + vertical_retreat * retreat_height
        ray_path.append(center_raised)
        ray_orientations.append(center_orientation)
        ray_types.append('return_to_center')
        
        if not is_last_ray:
            ray_path.append(center_3d)
            ray_orientations.append(center_orientation)
            ray_types.append('approach')
        
        return ray_path, ray_orientations, ray_types

    def calculate_surface_normal(self, depth_map, pixel_point, smooth_kernel_size=5):
      
        x, y = int(pixel_point[0]), int(pixel_point[1])
        
        
        if x < smooth_kernel_size or x >= depth_map.shape[1] - smooth_kernel_size or \
        y < smooth_kernel_size or y >= depth_map.shape[0] - smooth_kernel_size:
            return np.array([0, 0, 1])  
        
        
        kernel_size = smooth_kernel_size
        
      
        local_region = depth_map[y-kernel_size:y+kernel_size+1, x-kernel_size:x+kernel_size+1]
        
     
        valid_mask = local_region > 0
        if not np.any(valid_mask):
            return np.array([0, 0, 1])
        
    
        median_depth = np.median(local_region[valid_mask])
        local_region[~valid_mask] = median_depth
        
       
        smoothed_region = cv2.GaussianBlur(local_region, (5, 5), 1.0)
        
      
        dzdx = cv2.Sobel(smoothed_region, cv2.CV_32F, 1, 0, ksize=3)
        dzdy = cv2.Sobel(smoothed_region, cv2.CV_32F, 0, 1, ksize=3)
        
      
        center_idx = kernel_size
        grad_x = dzdx[center_idx, center_idx]
        grad_y = dzdy[center_idx, center_idx]
        
       
        if self.camera_matrix is not None:
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            
    
            depth_at_point = smoothed_region[center_idx, center_idx]
            grad_x_3d = grad_x * depth_at_point / fx
            grad_y_3d = grad_y * depth_at_point / fy
            
            normal = np.array([-grad_x_3d, -grad_y_3d, 1])
        else:
            normal = np.array([-grad_x, -grad_y, 1])
        
    
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        else:
            normal = np.array([0, 0, 1])
        
        return normal

    def debug_retreat_direction(self, edge_point, edge_normal, retreat_height):
     
        print(f"Edge point: {edge_point}")
        print(f"Edge normal: {edge_normal}")
        print(f"Retreat height: {retreat_height}")
        
        retreat_vector = edge_normal * retreat_height
        new_position = edge_point + retreat_vector
        
        print(f"Retreat vector: {retreat_vector}")
        print(f"New position: {new_position}")
        print(f"Movement distance: {np.linalg.norm(retreat_vector)}")
        
        return new_position

class CoordinateTransformer:
    def __init__(self, node):
        self.tf_buffer = Buffer()
        self.listener = TransformListener(self.tf_buffer, node)
        self.node = node
        
    def transform_to_base(self, points, source_frame='camera_optical_frame'):
        
        transformed_points = []
        target_frame = self.node.get_parameter('target_frame').value
        
        try:
            now = self.node.get_clock().now()
            
            self.node.get_logger().info(f"Transforming from {source_frame} to {target_frame}")
            
            if not self.tf_buffer.can_transform(target_frame, source_frame, now, timeout=rclpy.duration.Duration(seconds=1.0)):
                self.node.get_logger().warn(f"Transform not available from {source_frame} to {target_frame}")
                return None
            
            transform = self.tf_buffer.lookup_transform(
                target_frame, 
                source_frame,
                now,
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
            
            self.node.get_logger().info(f"Transform found: {transform.transform.translation}")
            
            for point in points:
                point_stamped = PointStamped()
                point_stamped.header.frame_id = source_frame
                point_stamped.header.stamp = now.to_msg()
                point_stamped.point = Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
                
                transformed = self.tf_buffer.transform(point_stamped, target_frame)
                transformed_points.append([
                    transformed.point.x,
                    transformed.point.y,
                    transformed.point.z
                ])
                
            self.node.get_logger().info(f"Transformed {len(points)} points successfully")
            return np.array(transformed_points)
            
        except TransformException as e:
            self.node.get_logger().error(f"Transform failed: {str(e)}")
            
            all_frames = self.tf_buffer.all_frames_as_string()
            self.node.get_logger().info(f"Available frames:\n{all_frames}")
            return None
        except Exception as e:
            self.node.get_logger().error(f"Transform error: {str(e)}")
            return None


class WoundCleaningPlannerNode(Node):
    def __init__(self):
        super().__init__('wound_cleaning_planner')
        self.bridge = CvBridge()
        self.declare_parameter('save_images', True)  
        self.declare_parameter('save_path', '~/wound_cleaning_ws/src/wound_cleaning_planner/images')
         
        self.declare_parameter('tool_diameter', 0.01)
        self.declare_parameter('coverage_overlap', 0.3)
        self.declare_parameter('num_rays', 24)
        self.declare_parameter('target_frame', 'camera_base')
        self.declare_parameter('source_frame', 'depth_camera_link')
        self.declare_parameter('cleaning_speed', 0.003)    
        self.declare_parameter('transition_speed', 0.015)  
        self.declare_parameter('max_acceleration', 0.008)  
        self.declare_parameter('retreat_height', 0.005)  
        self.declare_parameter('retreat_speed', 0.015)   
        self.declare_parameter('approach_speed', 0.01)   

        self.save_images = self.get_parameter('save_images').value
        self.save_path = self.get_parameter('save_path').value
        self.save_path = os.path.expanduser(self.save_path)
    
        if self.save_images:
            os.makedirs(self.save_path, exist_ok=True)
            self.get_logger().info(f"Images will be saved to: {self.save_path}")
                
        self.planner = WoundCleaningPlanner(
            tool_diameter=self.get_parameter('tool_diameter').value,
            coverage_overlap=self.get_parameter('coverage_overlap').value
        )

        self.planner.velocity_planner = VelocityProfile(
            cleaning_speed=self.get_parameter('cleaning_speed').value,
            transition_speed=self.get_parameter('transition_speed').value,
            retreat_speed=self.get_parameter('retreat_speed').value,
            approach_speed=self.get_parameter('approach_speed').value,
            max_acceleration=self.get_parameter('max_acceleration').value
        )
        
        self.transformer = CoordinateTransformer(self)
        
        self.create_subscription(Image, '/segmentation_result', self.mask_callback, 10)
        self.create_subscription(Image, '/depth/image_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/depth/camera_info', self.camera_info_callback, 10)
        self.create_subscription(Empty, '/start_cleaning', self.start_cleaning_callback, 10)
        
        self.path_pub = self.create_publisher(PoseArray, '/cleaning_path', 10)
        self.times_pub = self.create_publisher(Float32MultiArray, '/path_times', 10)
        self.velocities_pub = self.create_publisher(Float32MultiArray, '/path_velocities', 10)

        
        self.current_mask = None
        self.current_depth = None
        self.camera_info = None
        
        self.target_frame = self.get_parameter('target_frame').value
        self.source_frame = self.get_parameter('source_frame').value
        
        self.get_logger().info("Enhanced wound cleaning planner node started")
        self.get_logger().info(f"Target frame: {self.target_frame}, Source frame: {self.source_frame}")

    def save_current_images(self, prefix=""):
        
        if not self.save_images:
            return
            
        if self.current_mask is None or self.current_depth is None:
            self.get_logger().warn("Cannot save images: mask or depth is None")
            return
        
        try:
           
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            mask_filename = f"{prefix}mask_{timestamp}.png"
            mask_path = os.path.join(self.save_path, mask_filename)
            cv2.imwrite(mask_path, self.current_mask)
            
            depth_filename = f"{prefix}depth_{timestamp}.npy"
            depth_path = os.path.join(self.save_path, depth_filename)
            np.save(depth_path, self.current_depth)
            
            self.get_logger().info(f"Saved raw data: {mask_filename}, {depth_filename}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save images: {str(e)}")

    def camera_info_callback(self, msg):

        self.camera_info = msg
        camera_matrix = np.array(msg.k).reshape(3, 3)
        dist_coeffs = np.array(msg.d) if len(msg.d) > 0 else np.zeros(5)
        
        self.planner.set_camera_params(camera_matrix, dist_coeffs)
        self.get_logger().info(f"Camera parameters updated: fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")

    def mask_callback(self, msg):
        
        try:
            self.current_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            mask_info = f"shape: {self.current_mask.shape}, non-zero pixels: {np.count_nonzero(self.current_mask)}"
            self.get_logger().info(f"Received mask - {mask_info}")
        except Exception as e:
            self.get_logger().error(f"Failed to convert mask: {str(e)}")

    def depth_callback(self, msg):
       
        try:
            if msg.encoding == '16UC1':
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                self.current_depth = depth_image.astype(np.float32) / 1000.0  
            elif msg.encoding == '32FC1':
                self.current_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            else:
                self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
                return
            
            depth_stats = f"shape: {self.current_depth.shape}, range: {np.min(self.current_depth[self.current_depth>0]):.3f}-{np.max(self.current_depth):.3f}m"
            self.get_logger().info(f"Received depth - {depth_stats}")
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth: {str(e)}")

    def start_cleaning_callback(self, msg):
        
        if self.current_mask is None:
            self.get_logger().warn("No mask received")
            return
            
        if self.current_depth is None:
            self.get_logger().warn("No depth image received")
            return
            
        if self.camera_info is None:
            self.get_logger().warn("No camera info received")
            return
        
        self.save_current_images("cleaning_")
            
        try:
            self.get_logger().info("Starting enhanced path planning...")
            start_time = time.time()
            
            result = self.planner.generate_cleaning_path(
                self.current_mask,
                self.current_depth,
                num_rays=self.get_parameter('num_rays').value,
                retreat_height=self.get_parameter('retreat_height').value
            )
            
            
            if result is None:
                self.get_logger().error("Failed to generate cleaning path")
                return
                
            planning_time = time.time() - start_time
            self.get_logger().info(f"Path planning completed in {planning_time:.2f}s")
            self.get_logger().info(f"Generated {len(result['path_3d'])} waypoints on {result['num_rays']} rays")
            
            
            base_path = self.transformer.transform_to_base(
                result['path_3d'],
                source_frame=self.source_frame
            )
            
            if base_path is None:
                self.get_logger().error("Failed to transform path to base frame")
            
                self.get_logger().warn("Using camera frame path directly")
                base_path = result['path_3d']
                
            self.publish_path(base_path, result['orientations'],result['timestamps'], result['velocities'])
            
        except Exception as e:
            self.get_logger().error(f"Path planning failed: {str(e)}")
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")

    def publish_path(self, path_3d, orientations,timestamps, velocities):
        
        path_msg = PoseArray()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.target_frame
        
        for i, point in enumerate(path_3d):
            pose = Pose()
            pose.position = Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
            
            if i < len(orientations):
                q = orientations[i]
                pose.orientation = Quaternion(x=float(q[1]), y=float(q[2]), z=float(q[3]), w=float(q[0]))
            else:
                pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        
        times_msg = Float32MultiArray()
        times_msg.data = [float(t) for t in timestamps]
        self.times_pub.publish(times_msg)
        
        velocities_msg = Float32MultiArray()
        velocities_msg.data = [float(v) for v in velocities]
        self.velocities_pub.publish(velocities_msg)
        
        self.get_logger().info(f"Published path with {len(path_3d)} waypoints")
        self.get_logger().info(f"Speed profile: {len(velocities)} segments, avg cleaning speed: {np.mean([v for v in velocities if v < 0.01]):.4f}m/s")
        self.get_logger().info(f"Total trajectory time: {timestamps[-1]:.2f}s")

def main(args=None):
    rclpy.init(args=args)
    node = WoundCleaningPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down wound cleaning planner')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()