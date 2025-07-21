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
from surface_normals import SurfaceNormalEstimator


class WoundCleaningPlanner:
    def __init__(self, tool_diameter=0.01, coverage_overlap=0.2, edge_extension=0.005):
        self.tool_diameter = tool_diameter
        self.coverage_overlap = coverage_overlap
        self.edge_extension = edge_extension
        self.camera_matrix = None
        self.dist_coeffs = None
        
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

        norm = np.linalg.norm(q)
        if norm == 0:
            raise ValueError("Zero norm quaternion")
        q /= norm
        
        return q

    
    def generate_cleaning_path(self, mask, depth_map, num_rays=16):
        print("Starting simplified path planning...")
        center_result = self.find_wound_center(mask)
        if center_result is None:
            print("No wound center found")
            return None
        
        center_2d, main_contour = center_result
        print(f"Wound center found at: {center_2d}")
        
        if self.camera_matrix is None:
            fx = fy = 525.0
            cx, cy = depth_map.shape[1] / 2, depth_map.shape[0] / 2
            self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        all_path_points_3d = []
        all_segment_types = []
        all_orientations = []
        
        center_depth = self.get_depth_at_pixel(depth_map, center_2d)
        if center_depth is None or center_depth <= 0:
            print("Invalid center depth")
            return None
        
        center_3d = self.pixel_to_3d(center_2d, center_depth)
        print(f"Wound center 3D: {center_3d}")
        
        # Default orientation matrix
        default_orientation = np.eye(3)
        default_orientation[2, 2] = -1  # Flip z-axis
        
        valid_rays = 0
        
        # Generate all path points first
        for i, angle in enumerate(angles):
            ray_direction = np.array([np.cos(angle), np.sin(angle)])
            edge_2d = self.find_ray_intersection(center_2d, ray_direction, mask)
            if edge_2d is not None:
                ray_points_result = self.generate_ray_path_points(center_2d, edge_2d, depth_map)
                if ray_points_result is not None and len(ray_points_result[0]) > 0:
                    ray_points = ray_points_result[0]
                    valid_rays += 1
                    
                    # Approach
                    approach_point = ray_points[0] + np.array([0, 0, 0.02])
                    all_path_points_3d.append(approach_point)
                    all_segment_types.append('approach')
                    
                    # Cleaning path
                    for point in ray_points:
                        all_path_points_3d.append(point)
                        all_segment_types.append('cleaning')
                    
                    # Retreat
                    retreat_point = ray_points[-1] + np.array([0, 0, 0.01])
                    all_path_points_3d.append(retreat_point)
                    all_segment_types.append('retreat')
        
        if len(all_path_points_3d) == 0:
            print("No valid path points generated")
            return None
        
        print(f"Generated {valid_rays} valid rays out of {num_rays}")
        
        normal_estimator = SurfaceNormalEstimator()
        
        all_path_points_3d = np.array(all_path_points_3d)
        
        cleaning_indices = [i for i, seg_type in enumerate(all_segment_types) if seg_type == 'cleaning']
        cleaning_points = all_path_points_3d[cleaning_indices]
        
        print(f"Calculating surface normals for {len(cleaning_points)} cleaning points...")
        
        all_orientations = [default_orientation.copy() for _ in range(len(all_path_points_3d))]
        
        if len(cleaning_points) >= 10:  
            try:
        
                cleaning_normals = normal_estimator.calculate_surface_normals(
                    points_3d=cleaning_points,
                    radius=0.02,
                    min_points=20
                )
                
                unified_normals = normal_estimator.unify_normal_directions(cleaning_normals)
                
                cleaning_orientations = normal_estimator.generate_consistent_orientations(
                    positions=cleaning_points,
                    normals=unified_normals
                )
                
                for i, cleaning_idx in enumerate(cleaning_indices):
                    all_orientations[cleaning_idx] = cleaning_orientations[i]
                
                print(f"Successfully calculated orientations for {len(cleaning_orientations)} cleaning points")
                
            except Exception as e:
                print(f"Warning: Failed to calculate surface normals: {e}")
                print("Using default orientations for all points")
        else:
            print(f"Insufficient cleaning points ({len(cleaning_points)}) for normal calculation, using default orientations")
        
        print("Smoothing orientations...")
        try:
            smoothed_orientations = normal_estimator.smooth_orientations(
                orientations=all_orientations,
                smoothing_factor=0.3
            )
            all_orientations = smoothed_orientations
            print("Orientation smoothing completed")
        except Exception as e:
            print(f"Warning: Failed to smooth orientations: {e}")
        
        all_orientations_quaternions = []
        for orientation_matrix in all_orientations:
            try:
                quaternion = self.rotation_matrix_to_quaternion(orientation_matrix)
                all_orientations_quaternions.append(quaternion)
            except:
                default_quat = self.rotation_matrix_to_quaternion(default_orientation)
                all_orientations_quaternions.append(default_quat)
        
        print(f"Final result: {len(all_path_points_3d)} points with calculated orientations")
        
        return {
            'path_3d': all_path_points_3d,
            'orientations': all_orientations_quaternions,
            'segment_types': all_segment_types,
        }
    

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

        self.transformer = CoordinateTransformer(self)
        
        self.create_subscription(Image, '/segmentation_result', self.mask_callback, 10)
        self.create_subscription(Image, '/depth/image_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/depth/camera_info', self.camera_info_callback, 10)
        self.create_subscription(Empty, '/start_cleaning', self.start_cleaning_callback, 10)
        
        self.path_pub = self.create_publisher(Float32MultiArray, '/cleaning_path', 10)

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
                num_rays=self.get_parameter('num_rays').value
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
                
            self.publish_path(base_path, result['path_3d'] ,result['orientations'],result['segment_types'])
            
        except Exception as e:
            self.get_logger().error(f"Path planning failed: {str(e)}")
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")

    def publish_path(self, path_3d, orientations,segment_types):
        
        raw_msg = Float32MultiArray()
        data = []

        SEGMENT_TYPE_MAP = {
            'approach': 0,
            'cleaning': 1,
            'retreat': 2,
        }

        for i in range(len(path_3d)):
            point = path_3d[i]
            orientation = orientations[i] if i < len(orientations) else [1, 0, 0, 0]
            segment_str = segment_types[i] if i < len(segment_types) else 'cleaning'
            segment_index = SEGMENT_TYPE_MAP.get(segment_str, 1)  

           
            data.extend([
                float(point[0]), float(point[1]), float(point[2]),
                float(orientation[1]), float(orientation[2]), float(orientation[3]), float(orientation[0]),
                float(segment_index)
            ])

        raw_msg.data = data
        self.raw_path_pub.publish(raw_msg)
        self.get_logger().info(f"Published raw Float32MultiArray path with {len(path_3d)} points")


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