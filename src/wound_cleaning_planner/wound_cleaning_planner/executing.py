#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from std_msgs.msg import Float32MultiArray, Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MoveGroupAction, MoveGroupGoal, Constraints, PositionConstraint, OrientationConstraint
import numpy as np
import time
from threading import Thread
import tf2_ros
from tf2_geometry_msgs import do_transform_pose

class RobotTrajectoryExecutor(Node):
    def __init__(self):
        super().__init__('robot_trajectory_executor')
        
        self.declare_parameter('robot_name', 'ur5')  
        self.declare_parameter('planning_group', 'manipulator')
        self.declare_parameter('end_effector_link', 'tool0')
        self.declare_parameter('execution_timeout', 60.0)
        self.declare_parameter('position_tolerance', 0.001)  
        self.declare_parameter('orientation_tolerance', 0.05)  
        
        
        self.create_subscription(PoseArray, '/cleaning_path', self.path_callback, 10)
        self.create_subscription(Float32MultiArray, '/path_times', self.times_callback, 10)
        self.create_subscription(Float32MultiArray, '/path_velocities', self.velocities_callback, 10)
        
        self.status_pub = self.create_publisher(Bool, '/cleaning_execution_status', 10)
        self.progress_pub = self.create_publisher(Float32MultiArray, '/cleaning_progress', 10)
        
        self.moveit_client = ActionClient(self, MoveGroup, '/move_group')
        
        self.joint_client = ActionClient(self, FollowJointTrajectory, 
                                       f'/{self.get_parameter("robot_name").value}_arm_controller/follow_joint_trajectory')
        
        self.current_path = None
        self.current_times = None
        self.current_velocities = None
        self.execution_thread = None
        self.is_executing = False
        
        self.get_logger().info("Robot trajectory executor initialized")

    def path_callback(self, msg):
        
        self.current_path = msg
        self.get_logger().info(f"Received cleaning path with {len(msg.poses)} waypoints")
        
        
        if self.current_times is not None and self.current_velocities is not None:
            self.start_execution()

    def times_callback(self, msg):
        
        self.current_times = msg.data
        self.get_logger().info(f"Received timing data with {len(msg.data)} timestamps")

    def velocities_callback(self, msg):
    
        self.current_velocities = msg.data
        self.get_logger().info(f"Received velocity data with {len(msg.data)} velocities")

    def start_execution(self):
    
        if self.is_executing:
            self.get_logger().warn("Already executing trajectory")
            return
            
        if not all([self.current_path, self.current_times, self.current_velocities]):
            self.get_logger().warn("Missing trajectory data")
            return
            
        self.get_logger().info("Starting trajectory execution...")
        self.is_executing = True
        
        self.execution_thread = Thread(target=self.execute_trajectory)
        self.execution_thread.start()

    def execute_trajectory(self):
      
        try:
            
            success = self.execute_with_moveit()
            
            if success:
                self.get_logger().info("Trajectory execution completed successfully")
                self.publish_status(True)
            else:
                self.get_logger().error("Trajectory execution failed")
                self.publish_status(False)
                
        except Exception as e:
            self.get_logger().error(f"Execution error: {str(e)}")
            self.publish_status(False)
        finally:
            self.is_executing = False

    def execute_with_moveit(self):
    
        poses = self.current_path.poses
        total_points = len(poses)
        
        for i, pose in enumerate(poses):
            if not self.is_executing:
                return False
                
            self.get_logger().info(f"Moving to waypoint {i+1}/{total_points}")
            
            goal = MoveGroupGoal()
            goal.request.group_name = self.get_parameter('planning_group').value
            goal.request.num_planning_attempts = 3
            goal.request.allowed_planning_time = 5.0
            goal.request.max_velocity_scaling_factor = self.calculate_velocity_scaling(i)
            goal.request.max_acceleration_scaling_factor = 0.5
            

            pose_constraint = self.create_pose_constraint(pose)
            goal.request.goal_constraints.append(pose_constraint)
            
            future = self.moveit_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            
            if not future.result().accepted:
                self.get_logger().error(f"Goal {i} rejected")
                return False
                
            result_future = future.result().get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            
            if result_future.result().result.error_code.val != 1:  # SUCCESS = 1
                self.get_logger().error(f"Execution failed at waypoint {i}")
                return False
                
            progress = Float32MultiArray()
            progress.data = [float(i+1), float(total_points), float((i+1)/total_points * 100)]
            self.progress_pub.publish(progress)
            
        return True

    def execute_with_joint_trajectory(self):
        
        joint_trajectory = JointTrajectory()
        joint_trajectory.header.stamp = self.get_clock().now().to_msg()
        joint_trajectory.joint_names = self.get_joint_names()  
        
        for i, pose in enumerate(self.current_path.poses):
            point = JointTrajectoryPoint()
            
            joint_angles = self.inverse_kinematics(pose)
            if joint_angles is None:
                self.get_logger().error(f"IK failed for waypoint {i}")
                return False
                
            point.positions = joint_angles
            point.time_from_start.sec = int(self.current_times[i])
            point.time_from_start.nanosec = int((self.current_times[i] % 1) * 1e9)
            
    
            if i < len(self.current_velocities):
            
                point.velocities = self.cartesian_to_joint_velocity(
                    self.current_velocities[i], pose, joint_angles
                )
                
            joint_trajectory.points.append(point)
        
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_trajectory
        
        future = self.joint_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        if not future.result().accepted:
            return False
            
        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        return result_future.result().result.error_code == 0

    def calculate_velocity_scaling(self, waypoint_index):
        
        if waypoint_index >= len(self.current_velocities):
            return 0.1  
            
        current_velocity = self.current_velocities[waypoint_index]
        max_velocity = 0.02 
        
        return min(current_velocity / max_velocity, 1.0)

    def create_pose_constraint(self, pose):
        
        constraints = Constraints()
        
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = self.current_path.header.frame_id
        pos_constraint.link_name = self.get_parameter('end_effector_link').value
        pos_constraint.target_point_offset.x = pose.position.x
        pos_constraint.target_point_offset.y = pose.position.y
        pos_constraint.target_point_offset.z = pose.position.z
        
        pos_constraint.constraint_region.primitive_poses.append(pose)
        
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = self.current_path.header.frame_id
        ori_constraint.link_name = self.get_parameter('end_effector_link').value
        ori_constraint.orientation = pose.orientation
        ori_constraint.absolute_x_axis_tolerance = self.get_parameter('orientation_tolerance').value
        ori_constraint.absolute_y_axis_tolerance = self.get_parameter('orientation_tolerance').value
        ori_constraint.absolute_z_axis_tolerance = self.get_parameter('orientation_tolerance').value
        
        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(ori_constraint)
        
        return constraints

    def publish_status(self, success):
       
        status_msg = Bool()
        status_msg.data = success
        self.status_pub.publish(status_msg)

    def get_joint_names(self):
        
        robot_name = self.get_parameter('robot_name').value
        if 'ur' in robot_name.lower():
            return ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                   'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        else:
    
            return ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

    def inverse_kinematics(self, pose):
        
        pass

    def cartesian_to_joint_velocity(self, cartesian_vel, pose, joint_angles):
        
        pass

def main(args=None):
    rclpy.init(args=args)
    
    executor = RobotTrajectoryExecutor()
    
    try:
        rclpy.spin(executor)
    except KeyboardInterrupt:
        executor.get_logger().info('Shutting down trajectory executor')
    finally:
        executor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()