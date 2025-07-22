import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from apriltag_msgs.msg import AprilTagDetectionArray
import numpy as np
import cv2
import tf_transformations
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

class HandEyeCalibNode(Node):
    def __init__(self):
        super().__init__('hand_eye_calib_node')
        
        self.create_subscription(
            AprilTagDetectionArray,
            '/tag_detections',
            self.tag_callback,
            10)
        
        self.create_subscription(
            PoseStamped,
            '/robot/end_effector_pose',
            self.ee_pose_callback,
            10)
        
        self.tag_pose = None
        self.ee_pose = None
        
        self.data_pairs = []  
        
        self.timer = self.create_timer(1.0, self.collect_data)
        
        self.broadcaster = StaticTransformBroadcaster(self)
        
        self.calibrated = False

    def tag_callback(self, msg: AprilTagDetectionArray):
        if len(msg.detections) == 0:
            return
        
        detection = msg.detections[0]
        pose = detection.pose.pose
        self.tag_pose = pose

    def ee_pose_callback(self, msg: PoseStamped):
        self.ee_pose = msg.pose

    def collect_data(self):
        if self.tag_pose is None or self.ee_pose is None:
            self.get_logger().info('Waiting for tag and end-effector poses...')
            return
        
        R_gripper, t_gripper = self.pose_to_rt(self.ee_pose)
        R_tag, t_tag = self.pose_to_rt(self.tag_pose)

        self.data_pairs.append((R_gripper, t_gripper, R_tag, t_tag))
        self.get_logger().info(f'Collected {len(self.data_pairs)} pairs')

        if len(self.data_pairs) >= 15 and not self.calibrated:
            self.calibrate_hand_eye()

    def pose_to_rt(self, pose):
        q = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        R = tf_transformations.quaternion_matrix(q)[:3, :3]
        t = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ])
        return R, t

    def calibrate_hand_eye(self):
        self.get_logger().info('Starting hand-eye calibration...')

        R_gripper_list = []
        t_gripper_list = []
        R_tag_list = []
        t_tag_list = []

        for (R_g, t_g, R_t, t_t) in self.data_pairs:
            R_gripper_list.append(R_g)
            t_gripper_list.append(t_g)
            R_tag_list.append(R_t)
            t_tag_list.append(t_t)

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper_list,
            t_gripper_list,
            R_tag_list,
            t_tag_list,
            method=cv2.CALIB_HAND_EYE_TSAI)

        self.get_logger().info('Calibration finished.')
        self.get_logger().info(f'Rotation:\n{R_cam2gripper}')
        self.get_logger().info(f'Translation:\n{t_cam2gripper}')

        self.publish_static_transform(R_cam2gripper, t_cam2gripper)

        self.calibrated = True

    def publish_static_transform(self, R, t):
        quat = tf_transformations.quaternion_from_matrix(np.vstack((
            np.hstack((R, t.reshape(3,1))),
            np.array([0,0,0,1])
        )))

        static_tf = TransformStamped()
        static_tf.header.stamp = self.get_clock().now().to_msg()

        
        static_tf.header.frame_id = 'ee_link'            
        static_tf.child_frame_id = 'camera_optical_frame'  

        static_tf.transform.translation.x = t[0]
        static_tf.transform.translation.y = t[1]
        static_tf.transform.translation.z = t[2]
        static_tf.transform.rotation.x = quat[0]
        static_tf.transform.rotation.y = quat[1]
        static_tf.transform.rotation.z = quat[2]
        static_tf.transform.rotation.w = quat[3]

        self.broadcaster.sendTransform(static_tf)
        self.get_logger().info('Published static transform from ee_link to camera_optical_frame')

def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
