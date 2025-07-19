import time
import rclpy
import tf2_ros
import numpy as np
np.float = np.float64
import ros2_numpy as rnp
import transforms3d as tf3d
import matplotlib.pyplot as plt
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Transform, Pose, Quaternion
from sensor_msgs.msg import PointCloud2
from rclpy.node import Node
from rclpy.task import Future
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

def get_transform(node, source_frame: str, target_frame: str, timeout: float = None):
    """
    Get a transform between two frames.

    :param source_frame: The source frame name.
    :param target_frame: The target frame name.
    :param timeout: Optional timeout in seconds to wait for the transform.
    :return: The TransformStamped object containing the transform, or None if not found.
    """

    # Set up a Buffer and TransformListener
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)
    
    end_time = node.get_clock().now() + Duration(seconds=timeout) if timeout else None

    while rclpy.ok():
        # If there's a timeout specified, break if time is up
        if end_time and node.get_clock().now() > end_time:
            node.get_logger().error(f"Timeout reached while waiting for transform from '{source_frame}' to '{target_frame}'")
            return None

        # Look up the transform
        try:
            transform = tf_buffer.lookup_transform(target_frame, source_frame, Time())
            return transform
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # node.get_logger().warn(f"Waiting for transform from '{source_frame}' to '{target_frame}': {e}")
            pass
        
        # Sleep for a short duration before retrying
        rclpy.spin_once(node, timeout_sec=0.1)


def rot_matrix_to_quaternion(mat):
    quat_ = tf3d.quaternions.mat2quat(mat)
    quat = Quaternion()
    quat.x = quat_[1]
    quat.y = quat_[2]
    quat.z = quat_[3]
    quat.w = quat_[0]
    return quat

def quaternion_to_rot_matrix(quat):
    quat_ = [quat.w, quat.x, quat.y, quat.z]
    mat = tf3d.quaternions.quat2mat(quat_)
    return mat

def matrix_to_pose(mat):
    pose = Pose()
    pose.position.x = mat[0, 3]
    pose.position.y = mat[1, 3]
    pose.position.z = mat[2, 3]
    quat = rot_matrix_to_quaternion(mat[:3, :3])
    pose.orientation = quat
    return pose

def matrix_to_transform(mat):
    transform = Transform()
    transform.translation.x = mat[0, 3]
    transform.translation.y = mat[1, 3]
    transform.translation.z = mat[2, 3]
    quat = rot_matrix_to_quaternion(mat[:3, :3])
    transform.rotation = quat
    return transform

def pose_to_matrix(pose):
    mat = np.eye(4)
    mat[:3, :3] = quaternion_to_rot_matrix(pose.orientation)
    mat[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return mat

def transform_to_matrix(transform):
    mat = np.eye(4)
    mat[:3, :3] = quaternion_to_rot_matrix(transform.rotation)
    mat[:3, 3] = [transform.translation.x, transform.translation.y, transform.translation.z]
    return mat

def pointcloud2_to_numpy(msg):
    pcl = rnp.numpify(msg)
    pcl = np.stack([np.array(pcl['x']), np.array(pcl['y']), np.array(pcl['z'])], axis=1).astype(np.float32)
    return pcl

def numpy_to_pointcloud2(points, time_stamp, frame_id):

    pc_array = np.zeros(len(points), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
    ])
    pc_array['x'] = points[:, 0]
    pc_array['y'] = points[:, 1]
    pc_array['z'] = points[:, 2]
    pc_array['intensity'] = points[:, 0]

    pc_msg = rnp.msgify(PointCloud2, pc_array, stamp=time_stamp, frame_id=frame_id)
    return pc_msg


def convert_time_stamp(time_stamp):
    seconds = time_stamp.sec
    nanoseconds = time_stamp.nanosec
    timestamp_in_seconds = seconds + nanoseconds / 1e9

    # Convert to human-readable format
    human_readable_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(timestamp_in_seconds))
    return human_readable_time



def wait_for_message(node: Node, topic, msg_type, timeout=None):
    """
    Waits for a single message on the specified topic.

    :param node: The rclpy node instance.
    :param topic: The topic name to subscribe to.
    :param msg_type: The message type (e.g., std_msgs.msg.String).
    :param timeout: Timeout in seconds, or None for no timeout.
    :return: The received message, or None if the timeout occurs.
    """
    future = Future()

    def callback(msg):
        if not future.done():
            future.set_result(msg)

    # Create a temporary subscription
    subscription = node.create_subscription(msg_type, topic, callback, 10)

    # Spin until a message is received or timeout occurs
    try:
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    finally:
        # Clean up subscription
        node.destroy_subscription(subscription)

    return future.result() if future.done() else None


def label_keypoint_on_image(image, name=''):
    keypoints = []
    negative_keypoints = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.title.set_text(f'{name} keypoints')

    def onclick(event):
        if event.button == 1:  # Left-click for positive keypoints
            ix, iy = event.xdata, event.ydata
            print(f'Positive keypoint: x = {ix}, y = {iy}')
            ax.scatter(ix, iy, c='r', label='Positive' if len(keypoints) == 0 else "")
            keypoints.append([ix, iy])
        elif event.button == 3:  # Right-click for negative keypoints
            ix, iy = event.xdata, event.ydata
            print(f'Negative keypoint: x = {ix}, y = {iy}')
            ax.scatter(ix, iy, c='b', label='Negative' if len(negative_keypoints) == 0 else "")
            negative_keypoints.append([ix, iy])
        plt.legend()
        plt.draw()
        
    fig.canvas.mpl_connect('button_press_event', onclick)
    ax.imshow(image)
    plt.show()
    return keypoints, negative_keypoints
