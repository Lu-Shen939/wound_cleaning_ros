import os
import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from segment_anything import sam_model_registry, SamPredictor
from wound_cleaning_planner import ros_utils as utils


class SimpleSegmentationNode(Node):

    def __init__(self):
        super().__init__('simple_segmentation_node')
        self.br = CvBridge()

        self.get_logger().info("Loading SAM model...")
        cwd = os.path.dirname(__file__)
        sam_checkpoint = os.path.join(cwd, "weights/sam_vit_h_4b8939.pth")
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.get_logger().info("SAM model loaded.")

        self.current_image = None
        self.pos_queries = None
        self.neg_queries = None

        
        self.seg_pub = self.create_publisher(Image, '/segmentation_result', 10)

    
        self.subscription = self.create_subscription(
            Image,
            '/rgb_to_depth/image_raw',
            self._receive_image,
            10)
        
        self.get_logger().info("Simple segmentation node initialized.")

    def _receive_image(self, msg):
    
        try:
            
            image = self.br.imgmsg_to_cv2(msg)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.current_image = image
            self.time_stamp = msg.header.stamp
            
            self._perform_segmentation()
            
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")

    def _perform_segmentation(self):
    
        if self.current_image is None:
            return
            
        self.get_logger().info("Please label keypoints on the image...")
        keypoints, negative = utils.label_keypoint_on_image(self.current_image, 'segmentation')
        
        if len(keypoints) == 0:
            self.get_logger().warning("No positive keypoints selected!")
            return
            
        pos_queries = np.array([kp for kp in keypoints])
        neg_queries = np.array([kp for kp in negative])
        
        input_point = np.concatenate([pos_queries.astype(int), neg_queries.astype(int)], axis=0) if len(neg_queries) > 0 else pos_queries.astype(int)
        input_label = np.array([1] * len(pos_queries) + [0] * len(neg_queries))
        
        self.predictor.set_image(self.current_image)
        mask, _, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
    
        final_mask = self._get_largest_component(mask[0])
        
        self._publish_mask(final_mask)

    def _get_largest_component(self, mask):
       
        from skimage import measure
        
        labels_mask = measure.label(mask)
        if labels_mask.max() == 0:
            return mask
            
        regions = measure.regionprops(labels_mask)
        regions.sort(key=lambda x: x.area, reverse=True)
        
        largest_mask = np.zeros_like(mask)
        if len(regions) > 0:
            largest_region = regions[0]
            largest_mask[largest_region.coords[:, 0], largest_region.coords[:, 1]] = 1
            
        return largest_mask

    def _publish_mask(self, mask):
        try:
    
            mask_img = (mask * 255).astype(np.uint8)
            
            msg = self.br.cv2_to_imgmsg(mask_img, encoding="mono8")
            msg.header.stamp = self.time_stamp
            
            self.seg_pub.publish(msg)
            self.get_logger().info("Segmentation result published.")
            
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to publish mask: {e}")


def main():
    rclpy.init()
    node = SimpleSegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()