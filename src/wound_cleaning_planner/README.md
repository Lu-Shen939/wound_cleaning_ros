1 create workspace

source /opt/ros/humble/setup.bash

mkdir -p ~/wound_cleaning_ws/src
cd ~/wound_cleaning_ws
cd ~/wound_cleaning_ws/src

ros2 pkg create wound_cleaning_planner \
  --build-type ament_python \
  --dependencies rclpy sensor_msgs geometry_msgs

cd /wound_cleaning_ws/src/wound_cleaning_planner/wound_cleaning_planner
mkdir weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
mv sam_vit_h_4b8939.pth weights/


2 Renew workspace:

cd ~/wound_cleaning_ws && colcon build --symlink-install && source install/setup.bash

3 run nodes:

source ~/wound_cleaning_ws/install/setup.bash && ros2 run wound_cleaning_planner ros_segmentation

source ~/wound_cleaning_ws/install/setup.bash && ros2 run wound_cleaning_planner wound_cleaning_planner

source ~/wound_cleaning_ws/install/setup.bash && ros2 run wound_cleaning_planner test_start_cleaning

4  Receiving results:

ros2 topic echo /cleaning_path
ros2 topic echo /path_times

5 Do visualization: 
source ~/wound_cleaning_ws/install/setup.bash && ros2 run wound_cleaning_planner visualize_path
-----------------------------------------------------------------------------------------------------------------------------------
depth camera launching:

source install/setup.bash
ros2 launch azure_kinect_ros_driver driver.launch.py
rviz2





