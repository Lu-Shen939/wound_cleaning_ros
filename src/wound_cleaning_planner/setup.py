from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'wound_cleaning_planner'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include Python modules in the package
        (os.path.join('share', package_name, package_name), 
         glob(f'{package_name}/*.py')),
        # Include scripts if they exist
        (os.path.join('share', package_name, 'scripts'), 
         glob('scripts/*.py') if os.path.exists('scripts') else []),
        # Include launch files if they exist
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.py') if os.path.exists('launch') else []),
        # Include weights directory for SAM model
        (os.path.join('share', package_name, 'weights'),
         glob('weights/*') if os.path.exists('weights') else []),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'sensor_msgs',
        'geometry_msgs', 
        'std_msgs',
        'tf2_ros',
        'cv_bridge',
        'opencv-python',
        'numpy',
        'scipy',
        'scikit-image',
        'matplotlib',
        'torch',
        'torchvision',
        'segment-anything',
        'transforms3d',
        'ros2_numpy',
        'Pillow',
    ],
    zip_safe=True,
    maintainer='ls2244',
    maintainer_email='ls2244@cornell.edu',
    description='Advanced wound cleaning path planner for surgical robotics',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            
            'wound_cleaning_planner = wound_cleaning_planner.wound_cleaning_planner_node:main',
            'ros_segmentation = wound_cleaning_planner.ros_segmentation:main',
            'test_start_cleaning = wound_cleaning_planner.test_start_cleaning:main',
            'visualize_path = wound_cleaning_planner.visualize_path:main',
        ],
    },
)