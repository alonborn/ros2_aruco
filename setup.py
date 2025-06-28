from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'ros2_aruco'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nathan Sprague',
    maintainer_email='nathan.r.sprague@gmail.com',
    description='ROS2 ArUco marker detection package',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'aruco_node = ros2_aruco.aruco_node:main',
            'aruco_generate_marker = ros2_aruco.aruco_generate_marker:main'
        ],
    },
)