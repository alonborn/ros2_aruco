"""
This node locates Aruco AR markers in images and publishes their ids and poses.

Subscriptions:
   /camera/image_raw (sensor_msgs.msg.Image)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)

Published Topics:
    /aruco_poses (geometry_msgs.msg.PoseArray)
       Pose of all detected markers (suitable for rviz visualization)

    /aruco_markers (ros2_aruco_interfaces.msg.ArucoMarkers)
       Provides an array of all poses along with the corresponding
       marker ids.

Parameters:
    marker_size - size of the markers in meters (default .0625)
    aruco_dictionary_id - dictionary that was used to generate markers
                          (default DICT_5X5_250)
    image_topic - image topic to subscribe to (default /camera/image_raw)
    camera_info_topic - camera info topic to subscribe to
                         (default /camera/camera_info)

Author: Nathan Sprague
Version: 10/26/2020

"""

import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf_transformations
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from ros2_aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import debugpy
from collections import defaultdict, deque



class ArucoNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("aruco_node")
        self.last_publish_time = self.get_clock().now()  # in __init__

        # Declare and read parameters
        self.declare_parameter(
            name="marker_size",
            value=0.0625,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Size of the markers in meters.",
            ),
        )

        self.declare_parameter(
            name="aruco_dictionary_id",
            value="DICT_5X5_250",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Dictionary that was used to generate markers.",
            ),
        )

        self.declare_parameter(
            name="image_topic",
            value="/camera/image_raw",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Image topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="camera_info_topic",
            value="/camera/camera_info",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera info topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="camera_frame",
            value="",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera optical frame to use.",
            ),
        )

        self.marker_size = (
            self.get_parameter("marker_size").get_parameter_value().double_value
        )
        self.get_logger().info(f"Marker size: {self.marker_size}")

        dictionary_id_name = (
            self.get_parameter("aruco_dictionary_id").get_parameter_value().string_value
        )
        self.get_logger().info(f"Marker type: {dictionary_id_name}")

        image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self.get_logger().info(f"Image topic: {image_topic}")

        info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.get_logger().info(f"Image info topic: {info_topic}")

        self.camera_frame = (
            self.get_parameter("camera_frame").get_parameter_value().string_value
        )

        # Make sure we have a valid dictionary id:
        try:
            dictionary_id = cv2.aruco.__getattribute__(dictionary_id_name)
            if type(dictionary_id) != type(cv2.aruco.DICT_5X5_100):
                raise AttributeError
        except AttributeError:
            self.get_logger().error(
                "bad aruco_dictionary_id: {}".format(dictionary_id_name)
            )
            options = "\n".join([s for s in dir(cv2.aruco) if s.startswith("DICT")])
            self.get_logger().error("valid options: {}".format(options))

        # Set up subscriptions
        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self.info_callback, qos_profile_sensor_data
        )

        self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data
        )

        # Set up publishers
        self.poses_pub = self.create_publisher(PoseArray, "aruco_poses", 10)
        self.markers_pub = self.create_publisher(ArucoMarkers, "aruco_markers", 10)
        
        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)

        self.bridge = CvBridge()

        self.window_size = 5
        self.pose_history = defaultdict(lambda: deque(maxlen=self.window_size))  # or change 3 to any other size

    def info_callback(self, info_msg):
        self.info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)
        # Assume that camera parameters will remain the same...
        self.destroy_subscription(self.info_sub)


    def image_callback(self, img_msg):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return

        current_time = self.get_clock().now()
        # if (current_time - self.last_publish_time).nanoseconds < 0.5 * 1e9:  # 0.5s = 2Hz
        #     return
        self.last_publish_time = current_time


        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        markers = ArucoMarkers()
        pose_array = PoseArray()
        if self.camera_frame == "":
            markers.header.frame_id = self.info_msg.header.frame_id
            pose_array.header.frame_id = self.info_msg.header.frame_id
        else:
            markers.header.frame_id = self.camera_frame
            pose_array.header.frame_id = self.camera_frame

        markers.header.stamp = img_msg.header.stamp
        pose_array.header.stamp = img_msg.header.stamp

        # corners, marker_ids, rejected = cv2.aruco.detectMarkers(
        #     cv_image, self.aruco_dictionary, parameters=self.aruco_parameters
        # )
        corners, marker_ids, rejected = self.aruco_detector.detectMarkers(cv_image)
        if marker_ids is not None:
            if cv2.__version__ > "4.0.0":
                rvecs = []
                tvecs = []

                # Loop over detected markers
                for corner in corners:
                    obj_points = np.array([
                        [-self.marker_size / 2,  self.marker_size / 2, 0],
                        [ self.marker_size / 2,  self.marker_size / 2, 0],
                        [ self.marker_size / 2, -self.marker_size / 2, 0],
                        [-self.marker_size / 2, -self.marker_size / 2, 0]
                    ], dtype=np.float32)

                    img_points = corner[0].astype(np.float32)

                    retval, rvec, tvec = cv2.solvePnP(
                        obj_points,
                        img_points,
                        self.intrinsic_mat,
                        self.distortion
                    )

                    rvecs.append(rvec)
                    tvecs.append(tvec)

            else:
                rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.intrinsic_mat, self.distortion
                )

            
            
            for i, marker_id in enumerate(marker_ids):
                
                marker_id = marker_id[0]
                if marker_id != 1:  
                    continue  # Skip marker ID 1
                    
                # Compute pose
                #position = np.array(tvecs[i][0])
                position = np.array(tvecs[i].reshape(3,))

                rot_matrix = np.eye(4)
                rot_matrix[0:3, 0:3] = cv2.Rodrigues(rvecs[i].flatten())[0]
                quat = tf_transformations.quaternion_from_matrix(rot_matrix)

                # Add to history
                self.pose_history[marker_id].append((position, quat))

                # Compute averaged pose
                poses = self.pose_history[marker_id]
                avg_position = np.mean([p[0] for p in poses], axis=0)
                
                # Average quaternion using Slerp-like method (simplified)
                quats = np.array([p[1] for p in poses])
                avg_quat = quats[0]
                if len(quats) > 1:
                    for q in quats[1:]:
                        avg_quat = tf_transformations.quaternion_slerp(avg_quat, q, 0.5)
                avg_quat = avg_quat / np.linalg.norm(avg_quat)

                # Populate ROS Pose
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = avg_position
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = avg_quat

                pose_array.poses.append(pose)
                markers.poses.append(pose)
                markers.marker_ids.append(marker_id)
                # print(f"Marker ID: {marker_id}, Position: {avg_position}, Orientation: {avg_quat}")

            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)

def main():

    # debugpy.listen(("localhost", 5678))  # Port for debugger to connect
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()  # Ensures the debugger connects before continuing
    # print("Debugger connected.")
        
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
