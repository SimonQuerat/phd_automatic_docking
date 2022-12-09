#!/usr/bin/env python3

# ROS - Python librairies
import rospy
import tf

# cv_bridge is used to convert ROS Image message type into OpenCV images
import cv_bridge

# Import useful ROS types
from sensor_msgs.msg import Image, CameraInfo

# Python librairies
import numpy as np
import cv2
import cv2.aruco as aruco
import tf
from math import *
import time

# import message type MarkerPose
from drone_control.msg import MarkerPose

class Pose:
    def __init__(self):
        """Constructor of the class
        """
        #--- Define the aruco dictionary
        self.aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.parameters  = aruco.DetectorParameters_create()

        #-- Font for the text in the image
        self.font = cv2.FONT_HERSHEY_PLAIN

        # Initialize the bridge between ROS and OpenCV images
        self.bridge = cv_bridge.CvBridge()

        # Extract only one message from the camera_info topic as the camera
        # parameters do not change
        camera_info = rospy.wait_for_message("camera1/camera_info", CameraInfo)
        self.camera_matrix = np.reshape(camera_info.K, (3,3))
        self.camera_distortion = np.array(camera_info.D)

        # print(self.camera_matrix)
        # print(self.camera_distortion)

        #--- Define Tag
        self.id_to_find  = 72
        self.marker_size  = 1/3 #- [m]

        # Souscrire au topic image_raw
        self.sub_image = rospy.Subscriber("camera1/image_raw", Image,
                                          self.callback_image, queue_size=1)

        # Publier sur le topic cmd_vel
        self.pub_pose = rospy.Publisher("marker_pose", MarkerPose, queue_size=1)

    def callback_image(self, msg):
        """Function called each time a new ros Image message is received on
        the camera1/image_raw topic
        Args:
            msg (sensor_msgs/Image): a ROS image sent by the camera
        """
        # Convert the ROS Image into the OpenCV format
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        #cv2.imwrite("image.png", image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #-- remember, OpenCV stores color images in Blue, Green, Red

        #-- Find all the aruco markers in the image
        corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=self.aruco_dict, parameters=self.parameters, 
                                cameraMatrix=self.camera_matrix, distCoeff=self.camera_distortion)

        if ids is not None and ids[0] == self.id_to_find:
            #-- ret = [rvec, tvec, ?]
            #-- array of rotation and position of each marker in camera image
            #-- rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker respect to camera image
            #-- tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera image

            ret = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.camera_distortion)

            #-- Unpack the output, get only the first
            rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
            theta=sqrt(rvec[0]**2+rvec[1]**2+rvec[2]**2) #angle de rotation (angle-axis representation)
            
            #-- Draw the detected marker and put a reference image over it
            aruco.drawDetectedMarkers(image, corners)
            aruco.drawAxis(image, self.camera_matrix, self.camera_distortion, rvec, tvec, 1)

            #-- Print the tag position in camera image
            str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f"%(tvec[0], tvec[1], tvec[2])
            cv2.putText(image, str_position, (0, 100), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            marker_pose=MarkerPose()

            marker_pose.pose.position.x=tvec[0]
            marker_pose.pose.position.y=tvec[1]
            marker_pose.pose.position.z=tvec[2]
            marker_pose.pose.orientation.x=sin(theta/2)*rvec[0]/theta
            marker_pose.pose.orientation.y=sin(theta/2)*rvec[1]/theta
            marker_pose.pose.orientation.z=sin(theta/2)*rvec[2]/theta
            marker_pose.pose.orientation.w=cos(theta/2)

            marker_pose.corners=np.reshape(corners[0][0], (1, 8), 'C').tolist()[0]

            self.pub_pose.publish(marker_pose)
            
        # Display the image
        cv2.imshow("Preview", image)
        cv2.waitKey(1)
 


        
# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    # Declare the node
    rospy.init_node("pose")

    # Instantiate an object
    pose = Pose()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
 
