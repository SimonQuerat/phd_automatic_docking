#!/usr/bin/env python3

# ROS - Python librairies
import rospy

# cv_bridge is used to convert ROS Image message type into OpenCV images
import cv_bridge

# Import useful ROS types
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TwistStamped

# Python librairies
import numpy as np
import cv2
import cv2.aruco as aruco
import math

#--- Define the aruco dictionary
aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters  = aruco.DetectorParameters_create()

#-- Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN


class CameraView:
    def __init__(self):
        """Constructor of the class
        """
        # Initialize the bridge between ROS and OpenCV images
        self.bridge = cv_bridge.CvBridge()
        
        #--- 180 deg rotation matrix around the x axis
        self.R_flip  = np.zeros((3,3), dtype=np.float32)
        self.R_flip[0,0] = 1.0
        self.R_flip[1,1] =-1.0
        self.R_flip[2,2] =-1.0

        # Extract only one message from the camera_info topic as the camera
        # parameters do not change
        camera_info = rospy.wait_for_message("camera1/camera_info", CameraInfo)
        self.camera_matrix = np.reshape(camera_info.K, (3,3))
        self.camera_matrix_inv=np.linalg.inv(self.camera_matrix)
        self.camera_distortion = np.array(camera_info.D)

        # print(self.camera_matrix)
        # print(self.camera_distortion)

        #--- Define Tag
        self.id_to_find  = 72
        self.marker_size  = 1 #- [m]
        self.corners_tag=np.array([[-self.marker_size/2, self.marker_size/2, 0], [self.marker_size/2, self.marker_size/2, 0], 
                    [self.marker_size/2, -self.marker_size/2, 0], [-self.marker_size/2, -self.marker_size/2, 0]])

        # position désirée des coins du marqueur dans l'image

        self.fx=self.camera_matrix[0][0]
        self.fy=self.camera_matrix[1][1]
        self.cx=self.camera_matrix[0][2]
        self.cy=self.camera_matrix[1][2]
        m=200 #marge en pixels par rapport au bords de l'image
        self.s_des=np.array([[(m-self.cx)/self.fx],[(m-self.cy)/self.fy],[(800-m-self.cx)/self.fx],[(m-self.cy)/self.fy],
                            [(800-m-self.cx)/self.fx],[(800-m-self.cy)/self.fy],[(m-self.cx)/self.fx],[(800-m-self.cy)/self.fy]])

        # initialisation de l'erreur et de la matrice d'intéraction
        self.L=np.zeros((8,6))
        self.e=np.zeros((8,1))

        # initialisation de la vitesse
        self.V_est=np.zeros((6,1))

        # coeff asservissement visuel
        self.Lambda=0.3

        # pas de temps
        self.dt=1/50

        # Souscrire au topic image_raw
        self.sub_image = rospy.Subscriber("camera1/image_raw", Image,
                                          self.callback_image, queue_size=1)

        # Souscrire au topic velocity_local
        self.sub_velocity = rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped,
                                            self.callback_velocity, queue_size=1)

        # Publier sur le topic cmd_vel
        self.pub_velocity_cmd = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=1)

    def callback_velocity(self, msg):
        """Function called each time a new ros velocity message is received on
        the /mavros/local_position/velocity_local
        Args:
            msg (geometry_msg/TwistStamped): a ROS Twist sent by the drone sensors
        """
        self.V_est=np.array([[msg.twist.linear.x],[msg.twist.linear.y],[msg.twist.linear.z],
                            [msg.twist.angular.x],[msg.twist.angular.y],[msg.twist.angular.z]])

    def callback_image(self, msg):
        """Function called each time a new ros Image message is received on
        the camera1/image_raw topic
        Args:
            msg (sensor_msgs/Image): a ROS image sent by the camera
        """
        # Convert the ROS Image into the OpenCV format
        # global image
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        #cv2.imwrite("image.png", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #-- remember, OpenCV stores color images in Blue, Green, Red

        #-- Find all the aruco markers in the image
        corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters, 
                                cameraMatrix=self.camera_matrix, distCoeff=self.camera_distortion)

        if ids is not None and ids[0] == self.id_to_find:
        
            #-- ret = [rvec, tvec, ?]
            #-- array of rotation and position of each marker in camera image
            #-- rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker respect to camera image
            #-- tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera image
            ret = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.camera_distortion)

            #-- Unpack the output, get only the first
            rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]

            #-- Draw the detected marker and put a reference image over it
            aruco.drawDetectedMarkers(image, corners)
            aruco.drawAxis(image, self.camera_matrix, self.camera_distortion, rvec, tvec, 1)

            #-- Print the tag position in camera image
            global font
            str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f"%(tvec[0], tvec[1], tvec[2])
            cv2.putText(image, str_position, (0, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            #-- Obtain the rotation matrix tag->camera
            R_ct    = np.matrix(cv2.Rodrigues(rvec)[0])
            R_tc    = R_ct.T
            
            #-- calcul de la matrice d'intération et du vecteur erreur

            s=np.dot(self.camera_matrix_inv, np.concatenate((corners[0][0].T, np.array([[1, 1, 1, 1]])), axis=0)) #positions courantes des coins du marqueurs dans le plan image
            Z=np.diag(np.dot((1/s).T, np.dot(R_tc, self.corners_tag.T)+np.array([tvec]).T)/3) #position Z des coins du marqueurs dans le repère caméra

            self.L[0:7:2, 0]=-1/Z #mise à jour de la matrice d'interaction à partir des mesures images
            self.L[0:7:2, 2]=s[0,:]/Z
            self.L[0:7:2, 3]=s[0,:]*s[1,:]
            self.L[0:7:2, 4]=-(1+s[0,:]**2)
            self.L[0:7:2, 5]=s[1,:]
            self.L[1:8:2, 1]=-1/Z
            self.L[1:8:2, 2]=s[1,:]/Z
            self.L[1:8:2, 3]=1+s[1,:]**2
            self.L[1:8:2, 4]=-s[0,:]*s[1,:]
            self.L[1:8:2, 5]=s[0,:]

            s=np.reshape(s[0:2,:], (8,1), 'F') #mise en format vecteur de s

            e=s-self.s_des #erreur courante
            L_TRC=np.concatenate((self.L[:,:3],self.L[:,5:6]),axis=1)
            L_TRC_inv=np.linalg.pinv(L_TRC)
            L_inv=np.linalg.pinv(self.L)
            e_dérivée=(e-self.e)/self.dt
            self.e=e          #erreur précédente
            V_est_cam=self.V_est #vitesse drone repère caméra
            V_est_cam[0:3,0]=np.dot(self.R_flip,V_est_cam[0:3,0])
            V_est_cam[3:6,0]=np.dot(self.R_flip,V_est_cam[3:6,0])
            e_dérivée_partielle_temps=e_dérivée-np.dot(self.L, V_est_cam)
            # print(self.Lambda*np.dot(L_TRC_inv, e))
            # print(np.dot(L_TRC_inv, e_dérivée_partielle_temps))
            V_cmd=-self.Lambda*np.dot(L_TRC_inv, e)#-np.dot(L_TRC_inv, e_dérivée_partielle_temps) #commande en vitesse
            
            V_cmd_ros=TwistStamped() #commande en vitesse pour envoi sur le topic cmd_vel
            V_cmd_ros.twist.linear.x=V_cmd[0,0]
            V_cmd_ros.twist.linear.y=-V_cmd[1,0]
            V_cmd_ros.twist.linear.z=-V_cmd[2,0]
            V_cmd_ros.twist.angular.z=-V_cmd[3,0]
            self.pub_velocity_cmd.publish(V_cmd_ros)
            


        # Display the image
        cv2.imshow("Preview", image)
        cv2.waitKey(5)


        
# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    # Declare the node
    rospy.init_node("IBVS")

    # Instantiate an object
    camera_view = CameraView()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
 
