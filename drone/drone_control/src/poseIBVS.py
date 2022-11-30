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
import time

class CameraView:
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
        
        #--- 180 deg rotation matrix around the x axis
        self.R_flip  = np.zeros((3,3), dtype=np.float32)
        self.R_flip[0,1] = -1.0
        self.R_flip[1,0] =-1.0
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
        self.marker_size  = 0.5 #- [m]
        self.corners_tag=np.array([[-self.marker_size/2, self.marker_size/2, 0], [self.marker_size/2, self.marker_size/2, 0], 
                    [self.marker_size/2, -self.marker_size/2, 0], [-self.marker_size/2, -self.marker_size/2, 0]])

        # position désirée des coins du marqueur dans l'image
        marge=-200 #marge en pixels par rapport au bords de l'image
        m_des=np.array([[marge], [marge], [800-marge], [marge], [800-marge], [800-marge], [marge], [800-marge]]) #positions désirée en pixels
        self.s_des=np.dot(self.camera_matrix_inv, np.concatenate((np.reshape(m_des, (2, 4), 'F'), np.array([[1, 1, 1, 1]])), axis=0)) #positions désirées dans le plan image
        
        # matrice d'interaction finale
        ret = aruco.estimatePoseSingleMarkers([np.array([np.reshape(m_des, (4,2), 'C')], dtype=np.float32)], self.marker_size, self.camera_matrix, self.camera_distortion) #-- ret = [rvec, tvec, ?]

        rvec, tvec = ret[0][0,0,:], ret[1][0,0,:] #-- Unpack the output, get only the first

        R_ct    = np.matrix(cv2.Rodrigues(rvec)[0]) #-- Obtain the rotation matrix tag->camera
        R_tc    = R_ct.T

        self.L_finale=np.zeros((8,6))

        Z=np.diag(np.dot((1/self.s_des).T, np.dot(R_tc, self.corners_tag.T)+np.array([tvec]).T)/3) #position Z des coins du marqueurs dans le repère caméra
        
        self.L_finale[0:7:2, 0]=-1/Z #mise à jour de la matrice d'interaction à partir des mesures images
        self.L_finale[0:7:2, 2]=self.s_des[0,:]/Z
        self.L_finale[0:7:2, 3]=self.s_des[0,:]*self.s_des[1,:]
        self.L_finale[0:7:2, 4]=-(1+self.s_des[0,:]**2)
        self.L_finale[0:7:2, 5]=self.s_des[1,:]
        self.L_finale[1:8:2, 1]=-1/Z
        self.L_finale[1:8:2, 2]=self.s_des[1,:]/Z
        self.L_finale[1:8:2, 3]=1+self.s_des[1,:]**2
        self.L_finale[1:8:2, 4]=-self.s_des[0,:]*self.s_des[1,:]
        self.L_finale[1:8:2, 5]=-self.s_des[0,:]

        self.s_des=np.reshape(self.s_des[0:2,:], (8,1), 'F')

        # matrice d'intéraction courante 
        self.L_courante=np.zeros((8,6))

        # poids de la matrice d'intéraction courante, compris dans [0, 1]
        self.p = 0.5

        # initialisation de l'erreur
        self.e=np.zeros((8,1))

        # initialisation de la vitesse
        self.V_est=np.zeros((6,1))

        # coeff asservissement visuel
        self.Lambda=0.5

        # pas de temps
        self.dt=1/50

        # Souscrire au topic image_raw
        self.sub_image = rospy.Subscriber("camera1/image_raw", Image,
                                          self.callback_image, queue_size=1)

        # Souscrire au topic velocity_body
        self.sub_velocity2 = rospy.Subscriber("/mavros/local_position/velocity_body", TwistStamped,
                                            self.callback_velocity, queue_size=1)

        # Publier sur le topic cmd_vel
        self.pub_velocity_cmd = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=1)

    def callback_velocity(self, msg):
        """Function called each time a new ros velocity message is received on
        the /mavros/local_position/velocity_body
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

            #-- Draw the detected marker and put a reference image over it
            aruco.drawDetectedMarkers(image, corners)
            aruco.drawAxis(image, self.camera_matrix, self.camera_distortion, rvec, tvec, 1)

            #-- Print the tag position in camera image
            str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f"%(tvec[0], tvec[1], tvec[2])
            cv2.putText(image, str_position, (0, 100), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            #-- Obtain the rotation matrix tag->camera
            R_ct    = np.matrix(cv2.Rodrigues(rvec)[0])
            R_tc    = R_ct.T
            
            #-- calcul de la matrice d'intération courante et du vecteur erreur
            s=np.dot(self.camera_matrix_inv, np.concatenate((corners[0][0].T, np.array([[1, 1, 1, 1]])), axis=0)) #positions courantes des coins du marqueurs dans le plan image         
            Z=np.diag(np.dot((1/s).T, np.dot(R_tc, self.corners_tag.T)+np.array([tvec]).T)/3) #position Z des coins du marqueurs dans le repère caméra

            self.L_courante[0:7:2, 0]=-1/Z #mise à jour de la matrice d'interaction à partir des mesures images
            self.L_courante[0:7:2, 2]=s[0,:]/Z
            self.L_courante[0:7:2, 3]=s[0,:]*s[1,:]
            self.L_courante[0:7:2, 4]=-(1+s[0,:]**2)
            self.L_courante[0:7:2, 5]=s[1,:]
            self.L_courante[1:8:2, 1]=-1/Z
            self.L_courante[1:8:2, 2]=s[1,:]/Z
            self.L_courante[1:8:2, 3]=1+s[1,:]**2
            self.L_courante[1:8:2, 4]=-s[0,:]*s[1,:]
            self.L_courante[1:8:2, 5]=-s[0,:]           

            #-- calcul de la matrice d'intéraction
            self.L=self.p*self.L_courante+(1-self.p)*self.L_finale
 
            s=np.reshape(s[0:2,:], (8,1), 'F') #mise en format vecteur de s

            e=s-self.s_des #erreur courante
            L_TRC=np.concatenate((self.L[:,:3],self.L[:,5:6]),axis=1)
            L_TRC_inv=np.linalg.pinv(L_TRC)
            e_dérivée=(e-self.e)/self.dt
            self.e=e          #erreur précédente  
            V_est_cam=np.reshape(np.dot(self.R_flip, np.reshape(self.V_est, (3,2), 'F')), (6,1), 'F') #vitesse drone repère caméra
            e_dérivée_partielle_temps=e_dérivée-np.dot(self.L_courante, V_est_cam)

            V_cmd=-self.Lambda*np.dot(L_TRC_inv, e)-np.dot(L_TRC_inv, e_dérivée_partielle_temps) #commande en vitesse
            V_cmd[0:3,0]=np.dot(V_cmd[0:3,0], self.R_flip) #possage du repère caméra au repère drone FLU
            V_cmd[3,0]=-V_cmd[3,0]
            V_cmd_ros=TwistStamped() #commande en vitesse pour envoi sur le topic cmd_vel
            V_cmd_ros.twist.linear.x=V_cmd[0,0]
            V_cmd_ros.twist.linear.y=V_cmd[1,0]
            V_cmd_ros.twist.linear.z=V_cmd[2,0]
            V_cmd_ros.twist.angular.z=V_cmd[3,0]

            self.pub_velocity_cmd.publish(V_cmd_ros)           

        # Display the image
        cv2.imshow("Preview", image)
        cv2.waitKey(1)
 


        
# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    # Declare the node
    rospy.init_node("poseIBVS")

    # Instantiate an object
    camera_view = CameraView()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
 
