#!/usr/bin/env python3

# ROS - Python librairies
import rospy
from tf.transformations import quaternion_matrix

# Import useful ROS types
from sensor_msgs.msg import Image, CameraInfo, Imu

from geometry_msgs.msg import TwistStamped
from drone_control.msg import MarkerPose

# Python librairies
import numpy as np
import cv2
import cv2.aruco as aruco
from math import *
import time

class IBVS:
    def __init__(self):
        """Constructor of the class
        """  
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
        self.marker_size  = 1/3 #- [m]
        self.corners_tag=np.array([[-self.marker_size/2, self.marker_size/2, 0], [self.marker_size/2, self.marker_size/2, 0], 
                    [self.marker_size/2, -self.marker_size/2, 0], [-self.marker_size/2, -self.marker_size/2, 0]])

        # position désirée des coins du marqueur dans l'image
        marge=350 #marge en pixels par rapport au bords de l'image, 350 pour le suivi, -200 pour l'appontage
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
        self.Lambda=0.5 #0.5 pour le suivi, 2 pour l'appontage

        # pas de temps
        self.dt=1/20

        # Souscrire au topic marker_pose
        self.sub_pose = rospy.Subscriber("marker_pose", MarkerPose,
                                          self.callback_pose, queue_size=1)

        # Souscrire au topic velocity_body
        self.sub_velocity = rospy.Subscriber("/mavros/local_position/velocity_body", TwistStamped,
                                            self.callback_velocity, queue_size=1)

        # Souscrire au topic /mavros/imu/data
        self.sub_imu = rospy.Subscriber("/mavros/imu/data", Imu,
                                            self.callback_imu, queue_size=1)

        # Publier sur le topic cmd_vel
        self.pub_velocity_cmd = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=1)

    def callback_velocity(self, msg):
        """Function called each time a new mavros velocity message is received on
        the /mavros/local_position/velocity_body topic
        Args:
            msg (geometry_msgs/TwistStamped): a ROS Twist sent by the drone sensors
            """
        self.V_est=np.array([[msg.twist.linear.x],[msg.twist.linear.y],[msg.twist.linear.z],
                        [msg.twist.angular.x],[msg.twist.angular.y],[msg.twist.angular.z]])

    def callback_imu(self, msg):
        """Function called each time a new mavros imu message is received on
        the /mavros/imu/data topic
        Args:
            msg (sensors_msgs/Imu): a mavros Imu sent by the drone sensors
            """
        x=msg.orientation.x
        y=msg.orientation.y
        z=msg.orientation.z
        w=msg.orientation.w
        phi=atan2(2*(w*x+y*z),1-2*(x*2+y*2))
        theta=asin(2*(w*y-z*x))
        self.R_vf=np.dot(np.array([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]]), np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]))

    def callback_pose(self, msg):
        """Function called each time a new ros Image message is received on
        the marker_pose topic
        Args:
            msg (drone_control/MarkerPose): markers corners position and pose estimation sent by the pose node
        """
        #-- Obtain the and translation vector and rotation angles (quaternion)
        tvec=np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        theta=2*acos(msg.pose.orientation.w)
        rvec=np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])*theta/sin(theta/2)

        #-- Obtain the rotation matrix tag->camera
        R_ct    = np.matrix(cv2.Rodrigues(rvec)[0])
        R_tc    = R_ct.T

        #-- Obtain the corners position in the image
        corners=np.reshape(np.array([msg.corners]), (4, 2), 'C')
        
        #-- calcul de la matrice d'intération courante et du vecteur erreur         
        s=np.dot(self.camera_matrix_inv, np.concatenate((corners.T, np.array([[1, 1, 1, 1]])), axis=0)) #positions courantes des coins du marqueurs dans le plan image
        # s=np.dot(self.R_vf, np.dot(self.camera_matrix_inv, np.concatenate((corners.T, np.array([[1, 1, 1, 1]])), axis=0)))
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
        V_cmd[0:3,0]=np.dot(V_cmd[0:3,0], self.R_flip) #passage du repère caméra au repère drone FLU
        V_cmd[3,0]=-V_cmd[3,0]
        V_cmd_ros=TwistStamped() #commande en vitesse pour envoi sur le topic cmd_vel
        V_cmd_ros.twist.linear.x=V_cmd[0,0]
        V_cmd_ros.twist.linear.y=V_cmd[1,0]
        V_cmd_ros.twist.linear.z=V_cmd[2,0]
        V_cmd_ros.twist.angular.z=V_cmd[3,0]

        self.pub_velocity_cmd.publish(V_cmd_ros)           

 
        
# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    # Declare the node
    rospy.init_node("IBVS")

    # Instantiate an object
    IBVS = IBVS()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
 
