#!/usr/bin/env python

from math import atan2
import contour_gmm as ctg
import rospy
from numpy.linalg import inv
from math import pi
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from turtlesim.msg import Pose
from geometry_msgs.msg import PoseArray
import numpy as np

import calc
import const
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

class OferBotController:

    def __init__(self):
        # Creates a node with name 'OferBotController' and make sure it is a
        # unique node (using anonymous=True).
        print("Initiated")

        rospy.init_node(const.MOVE_NODE_NAME, anonymous=True)

        # Publisher which will publish to the topic '/mybot/cmd_vel'.
        self.velocity_publisher = rospy.Publisher(const.CMD_VEL_TOPIC, Twist, queue_size=10)
        
        
        
        # A subscriber to the topic '/odom_pub'. self.update_pose is called
        # when a message of type Pose is received.
        self.odom_subscriber = rospy.Subscriber(const.ODOMETRY_TOPIC, Odometry, self.update_odom)
        self.goal_subscriber = rospy.Subscriber(const.GOAL_POSE_TOPIC, PoseArray, self.update_goalSet)
        self.laser_scan_subscriber = rospy.Subscriber(const.LASER_SCAN_TOPIC, LaserScan, self.update_obs)
        
        self.odom = Odometry()
        self.goal_pose = Pose()
        self.goal_set = PoseArray()
        self.vel_msg = Twist()
        self.teta = float()
        #self.obstacles = []
        self.obs_x = np.array([1])
        self.obs_y = np.array([1])
        self.laser_angles = np.array([1])
        self.there_is_goal = bool
        self.there_are_obs = bool
        self.rate = rospy.Rate(10)
        self.gmm = GaussianMixture()

    

    def update_odom(self, data):
        # Callback function which is called when a new message of type Odometry is
        # received by the subscriber.
        
        #print("update_odom has been called - updating self location")
        #print(data)
        self.odom = data
        self.odom.pose.pose.position.x = round(self.odom.pose.pose.position.x, 2)
        self.odom.pose.pose.position.y = round(self.odom.pose.pose.position.y, 2)
        self.odom.pose.pose.orientation.z = round(self.odom.pose.pose.orientation.z, 4)
        self.teta = atan2(2 * self.odom.pose.pose.orientation.w * self.odom.pose.pose.orientation.z,
                          1 - 2 * self.odom.pose.pose.orientation.z ** 2)
        self.teta = calc.fix_angle(self.teta)
        self.teta = round(self.teta,3)
        #print("self location is updated.")

    def update_goalSet(self, data):
        self.goal_set = data
        
        for goalPoint in self.goal_set.poses:
            #print(goalPoint)
            self.update_goalPoint(goalPoint)

        

    def update_goalPoint(self, data):
        # CALLBACK function
        #print("update_goalPoint has been called - updating next point location")
        #print("data:")
        #print(data)
        #print(data.position)
        self.goal_pose = data.position 
        self.goal_pose.x = round(self.goal_pose.x, 2)
        self.goal_pose.y = round(self.goal_pose.y, 2)
        #print("next point location has been updated.")
        self.move2goal()

    def update_obs(self, data):
        """callback function - when a laser scan msg is recieved, check to see if there are obstacles ahead"""
        #print("update_obs has been called - updating self.obs")
        #print(data)
        # if we are not moving, no point to calculate obstacles
        if self.there_is_goal:
            #print("there is goal. calculating new obs")
            self.laser_angles = np.linspace( data.angle_min, data.angle_max, const.HOKUYO_SAMPLES )
            
             #locates indices holding inf == meaning lidar did not detect anything in that angle
            bad_indx = np.nonzero( np.isinf( data.ranges ) )  
                        
            rng = np.delete( data.ranges , bad_indx )

            if rng.size: # only apply if saw obs
                self.there_are_obs = True

                # beta is laser_angle only for relevant measures
                # need to multiply in (-1) in order to set beta in respect to right-handed system[]

                beta = const.RIGHT_HAND_CORRECTION * np.delete( self.laser_angles, bad_indx )  
            
            
                # calculate obstacles position in global system axis
                x = self.odom.pose.pose.position.x
                y = self.odom.pose.pose.position.y
            
                self.obs_x = x+rng*np.cos(self.teta+beta)
                self.obs_y = y+rng*np.sin(self.teta+beta)
            

                listOfObs = np.transpose( np.array( [ self.obs_x, self.obs_y] ) )
                self.gmm = GaussianMixture(n_components=const.NUM_OF_GMM, covariance_type="full", random_state=0).fit(listOfObs)
            else:
                self.there_are_obs = False

            #print("listofObs:",listOfObs)
            
            #print("iterations:", self.gmm.n_iter_)
            #print("means:", self.gmm.means_)
            #print("weights:", self.gmm.weights_)
            #print("covariance:", self.gmm.covariances_)
            #
            
        
            #print("self.obs has been calculated.")
            
    




    def push(self):
        """calculate the PUSH direction, using a known list of obstacles in world axis system """
        print("PUSH has been called. calculating avoid_obs_vector")
        
        avoid_obs_vector = np.array( [ 0, 0] )
        if self.there_are_obs is False:
            return

        X = np.array( [ self.odom.pose.pose.position.x, self.odom.pose.pose.position.y ] ).reshape(1,2)
    
        precMat = self.gmm.precisions_
        covMat = self.gmm.covariances_
        means = self.gmm.means_
        
        
        for i in range(const.NUM_OF_GMM):
            mu = means[i]
            Sxx = covMat[i][0][0]
            Syy = covMat[i][1][1]
            Sxy  = covMat[i][0][1]
            rho = (Sxx*Syy-Sxy**2)
            w = 1/(2*pi*rho**0.5)
            K =  -0.5* np.matmul( np.matmul( (X-mu) ,precMat) , (X-mu).T)  ## == -0.5 * (X-mu) @ precMat @ (X-mu).T
            coefficient = w*np.exp(K)
            avoid_obs_vector = coefficient *  np.matmul( (X-mu) , precMat ) ## == coefficient * (X-mu) @ precMat
                

        #print(" avoid _obs_vector calculated. result is:") 
        #print(avoid_obs_vector)
        return avoid_obs_vector



       

    


    def attract(self):
    
        """ calculate the attraction attribute towards goal pose"""
        print("ATTRACT has been called. calculating dir_vector")
        self_x = self.odom.pose.pose.position.x
        self_y = self.odom.pose.pose.position.y
        x_direction = - 2 * ( const.SIG_X_DEST * ( self_x - self.goal_pose.x ) + const.SIG_XY_DEST * ( self_y - self.goal_pose.y ) )
        y_direction = - 2 * ( const.SIG_Y_DEST * ( self_y - self.goal_pose.y ) + const.SIG_XY_DEST * ( self_x - self.goal_pose.x ) )
        dir_vector = np.array( [ x_direction , y_direction ] )
        #print("dir_vector calculated. result is:")
        #print(dir_vector)
        return dir_vector
        

    def stop_the_bot(self):
        # setting a 0 [m/s] to all speed parameters and publishing
        print("STOP THE BOT")
        self.vel_msg.linear.x = 0
        self.vel_msg.linear.y = 0
        self.vel_msg.linear.z = 0
        self.vel_msg.angular.z = 0  # *rad/sec
        self.velocity_publisher.publish(self.vel_msg)
        self.there_is_goal = False

    def print_loop_msg(self, command_number):

        print('========= command number is:', command_number, '=========')
        print("current goal destination is:")
        print(self.goal_pose.x, self.goal_pose.y)
        print('dist to location: ')
        print(calc.euclidean_distance(self.odom, self.goal_pose))
        print("Current Position:")
        print("current x= ")
        print(self.odom.pose.pose.position.x)
        print("current y= ")
        print(self.odom.pose.pose.position.y)
        print("self orientation z (rad)= ")
        print(self.teta)




    def move2goal(self):
        
       # print(self.goal_set)
       # if self.goal_set.poses:
       #     self.update_goalPoint(self.goal_set.poses[0])
       # else:
       #     return

        print("moving 2 goal initiated")
        # initiating commands before loop
        command_number = 0

        # vel_msg.angular.y = 0
        I_MOVED_FLAG = False
        
        #### Optional: plot visual data for debugging: #######
        print("initiating contour graphs")
        fig, ax, cont, selfPoint, goalPoint, obsPoint = ctg.init_plot( self.odom.pose.pose.position, self.goal_pose,
                                                                self.obs_x, self.obs_y,self.gmm.means_,self.gmm.covariances_,self.gmm.weights_,self.gmm,self.there_are_obs)
        ###################
        

        while calc.euclidean_distance(self.odom, self.goal_pose) >= const.DISTANCE_TOLERANCE:
            I_MOVED_FLAG = True
            self.there_is_goal = True
            self.print_loop_msg(command_number)
            command_number += 1
            pose_x = self.odom.pose.pose.position.x
            pose_y = self.odom.pose.pose.position.y

            # Gradient to the potential energy function.
            gradient =   (self.attract() + self.push() ).reshape(2,)
            print("gradient vector is:")
            print(gradient)
            
            #### Optional: plot visual data for debugging: #######
            print("updating contour graphs")
            ax, cont, selfPoint, goalPoint, obsPoint = ctg.update_plot(self.odom.pose.pose.position, 
                                                                        self.goal_pose,
                                                                        self.obs_x, self.obs_y,
                                                                        self.gmm.means_,self.gmm.covariances_,self.gmm.weights_,
                                                                        cont, selfPoint, goalPoint, obsPoint, ax, self.gmm,self.there_are_obs)
            plt.pause(0.001)
            ############################################################

            # Linear velocity in the x-axis.
            self.vel_msg.linear.x = const.VELOCITY_CONST

            # Angular velocity in the z-axis.
            self.vel_msg.angular.z = const.ANGULAR_GAIN * calc.angular_vel( self.teta, gradient )

            # Publishing vel_msg
            self.velocity_publisher.publish(self.vel_msg)
            #print("vel_msg has been published")

            # Publish at the desired rate.
            self.rate.sleep()

        # Stopping our robot after the movement is over.
        self.stop_the_bot()
        plt.close(fig)
        
        #goal point reached - remove from goal_set
        #if not (self.goal_set == 0):
       #     del self.goal_set.poses[0]

        if I_MOVED_FLAG:
            calc.print_reached_msg()


if __name__ == '__main__':
    try:
        x = OferBotController()

        
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
