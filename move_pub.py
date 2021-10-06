#!/usr/bin/env python

import math
from math import pow, atan2

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from turtlesim.msg import Pose
import numpy as np
import calc
import const
from calc import Point


class OferBotController:

    def __init__(self):
        # Creates a node with name 'OferBotController' and make sure it is a
        # unique node (using anonymous=True).
        rospy.init_node(const.MOVE_NODE_NAME, anonymous=True)

        # Publisher which will publish to the topic '/mybot/cmd_vel'.
        self.velocity_publisher = rospy.Publisher(const.CMD_VEL_TOPIC, Twist, queue_size=10)

        # A subscriber to the topic '/odom_pub'. self.update_pose is called
        # when a message of type Pose is received.
        self.odom_subscriber = rospy.Subscriber(const.ODOMETRY_TOPIC, Odometry, self.update_odom)
        self.goal_subscriber = rospy.Subscriber(const.GOAL_POSE_TOPIC, Pose, self.update_goal)
        self.laser_scan_subscriber = rospy.Subscriber(const.LASER_SCAN_TOPIC, LaserScan, self.update_obs)

        self.odom = Odometry()
        self.goal_pose = Pose()
        self.vel_msg = Twist()
        self.teta = float()
        self.obstacles = []

        self.there_is_goal = bool
        self.rate = rospy.Rate(10)

    def update_odom(self, data):
        # Callback function which is called when a new message of type Odometry is
        # received by the subscriber.
        # print("update_odom has been called")
        self.odom = data
        self.odom.pose.pose.position.x = round(self.odom.pose.pose.position.x, 2)
        self.odom.pose.pose.position.y = round(self.odom.pose.pose.position.y, 2)
        self.odom.pose.pose.orientation.z = round(self.odom.pose.pose.orientation.z, 4)
        self.teta = atan2(2 * self.odom.pose.pose.orientation.w * self.odom.pose.pose.orientation.z,
                          1 - 2 * self.odom.pose.pose.orientation.z ** 2)
        self.teta = calc.fix_angle(self.teta)
        self.teta = round(self.teta,3)

    def update_goal(self, data):
        # CALLBACK function
        # print("update_goal has been called")
        self.goal_pose = data
        self.goal_pose.x = round(self.goal_pose.x, 2)
        self.goal_pose.y = round(self.goal_pose.y, 2)
        self.move2goal()

    def update_obs(self, data):
        """callback function - when a laser scan msg is recieved, check to see if there are obstacles ahead"""
        print("update_obs has been called")
        print(data)
        # if we are not moving, no point to calculate obstacles
        if self.there_is_goal:
            self.search_for_obstacles(data)

    def search_for_obstacles(self, data):
        """ determine obstacles location and add them to obstacles list"""
        for i in range(int(
                math.floor((data.angle_max - data.angle_min) / data.angle_increment))):  # checking all measured angles

            # the current angle of obstacle in self-robot axis system
            beta = data.angle_min + i * data.angle_increment
            beta = round(beta,3)
            # check if there is an obstacle within vision range
            if not (data.range_min < data.ranges[i] < data.range_max):
                continue

            # calculate obstacles position in global system axis
            x = self.odom.pose.pose.position.x
            y = self.odom.pose.pose.position.y
            obs = Point()
            obs.x = x + data.ranges[i] * math.cos(self.teta + 2 * beta)
            obs.x = round(obs.x,2)
            obs.y = y + data.ranges[i] * math.sin(self.teta + 2 * beta)
            obs.y = round(obs.y,2)

            if calc.is_new_point(obs, self.obstacles):  # append to obstacle list, if its a new obstacle
                self.obstacles.append(obs)
                #print(" new obstacle appended: ", obs.x, obs.y)

    def push(self):
        """calculate the PUSH direction, using a known list of obstacles in world axis system """
        push_value = Point()

        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y

        for obstacle in self.obstacles:
            # exponent is i dependent variable
            exponent = math.exp(-(pow(x - obstacle.x, 2) + pow(y - obstacle.y, 2)))

            # sums up the gradients in respect to the obstacles
            push_value.x += -2 * (x - obstacle.x) * exponent * const.EXPONENT_GAIN
            push_value.y += -2 * (y - obstacle.y) * exponent * const.EXPONENT_GAIN
            # ANOTHER OPTION: push_value += -2 * (current_pos - obstacle) * exponent * const.EXPONENT_GAIN
        print("num of obstacles: ", len(self.obstacles))
        #push_value = push_value * -1
        # print("push value: ", push_value)
        del self.obstacles[:]
        return push_value

    def attract(self):
        """ calculate the attraction attribute towards goal pose"""
        attract_value = Point()
        grad_x = 2 * (self.odom.pose.pose.position.x - self.goal_pose.x) / const.SINK_WIDTH_GAIN
        grad_y = 2 * (self.odom.pose.pose.position.y - self.goal_pose.y) / const.SINK_WIDTH_GAIN
        attract_value.x = -grad_x
        attract_value.y = -grad_y
        # print("attract value: ", attract_value)
        return attract_value

    def stop_the_bot(self):
        # setting a 0 [m/s] to all speed parameters and publishing
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
        print('dist to location: ', calc.euclidean_distance(self.odom, self.goal_pose))
        print("Current Position:")
        print("current x= ", self.odom.pose.pose.position.x)
        print("current y= ", self.odom.pose.pose.position.y)
        print("self orientation z (rad)= ", self.teta)

    def move2goal(self):

        # initiating commands before loop
        command_number = 0

        # vel_msg.angular.y = 0
        I_MOVED_FLAG = False

        while calc.euclidean_distance(self.odom, self.goal_pose) >= const.DISTANCE_TOLERANCE:
            I_MOVED_FLAG = True
            self.there_is_goal = True
            self.print_loop_msg(command_number)
            command_number += 1

            # Gradient to the potential energy function.
            gradient =  self.push()  +self.attract() 
            
            # Linear velocity in the x-axis.
            self.vel_msg.linear.x = const.VELOCITY_CONST

            # Angular velocity in the z-axis.
            self.vel_msg.angular.z = const.ANGULAR_GAIN * calc.angular_vel(self.teta, gradient)

            # Publishing vel_msg
            self.velocity_publisher.publish(self.vel_msg)

            # Publish at the desired rate.
            self.rate.sleep()

        # Stopping our robot after the movement is over.
        self.stop_the_bot()

        if I_MOVED_FLAG:
            calc.print_reached_msg()


if __name__ == '__main__':
    try:
        x = OferBotController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
