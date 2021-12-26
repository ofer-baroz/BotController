#!/usr/bin/env python
from math import pi, pow, sqrt
import matplotlib.pyplot as plt

import numpy as np

#from scripts.const import PLOT_RANGE
def calc_avoid_obs_vector(X,precMat, covMat, means):
            avoid_obs_vector = np.array( [0,0] )
            mu = means
            Sxx = covMat[0][0]
            Syy = covMat[1][1]
            Sxy  = covMat[0][0]
            rho = (Sxx*Syy-Sxy**2)
            w = 1/(2*pi*rho**0.5)
            K = -0.5 * np.matmul( np.matmul( (X-mu) ,precMat ) , (X-mu).T)
            coefficient = w*np.exp(K)
            avoid_obs_vector += coefficient * np.matmul( (X-mu) , precMat )
            
            return avoid_obs_vector

def euclidean_distance(odom, goal_pose):
    # Euclidean distance between current location and the goal
    dist = sqrt(pow((goal_pose.x - odom.pose.pose.position.x), 2) + pow((goal_pose.y - odom.pose.pose.position.y), 2))
    return dist


def fix_angle(angle):
    # make sure angle is positive in respect to x axis
    if angle < 0:
        angle = angle + 2 * pi
    return angle


def next_goal_angle(grad_x, grad_y):
    # calculating direction in global axis system. does not consider self orientation
    ng_angle = np.arctan2(grad_y, grad_x)
    ng_angle = fix_angle(ng_angle)
    print('steering angle is:', ng_angle)
    print("steering vector: ", grad_x, grad_y)
    return ng_angle


def angular_vel(current_orientation, direction_vector):
    # calculating angular velocity. determine weather to go CW or CWC. 
    # all angles must be positive in respect to x axis
    alpha = next_goal_angle(direction_vector[0], direction_vector[1])
    teta = current_orientation

    if abs(alpha - teta) > pi:
        if alpha > teta:
            a_vel = -(teta + (2 * pi - alpha))
        else:
            a_vel = (alpha + (2 * pi - teta))
    else:
        a_vel = alpha - teta
    a_vel = round(a_vel,2)
    print("angular velocity is: ", a_vel)
    
    return a_vel


class Point:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if type(other) == float or type(other) == int:
            return Point(self.x * other, self.y * other)

        raise NotImplementedError

    def __str__(self):
        return "({}, {})".format(self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


def is_new_point(given_obs, given_list):
    """a function to verify weather a given point is already exist in a given list"""

    if not given_list:
        return True

    

    for p in given_list:
        #print("p = ", p.x , p.y)
        #print("given_obs = ", given_obs.x , given_obs.y)
        if p == given_obs:
            return False

    return True


def print_reached_msg():
    print("==============================")
    print("goal point reached.")
    print("waiting for a different goal.")
    print("==============================")

def plot_obs(x_array,y_array):
    plt.plot(x_array, y_array, 'ro')
    plt.show()

#def plot_gradient(gradient,cur_x,cur_y):
#    
#    x = np.linspace(cur_x-PLOT_RANGE,cur_x+PLOT_RANGE, 30)
#    y = np.linspace(cur_y-PLOT_RANGE,cur_y+PLOT_RANGE, 30)
#
#    X, Y = np.meshgrid(x, y)    
#    Z = gradient
#
#    fig = plt.figure()
#    ax = plt.axes(projection='3d')
#    ax.contour3D(X, Y, Z, 50, cmap='binary')
#    ax.set_xlabel('x')
#    ax.set_ylabel('y')
#    ax.set_zlabel('z')
#    plt.show()