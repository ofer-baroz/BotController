#!/usr/bin/env python

import numpy as np

ODOMETRY_TOPIC = "/odom"  # blatto - "/odom"
GOAL_POSE_TOPIC = "/goal_pose"
CMD_VEL_TOPIC = "/cmd_vel"  # "/mybot/cmd_vel" # blatto - "/cmd_vel"
LASER_SCAN_TOPIC = '/scan'
MODEL_NAME = "blattoidea"  # "mybot"            # blatto - "blattoidea"
MOVE_NODE_NAME = 'oferbot_controller'
OBS_X_NODE_NAME = 'obstacles_x'
OBS_Y_NODE_NAME = 'obstacles_y'
VELOCITY_CONST = 0.0    #0.18
ANGULAR_GAIN = 0.8    # 0.8
DISTANCE_TOLERANCE = 0.3
EXPONENT_GAIN = 0.05
SINK_WIDTH_GAIN = 4
PLOT_BOUNDRIES = 200
LASER_BOUNDRIES = 1.484
HOKUYO_SAMPLES = 271
RIGHT_HAND_CORRECTION = -1    # the Lidar geometric Z direction. in blattoidea the sensor is up-side-down, so need to multiply in -1
NUM_OF_GMM = 1

### naive approach parameters ##
SIG_X_OBS = 2
SIG_Y_OBS = 2
SIG_XY_OBS = 0
OMEGA_I_OBS = 0.15
covMat = np.array( [SIG_X_OBS,SIG_XY_OBS, SIG_XY_OBS, SIG_Y_OBS]).reshape(2,2)

### contour modul parameters ##
CONTOUR_COUNT = 30
RNG_MIN = -1
RNG_MAX = 5
RNG_DELTA = 3

### any approach parameters for destination ###
SIG_X_DEST = 0.01 # for naiv, value is 2
SIG_Y_DEST = 0.01 #for naiv, value is 2
SIG_XY_DEST = 0

######################################################
TALKER_NAME = "point_goal_publisher"  ##talker is the node that publish the goal point/path
POINT_X = -4
POINT_Y = -4
