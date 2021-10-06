#!/usr/bin/env python

ODOMETRY_TOPIC = "/odom"  # blatto - "/odom"
GOAL_POSE_TOPIC = "/goal_pose"
CMD_VEL_TOPIC = "/cmd_vel"  # "/mybot/cmd_vel" # blatto - "/cmd_vel"
LASER_SCAN_TOPIC = '/scan'
MODEL_NAME = "blattoidea"  # "mybot"            # blatto - "blattoidea"
MOVE_NODE_NAME = 'oferbot_controller'
VELOCITY_CONST = 0.0
ANGULAR_GAIN = 1
DISTANCE_TOLERANCE = 0.25
EXPONENT_GAIN = 0.05
SINK_WIDTH_GAIN = 4
PLOT_RANGE = 6

######################################################
TALKER_NAME = "point_goal_publisher"  ##talker is the node that publish the goal point/path
POINT_X = -4
POINT_Y = -4
