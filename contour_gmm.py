#!/usr/bin/env python

"contour.py - a module for RT visual debugging tool for the robut."


import numpy as np
from math import degrees, exp, radians 
from calc import Point
import const

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import matplotlib
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter

#SIG_X_OBS = 10
#SIG_Y_OBS = 10
#SIG_XY_OBS = 0
#OMEGA_I_OBS = 0.5
#SIG_X_DEST = 0.02
#SIG_Y_DEST = 0.02
#SIG_XY_DEST = 0



def e(X,Y,cur_x,cur_y,covMat, weight):
    "calculating"
    # calculation based on e(x,xi) = w_i * exp[-(x-xi)' * [covariance_matrix] * (x-xi)]
    SIG_X_OBS = covMat[0][0]
    SIG_XY_OBS = covMat[0][1]
    SIG_Y_OBS = covMat[1][1]

    return weight*np.exp(-(SIG_X_OBS*(X-cur_x)**2 + SIG_Y_OBS *(Y-cur_y)**2 + 2*SIG_XY_OBS*(Y-cur_y)*(X-cur_x)))


def calc_obs_surf(X,Y,mean_x,mean_y, covMat, weight, gmm,there_are_obs):
    "superpositioning all obstacles to form a single obstacles surface contribute"
    "X,Y represent span of x-y. mean_x, mean_y repressent array of means by num of gaussians - n, covMat array of n matrices"
    if there_are_obs==False:
        return 0

    XX = np.array( [ X.ravel(), Y.ravel() ] ).T
    Z = gmm.score_samples(XX)   # yields log likelihood
    Z = np.exp(Z)   # yields Probability density function
    Z = Z.reshape(X.shape)
    means = np.array( [ mean_x, mean_y] ).T
    forPrint = np.exp(gmm.score_samples(means))   
    print("values - for print:", forPrint)
    ### Old - delete soon ###
    #Z = 0
    #
    #for i in range(0,len(mean_x)):
    #    current_x = mean_x[i]
    #    #print("current_x",current_x)
    #    current_y = mean_y[i]
    #   # print("current_y",current_y)
    #    Z += e(X,Y,current_x,current_y,covMat[i], weight[i])
    #    #print("Z",Z)
    
    return Z

def calc_dest_surf(X,Y,goal_x,goal_y):
    "computing goal point paraboloid (sink)"
    # calculation based on e(x,xt) = (x-xt)' * [covariance_matrix] * (x-xt)
    Z = ( const.SIG_X_DEST * ( X - goal_x )**2 + const.SIG_Y_DEST*( Y - goal_y )**2 + 2*const.SIG_XY_DEST*( Y - goal_y )*( X - goal_x ) )
    return Z

#def calc_dir_by_dest(self_pose,dest):
    "computing vector towards target"
    x_direction = - 2 * ( SIG_X_DEST * ( self_pose.x - dest.x ) + SIG_XY_DEST * ( self_pose.y - dest.y ) )
    y_direction = - 2 * ( SIG_Y_DEST * ( self_pose.y - dest.y ) + SIG_XY_DEST * ( self_pose.x - dest.x ) )
    dir_vector = np.array( [ x_direction , y_direction ] )
    return dir_vector

#def single_obs_vector(self_pose, obs_x, obs_y):
    " computing vector to avoid a single obstacle"
    K = SIG_X_OBS * ( self_pose.x - obs_x )**2 + SIG_Y_OBS * ( self_pose.y - obs_y )**2 + 2*SIG_XY_OBS*( self_pose.y - obs_y )*( self_pose.x - obs_x )
    exponent = OMEGA_I_OBS * exp( -K )
    x_direction = exponent * 2 * ( SIG_X_OBS * ( self_pose.x - obs_x ) + SIG_XY_OBS * ( self_pose.y - obs_y ) )
    y_direction = exponent * 2 * ( SIG_Y_OBS * ( self_pose.y - obs_y ) + SIG_XY_OBS * ( self_pose.x - obs_x ) )
    dir_vector = np.array( [ x_direction , y_direction ] )
    return dir_vector


#def calc_dir_by_obs( self_pose, obs_x, obs_y ):
    "calculating the total contibute of the obstacles to the direction vector"
    avoid_obs_vector = np.array( [ 0, 0] )
    K = SIG_X_OBS * ( self_pose.x - obs_x )**2 + SIG_Y_OBS * ( self_pose.y - obs_y )**2 + 2*SIG_XY_OBS*( self_pose.y - obs_y )*( self_pose.x - obs_x )
    exponent = OMEGA_I_OBS * np.exp( -K )
    push_value_x = exponent * 2 * ( SIG_X_OBS * ( self_pose.x - obs_x ) + SIG_XY_OBS * ( self_pose.y - obs_y ) )
    push_value_y = exponent * 2 * ( SIG_Y_OBS * ( self_pose.y - obs_y ) + SIG_XY_OBS * ( self_pose.x - obs_x ) )
    push_value_x = np.sum(push_value_x)
    push_value_y = np.sum(push_value_y)

    avoid_obs_vector = np.array( push_value_x, push_value_y )
    return avoid_obs_vector






def delete_cont( cont ):
    "delete old contour lines and draw new ones"

    for c in cont.collections:
        c.remove()  # removes only the contours, leaves the rest intactplt.draw()
        
        
    return cont


def delete_dots(dotsObject):
    if(dotsObject):
        dotsObject.remove()
   
def plot_dots( x, y, color = 'ro'):
    new_dots, = plt.plot( x, y, color)
    return new_dots

def update_axes( ax, selfPose):
    ax.set_xlim( [selfPose.x - const.RNG_DELTA , selfPose.x + const.RNG_DELTA] )
    ax.set_ylim( [selfPose.y - const.RNG_DELTA, selfPose.y + const.RNG_DELTA] )
    return ax

def update_ranges( selfPose, x_min, x_max, y_min, y_max ):

    x = np.arange( x_min, x_max, 0.02)
    y = np.arange( y_min, y_max, 0.02)
    X,Y = np.meshgrid(x,y)

    return X,Y

def calc_Z(X, Y, mean_x, mean_y, goalPose, covMat, weightArr, gmm, there_are_obs):
    
    Z = calc_obs_surf(X, Y, mean_x, mean_y, covMat, weightArr, gmm, there_are_obs) + calc_dest_surf(X, Y, goalPose.x, goalPose.y) 
    return Z

def update_bounds( self_pose ):
    x_min = self_pose.x - const.RNG_DELTA
    x_max = self_pose.x + const.RNG_DELTA
    y_min = self_pose.y - const.RNG_DELTA
    y_max = self_pose.y + const.RNG_DELTA

    return x_min, x_max, y_min, y_max 


def init_plot(selfPose, goalPose, obs_x, obs_y, mean_obs, covMat, weight, gmm, there_are_obs):
    
    ##### init function
    # set x and y range to compute
    
    x_min, x_max, y_min, y_max = update_bounds( selfPose )

    X,Y = update_ranges( selfPose,
                        x_min, x_max,
                        y_min, y_max)
    
    # calculate the z height of every point in span
    Z = calc_Z( X, Y, mean_obs[:,0], mean_obs[:,1], goalPose, covMat, weight, gmm, there_are_obs)

    ####test:###
    #XX = np.array( [ X.ravel(), Y.ravel() ] ).T
    #Z = gmm.score_samples(XX)
    #Z = Z.reshape(X.shape)
    #####################33

    # setting figure to plot
    fig = plt.figure()
    ax = plt.axes(xlim=(x_min , x_max), ylim=(y_min , y_max), xlabel='x', ylabel='y')
    Nt = 80
    cvals = np.linspace(0,1,Nt+1)      # set contour values 
    ax.set_title('Live data')
    # initial plot
    cont = ax.contourf(X, Y, Z, const.CONTOUR_COUNT)
    selfPoint = plot_dots(selfPose.x, selfPose.y, 'go')
    goalPoint = plot_dots(goalPose.x, goalPose.y )
    obsPoint = plot_dots(obs_x, obs_y)

    return fig, ax, cont, selfPoint, goalPoint, obsPoint


def update_plot(selfPose, goalPose, obs_x, obs_y, means, covMat, weights, cont, selfPoint, goalPoint, obsPoint, ax, gmm, there_are_obs):
    
    x_min, x_max, y_min, y_max = update_bounds( selfPose )
    ax = update_axes( ax, selfPose)
    X,Y = update_ranges( selfPose, x_min, x_max, y_min, y_max )
    Z = calc_Z( X, Y, means[:,0], means[:,1], goalPose, covMat, weights,gmm, there_are_obs)
    ####test:###
#    XX = np.array( [ X.ravel(), Y.ravel() ] ).T
#    Z = gmm.score_samples(XX)
#    Z = Z.reshape(X.shape)
#    forPrint = np.array( [ means[:,0], means[:,1] ] ).T
#    print("gmm value:", gmm.score_samples( forPrint) )
#    #print("gmm 2 value:", gmm.score_samples( [ means[1], means[1][1] ] ) )
#
    #####################33

    # deleting old stuff
    cont = delete_cont( cont )
    delete_dots(selfPoint)
    delete_dots(goalPoint)   #maybe shouldnt delete?
    delete_dots(obsPoint)

    # plotting new data
    cont =  ax.contourf( X, Y, Z, const.CONTOUR_COUNT)
    newSelfPoint = plot_dots(selfPose.x, selfPose.y, color = 'ks' )
    newObsPoint = plot_dots(obs_x, obs_y)
    newGoalPoint = plot_dots(goalPose.x, goalPose.y, color = 'gx')

    return ax, cont, newSelfPoint, newGoalPoint, newObsPoint




if __name__ == '__main__':
    "the main if just for trying new stuff and testing the module"
    goal_x = 4
    goal_y = 4
    Lx = 5
    Ly = 5
    obs_x = np.array([1,3,5])
    obs_y = np.array([1,3,5])
    pose_x = 0
    pose_y = 0
    pose_z = calc_obs_surf(pose_x ,pose_y, obs_x, obs_y) + calc_dest_surf(pose_x, pose_y, goal_x, goal_y)
    origin = [pose_x, pose_y, pose_z] 
    self_pose = Point(pose_x,pose_y)
    goal_pose = Point( goal_x, goal_y )
    Nt = 80

    ax, cont, selfPoint, goalPoint, obsPoint = init_plot( self_pose, goal_pose, obs_x, obs_y)
    plt.pause(1)

    for i in range(10):
        self_pose.x = self_pose.x + 1
        goal_pose.y =goal_pose.y- 1

        ax, cont, selfPoint, goalPoint, obsPoint = update_plot(self_pose, goal_pose,
                                                                        obs_x, obs_y, cont,
                                                                        selfPoint, goalPoint, obsPoint, ax)
        plt.pause(3)
    
    

   
