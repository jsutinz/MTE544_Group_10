import sys
from turtle import stamp

from utilities import Logger

from rclpy.time import Time

from utilities import euler_from_quaternion, calculate_angular_error, calculate_linear_error
from rclpy.node import Node
from geometry_msgs.msg import Twist

from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry as odom

from sensor_msgs.msg import Imu
from kalman_filter import kalman_filter

from rclpy import init, spin, spin_once

import numpy as np
import message_filters

#Additional imports -- Same as Lab 1
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import DurabilityPolicy

rawSensors=0
kalmanFilter=1
odom_qos=QoSProfile(reliability=2, durability=2, history=1, depth=10)

class localization(Node):
    
    def __init__(self, type, dt, loggerName="robotPose.csv", loggerHeaders=["imu_ax", "imu_ay", "kf_ax", "kf_ay","kf_vx","kf_w","odom_v", "odom_w","kf_x", "kf_y","stamp"]):

        # if we want to check odom values too for data collection: loggerHeaders=["imu_ax", "imu_ay", "kf_ax", "kf_ay", "kf_vx", "kf_w","odom_v", "odom_w", "kf_x", "kf_y", "stamp"]

        super().__init__("localizer")

        self.loc_logger=Logger( loggerName , loggerHeaders)
        self.pose=None
        
        if type==rawSensors:
            self.initRawSensors()
        elif type==kalmanFilter:
            self.initKalmanfilter(dt)
        else:
            print("We don't have this type for localization", sys.stderr)
            return  

    def initRawSensors(self):
        self.create_subscription(odom, "/odom", self.odom_callback, qos_profile=odom_qos)
        
    def initKalmanfilter(self, dt):
        
        # TODO Part 3: Set up the quantities for the EKF (hint: you will need the functions for the states and measurements)

        # Start with x as initial values, x = [x,y,th,w,v,vdot]
        
        x= np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Q is the process covariance matrix
        # choosing small values of Q to start off with
        Q= np.diag([0.5,0.5,0.5,0.5,0.5,0.5])

        # R is the measurement covariance matrix
        # As measurements, data from odometry and IMU z = [x,y,th,w,v,ay].
        R = np.diag([1.0, 1.0, 1.0, 1.0])
    # R = np.diag([v, w, ax, ay])


        
        P= np.identity(6) # initial covariance, could not acquire from the Turtlebot, this was acceptable as per Lab TAs
        
        self.kf=kalman_filter(P,Q,R, x, dt)
        
        # TODO Part 3: Use the odometry and IMU data for the EKF

        # message_filters.Subscriber(node, msg_type, topic_name, qos_profile=...), qos_profile is optional

        # Profiles same as Lab 2
        odom_qos= QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=5)
        imu_qos= QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=5)

        # Subscribe to messages as per Lab 2
        self.odom_sub=message_filters.Subscriber(self, odom, "/odom", qos_profile=odom_qos)
        self.imu_sub=message_filters.Subscriber(self, Imu, "/imu", qos_profile=imu_qos)
        
        time_syncher=message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.imu_sub], queue_size=10, slop=0.1)
        time_syncher.registerCallback(self.fusion_callback)
    
    def fusion_callback(self, odom_msg: odom, imu_msg: Imu):
        # TODO Part 3: Use the EKF to perform state estimation
        # Take the measurements
        # your measurements are the linear velocity and angular velocity from odom msg
        # and linear acceleration in x and y from the imu msg
        # the kalman filter should do a proper integration to provide x,y and filter ax,ay

        # z = [v,w,ax,ay].
        # velocity is only in the "x" direction and angular only spins about the z axis
        z= np.array([odom_msg.twist.twist.linear.x, odom_msg.twist.twist.angular.z, imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y])

        # Set stamp values to the same as previous labs
        stamp = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9

        if getattr(self, 'last_stamp', None) is not None:
            dt = stamp - self.last_stamp
            if dt > 0:
                self.kf.dt = dt 
        self.last_stamp = stamp
        # Implement the two steps for estimation
        # Prediction step: from the lecture
        # x = f(x, u) # This is the motion model function
        # P = A*P*A'+Q 
        # # note that Q is the covariance matrix of the states

        self.kf.predict()

        # Update step:
        # S = C*P*C' + R
        # #// note that R is the covariance matrix of the measurements
        # K = P*C'*inv(S)
        # Y_bar = z - h(x) #// h is the measurement function
        # x = x + K*Y_bar
        # P = (1 - K*C)*P

        self.kf.update(z)
        
        # Get the estimate
        xhat=self.kf.get_states()

        # Update the pose estimate to be returned by getPose
        self.pose=np.array([xhat[0], xhat[1], xhat[2], odom_msg.header.stamp],dtype=object)

        # TODO Part 4: log your data
        # self.loc_logger.log_values("imu_ax", "imu_ay", "kf_ax", "kf_ay","kf_vx","kf_w","kf_x", "kf_y","stamp")
        self.loc_logger.log_values([z[2], z[3], xhat[5], xhat[4]*xhat[3], xhat[4], xhat[3],z[0], z[1], xhat[0], xhat[1],stamp])

# if checking the odom values: self.loc_logger.log_values([z[2], z[3], xhat[5], xhat[4]*xhat[3], xhat[4], xhat[3],z[0], z[1], xhat[0], xhat[1], stamp])
    def odom_callback(self, pose_msg):
        
        self.pose=[ pose_msg.pose.pose.position.x,
                    pose_msg.pose.pose.position.y,
                    euler_from_quaternion(pose_msg.pose.pose.orientation),
                    pose_msg.header.stamp]

    # Return the estimated pose
    def getPose(self):
        return self.pose


if __name__=="__main__":
    
    init()
    
    LOCALIZER=localization()
    
    spin(LOCALIZER)
