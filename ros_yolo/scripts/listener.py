#!/usr/bin/env python

import roslib

import rospy
from rospy_tutorials.msg import Floats
from std_msgs.msg import Float64MultiArray
from rospy.numpy_msg import numpy_msg

def callback(data):
    print(rospy.get_name(), "I heard %s"%str(data.data))

def listener():
    rospy.init_node('listener')
    #rospy.Subscriber("chatter", numpy_msg(Floats), callback)
    rospy.Subscriber("chatter", Float64MultiArray, callback)
    
    rospy.spin()

if __name__ == '__main__':
    listener()