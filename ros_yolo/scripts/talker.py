#! /usr/bin/env python3

import rospy
import roslib
import numpy as np
from std_msgs.msg import Float64MultiArray, Header, Float64
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')

def talker():
    pub = rospy.Publisher('chatter', Float64MultiArray, queue_size=10)
    #pub = rospy.Publisher('chatter', numpy_msg(Floats), queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        myarray = Float64MultiArray()
        d = np.random.randint(10, size=4).astype(np.float64)  # only work for 1D array?
        #d = [[float(d[i][j]) for j in range(d.shape[1])] for i in range(d.shape[0])]

        #d = np.array([[3.5, 4.1, 7.8, 9], [10.4, 1, 9.3, 6.7]], dtype=np.float32)
        print(d)
        myarray.data = d
        pub.publish(myarray)
        #pub.publish(d)
        rate.sleep()

if __name__ == '__main__': 
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
