#!/usr/bin/env python3

import rospy

import tf2_ros
import geometry_msgs.msg


if __name__ == '__main__':
    rospy.init_node('tf2_listener')

    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    rate = rospy.Rate(10.0)

    while not rospy.is_shutdown():
        try:
            tf_wb = tf_buffer.lookup_transform('map', "base_link", rospy.Time())
            tf_bc = tf_buffer.lookup_transform("base_link", "camera_link", rospy.Time())
            tf_wc = tf_buffer.lookup_transform("map", "camera_link", rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("BOOM!")
            rate.sleep()
            continue

        #msg.angular.z = 4 * math.atan2(trans.transform.translation.y, trans.transform.translation.x)
        #msg.linear.x = 0.5 * math.sqrt(trans.transform.translation.x ** 2 + trans.transform.translation.y ** 2)

        print("World to base: ", tf_wb)
        print("World to camera: ", tf_wc)
        print("Base to camera1: ", tf_bc)
        print("============")
        new_trans = geometry_msgs.msg.Transform()
        new_trans *= tf_wb.transform * tf_bc.transform
        print("Computed: ", new_trans)

        rate.sleep()
