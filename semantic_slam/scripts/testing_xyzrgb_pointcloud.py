#!/usr/bin/env python3

# PointCloud 2 with default values
# https://answers.ros.org/question/289576/understanding-the-bytes-in-a-pcl2-message/
import rospy
import struct

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

label_rgb_values = {0: 0xFF0000, 1:0x00FF00, 2:0x0000FF, 3:0xFF00FF, 4:0x00FFFF}

def construct_pointcloud():
    points = [[0, 0, 0, 0xFFFFFF],
            [1, 0, 0, 0xFF0000],
            [0, 1, 0, 0x00FF00],
            [0, 0, 1, 0x0000FF],
            [-1, 0, 0, 0x00FFFF]]


    print("Points: ", points)

    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('rgb', 12, PointField.UINT32, 1),
              ]


    header = Header()
    header.frame_id = "map"
    pc2 = point_cloud2.create_cloud(header, fields, points)
    return pc2


if __name__ == "__main__":
    # Initialize ROS
    rospy.init_node("my_xyzrgb_pointcloud_node")
    pub = rospy.Publisher("/xyzrgb_pc", PointCloud2, queue_size=2)

    rate = rospy.Rate(5)    

    my_pc = construct_pointcloud()

    while not rospy.is_shutdown():
        my_pc.header.stamp = rospy.Time.now()
        pub.publish(my_pc)

        rate.sleep()



