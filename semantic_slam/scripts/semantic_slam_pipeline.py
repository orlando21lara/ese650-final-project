#!/usr/bin/env python3

import rospy
import tf2_ros
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

import numpy as np
import cv2
from scipy.spatial.distance import cdist
import random
import copy

from semantic_process import *
from bb_match import *

import sys
sys.path.remove('/opt/ros/noetic/lib/python3/dist-packages')

Labels = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # all the possible labels, keys are the class number in COCO
label_rgb_values = {0:0xFF0000, 1:0x00FF00, 2:0xFC8803, 3:0xFF00FF, 4:0x873E23, 5:0x76B5C5}

def left_image_callback(img_msg):
    global left_img
    global left_img_received

    if not left_img_received:
        left_img_received = True
        left_img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)

def right_image_callback(img_msg):
    global right_img
    global right_img_received

    if not right_img_received:
        right_img_received = True
        right_img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)

def left_pred_callback(pred_msg):
    global left_pred
    global left_pred_received

    if not left_pred_received:
        left_pred_received = True
        left_pred = pred_msg.data      # This extracts the data from the ros msg as a tuple
        left_pred = np.array(left_pred).reshape(-1, 6)

def right_pred_callback(pred_msg):
    global right_pred
    global right_pred_received

    if not right_pred_received:
        right_pred_received = True
        right_pred = pred_msg.data
        right_pred = np.array(right_pred).reshape(-1, 6)

def pose_callback(pose):
    """
    We actually don't want the pose because it corresponds to the orbslam_link frame and we need
    the transformation from the map frame to the camera_link frame. Nonetheless we are extracting
    this transformation to keep the system synchronized.
    """
    global pose_received

    if not pose_received:
        try:
            tf_wc = tf_buffer.lookup_transform("map", "camera_link", rospy.Time())
            pose_received = True
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return

        X_c = tf_wc.transform.translation.x
        Y_c = tf_wc.transform.translation.y
        Z_c = tf_wc.transform.translation.z

        i = tf_wc.transform.rotation.x
        j = tf_wc.transform.rotation.y
        k = tf_wc.transform.rotation.z
        w = tf_wc.transform.rotation.w

        #X_c = pose.pose.position.x
        #Y_c = pose.pose.position.y
        #Z_c = pose.pose.position.z

        ## camera orientation in quaternion form
        #i = pose.pose.orientation.x
        #j = pose.pose.orientation.y
        #k = pose.pose.orientation.z
        #w = pose.pose.orientation.w

        global R_wc
        global T_wc
        R_wc = Rotation.from_quat([i, j, k, w]).as_matrix()
        T_wc = np.array([X_c, Y_c, Z_c])

def pipeline(bb_pairs):
    """
    The pipeline of semantic slam
    """

    ################################
    # bounding box matching functions
    ################################

    global landmarks
    global R_wc
    global T_wc
    depths, u_rects, v_rects, confs, labels = landmark_cal(bb_pairs)

    if len(landmarks) == 0:
        for i in range(len(depths)):
            # iterate over all bounding boxes
            Pw_new = backprojection(R_wc, T_wc, u_rects[i], v_rects[i], depths[i])  # the world coordinates of current bb centroid
            prior = likelihood_vector(confs[i], labels[i])
            lm = Landmark(Pw_new, prior)
            landmarks.append(lm)
    else:
        for i in range(len(depths)):
            # iterate over all bounding boxes
            Pw_new = backprojection(R_wc, T_wc, u_rects[i], v_rects[i], depths[i])  # the world coordinates of current bb centroid
            likelihood = likelihood_vector(confs[i], labels[i])

            dist_min = 10000
            dmin_lm_idx = None
            m = 0
            for lm in landmarks:
                same_lm, dist = lm.is_same_landmark(Pw_new, threshold=5)
                if same_lm:
                    if dist < dist_min:
                        dist_min = dist
                        dmin_lm_idx = m
                m += 1
            
            if dmin_lm_idx is None:
                # initialize a new landmarks
                new_lm = Landmark(Pw_new, likelihood)
                landmarks.append(new_lm)
            else:
                # update the current landmark
                landmarks[dmin_lm_idx].update_likelihood(likelihood)

def publish_pointcloud(landmark_list):
    points = []
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('rgb', 12, PointField.UINT32, 1)]
    header = Header()
    header.frame_id = "map"
    
    print("\n\n=======================\nNum of landmarks: ", len(landmark_list))

    for landmark in landmark_list:
        # Populate the points list with the landmark data
        x = landmark.Pw[0]
        y = landmark.Pw[1]
        z = landmark.Pw[2]
        rgb = label_rgb_values[landmark.label]
        points.append([x, y, z, rgb])

    landmark_pc = point_cloud2.create_cloud(header, fields, points)
    landmark_pc.header.stamp = rospy.Time.now()
    landmark_pc_pub.publish(landmark_pc)


if __name__ == "__main__":
    """ Initialize ROS """
    rospy.init_node('semantic_slam')

    # Get Parameters
    manual_sync = rospy.get_param("/semantic_slam_node/manual_sync")
    timeout_duration = rospy.get_param("/semantic_slam_node/timeout_duration")
    loop_rate = rospy.get_param("/semantic_slam_node/loop_rate")

    # Its ok if the system is slower but we don't want to excede 5 Hz because OrbSlam won't keep up
    rate = rospy.Rate(loop_rate)    

    # Subscriber topics
    rospy.Subscriber('/left/image_rect', Image, left_image_callback, queue_size=2, buff_size=52428800)
    rospy.Subscriber('/right/image_rect', Image, right_image_callback, queue_size=2, buff_size=52428800)

    rospy.Subscriber('/left_prediction', Float64MultiArray, left_pred_callback, queue_size=2, buff_size=52428800)
    rospy.Subscriber('/right_prediction', Float64MultiArray, right_pred_callback, queue_size=2, buff_size=52428800)

    rospy.Subscriber('/robot_pose', PoseStamped, pose_callback, queue_size=2)

    # Publisher topics
    sync_pub = rospy.Publisher("/kitti_player/synch", Bool, queue_size=10)
    landmark_pc_pub = rospy.Publisher("/semantic_slam/landmark_pc", PointCloud2, queue_size=2)

    # Transformations
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    """ Global variables and main workflow """
    # Global Variables
    global landmarks
    landmarks = []

    left_img = Image()
    right_img = Image()
    left_pred = Float64MultiArray()
    right_pred = Float64MultiArray()

    left_img_received = False
    right_img_received = False
    left_pred_received = False
    right_pred_received = False
    pose_received = False

    all_topics_received = False

    max_timeout = rospy.Duration.from_sec(timeout_duration)  # Wait a maximum of 0.2 sec (5hz) for all topic to be received
    timeout_occured = False

    time_at_last_sync = rospy.get_rostime()

    while not rospy.is_shutdown():
        # Check if a timeout has occured
        current_time = rospy.get_rostime()
        if (current_time - time_at_last_sync) > max_timeout:
            timeout_occured = True

        # Check if all topics have been received
        if left_img_received and right_img_received and left_pred_received and right_pred_received and pose_received:
            all_topics_received = True


        # Run a single pass of the pipeline
        if all_topics_received:
            """ Run PIPELINE HERE """
            bb_matching_pairs = stereo_inference(left_img, right_img, left_pred, right_pred)
            print("BB matching pairs: ", len(bb_matching_pairs))

            pipeline(bb_matching_pairs)

            publish_pointcloud(landmarks)


        # Publish the synchronizing topic
        if manual_sync:
            # Since we are manually publishing the synchronizing topic, then we don't care for
            # a timeout and thus should only care about whether or not all the topics have
            # been received.
            if all_topics_received:
                all_topics_received = False

                left_img_received = False
                right_img_received = False
                left_pred_received = False
                right_pred_received = False
                pose_received = False
        else:
            if all_topics_received or timeout_occured:
                time_at_last_sync = rospy.get_rostime()
                timeout_occured = False

                all_topics_received = False

                left_img_received = False
                right_img_received = False
                left_pred_received = False
                right_pred_received = False
                pose_received = False

                sync_pub.publish(True)

        rate.sleep()



