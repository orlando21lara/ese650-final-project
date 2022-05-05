#! /usr/bin/env python3

from semantic_process import *
import roslib
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')

Labels = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # all the possible labels, keys are the class number in COCO



def pipeline():
    """
    The pipeline of semantic slam
    """

    ################################
    # bounding box matching functions
    ################################
    bb_pairs = []
    ################################

    global landmarks
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
                same_lm, dist = lm.is_same_landmark(Pw_new, threshold=2)
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
    



def pose_callback(pose):
    X_c = pose.pose.position.x
    Y_c = pose.pose.position.y
    Z_c = pose.pose.position.z

    # camera orientation in quaternion form
    i = pose.pose.orientation.x
    j = pose.pose.orientation.y
    k = pose.pose.orientation.z
    w = pose.pose.orientation.w

    global R_wc
    global T_wc
    R_wc = Rotation.from_quat([i, j, k, w]).as_matrix()
    T_wc = np.array([X_c, Y_c, Z_c])
    

if __name__ == '__main__':
    global landmarks
    landmarks = []
    
    
    rospy.init_node('')
    pose_topic = '/orb_slam2_stereo/pose'
    rospy.Subscriber(pose_topic, PoseStamped, pose_callback, queue_size=1)

    ################################
    #yolo Subscriber
    ################################

    rospy.spin()

