import numpy as np
from scipy.spatial.transform import Rotation

baseline = 0.54  # camera baseline [m]
cx = 607.1928  # imaimage center in x direction
cy = 185.2157  # imaimage center in y direction
W = 1241  # image width
H = 376  # image height
fx = 718.856  # focal length of x direction
fy = 718.856  # focal length of y direction
cam0_to_cam2 = 0.06  # the distance between cam0(gray) and cam2(color)

Labels_idx = {0:0, 1:1, 2:2, 3:3, 4:5, 5:7}  # labels to predict as value, keys are index (for likelihood vector)


class Landmark():
    """
    The class of Landmark
    """

    def __init__(self, world_position, prior):
        self.Pw = world_position
        self.likelihood = prior  # likelihood array of all labels
        self.label = int(np.argmax(self.likelihood))  # the most likely label index

    def is_same_landmark(self, Pw_new, threshold=2):
        """
        Check whether to initialize a new landmark
        Input:
            Pw_new: the world coordinates of the backprojected landmark in a new frame, array of shape (3,)
            threshold: default to be 2 meters (Euclidean distance)
        Ouput:
            boolean: whether to initialize a new landmark

        """
        dist = np.linalg.norm(self.Pw - Pw_new)
        if dist <= threshold:
            return True, dist
        else:
            return False, dist
    
    def update_likelihood(self, likelihood):
        """
        Beyersian update of the label likelihood vector
        """
        likelihood = likelihood * self.likelihood
        normalizer = np.sum(likelihood)
        self.likelihood = likelihood / normalizer


def landmark_cal(bb_pairs):
    """
    Compute the landmark (bounding box centroid) depth with respect to left camera, centroid, confidence and labels
    Input:
        bb_pairs: a list of bounding box pairs, each pair is a (2, 6) array
    Output:
        depth: a list of depth of the centroid of all the landmarks in left camera frame
        u_rects: a list of all the centroid coordinates u of bounding boxes in left frame
        v_rects: a list of all the centroid coordinates v of bounding boxes in left frame
        confs: a list of confidences of bounding boxes from YOLO
        labels: a list of labels of all bounding boxes
    """
    depth = []
    u_rects = []
    v_rects = []
    confs = []
    labels = []
    for bb_pair in bb_pairs:
        bb_l = bb_pair[0, :]
        bb_r = bb_pair[1, :]
        u_l = (bb_l[0] + bb_l[2]) / 2  # only take the disparity on x direction
        u_r = (bb_r[0] + bb_r[2]) / 2
        v_l = (bb_l[1] + bb_l[3]) / 2
        z = fx * (baseline / (u_l - u_r))
        if z < 0 or z > 50:
            continue
        conf = (bb_l[-2] + bb_r[-2]) / 2
        depth.append(z)
        u_rects.append(u_l)
        v_rects.append(v_l) 
        confs.append(conf)
        labels.append(int(bb_l[-1]))
    return depth, u_rects, v_rects, confs, labels


def backprojection(R, T, u_rect, v_rect, depth, colorcam=True):
    """
    Backproject the pixel in current frame (left camera) to the world frame
    Input:
        R: the rotation matrix of current left camera pose
        T: the translation vector of current left camera pose
        u_rect: the rectified pixel coordinates
        v_rect: the rectified pixel coordinates
        depth: the depth wrt the current left camera frame
    Output:
        Pw = the backprojected coordinates in world frame, (3,) array
    """
    K_corr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    uv = np.array([u_rect, v_rect, 1])

    Pc = depth * (np.linalg.inv(K_corr) @ uv)
    Pc_homo = np.append(Pc, [1])

    # transformation from camera to world frame
    R_cw = R.T
    T_cw = -R.T @ T 
    Trans_cw = np.append(R_cw, T_cw[:, np.newaxis], axis=1)
    Trans_cw = np.append(Trans_cw, np.array([[0, 0, 0, 1]]), axis=0)
    #if colorcam:
        #Trans_c0c2 = np.append(np.identity(3), np.array([-cam0_to_cam2, 0, 0])[:, np.newaxis])
        #Trans_c0c2 = np.append(Trans_c0c2, np.array([[0, 0, 0, 1]]))
        

    Pw = np.linalg.inv(Trans_cw) @ Pc_homo

    return Pw[:-1]


def likelihood_vector(conf, label):
    """
    Build the likelihood vector
    Input:
        conf: the confidence of the likelihood
    Output:
        likelihood: the likelihood vector
    """
    likelihood = np.ones(len(Labels_idx))
    likelihood *= (1 - conf) / (len(Labels_idx) - 1)
    likelihood[list(Labels_idx.keys())[list(Labels_idx.values()).index(label)]] = conf
    
    return likelihood
