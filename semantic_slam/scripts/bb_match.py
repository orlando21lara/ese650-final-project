""" Bounding Box Matching Functions """
import numpy as np
import cv2
from scipy.spatial.distance import cdist
import random
import copy

def match_features(f1,f2):
    """
    f1, f2: N * feature_size, two features to be matched
    """

    # compute pairwise distance between f1 and f2
    dist = cdist(f1,f2)
    threshold = 0.7

    # forward matching, find closet two matches in f2 to f1 and do a ratio test
    f_sorted_ind = np.argsort(dist, axis=1)
    f_sorted_dist = np.take_along_axis(dist, f_sorted_ind[:,0:2], axis=1)

    ratio_f = f_sorted_dist[:,0] / f_sorted_dist[:,1]
    keep_f = np.argwhere(ratio_f < threshold)
    keep_f = keep_f.flatten()

    match_f = np.take(f_sorted_ind[:,0], keep_f, axis=0)
    match_fwd = np.stack((keep_f, match_f),axis=-1)

    # backward matching, find closet two matches in f1 to f2 and do a ratio test
    b_sorted_ind = np.argsort(dist, axis=0)
    b_sorted_dist = np.take_along_axis(dist, b_sorted_ind[0:2,:], axis=0)
    b_sorted_dist = np.transpose(b_sorted_dist)

    ratio_b = b_sorted_dist[:,0] / b_sorted_dist[:,1]
    keep_b = np.argwhere(ratio_b < threshold)
    keep_b = keep_b.flatten()

    match_b = np.take(b_sorted_ind[0,:], keep_b, axis=0)
    match_bkwd = np.stack((match_b, keep_b),axis=-1)

    # get the intersect of two matching as the final result, python set 
    intersect = set(map(tuple, match_fwd)) & set(map(tuple, match_bkwd))
    match = np.array(list(intersect))

    return match, match_fwd, match_bkwd


def lr_pair(I1, I2):
    # create grayscale images
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # convert images to RGB format for display
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)

    # compute SIFT features
    sift = cv2.SIFT_create() #cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(I1_gray, None)
    kp2, des2 = sift.detectAndCompute(I2_gray, None)

    # match features
    match, match_fwd, match_bkwd = match_features(des1, des2)

    # get corresponding points p1, p2
    p1 = np.array([kp.pt for kp in kp1])[match[:, 0]]
    p2 = np.array([kp.pt for kp in kp2])[match[:, 1]]

    return p1, p2


def insideBB(p, rect_dim):
    #rect_dim : [xmin        ymin        xmax        ymax]
    xmin, ymin, xmax, ymax = rect_dim
    isinx = np.logical_and( p[:,0] >= xmin,  p[:,0] <= xmax)
    isiny =  np.logical_and( p[:,1] >= ymin,  p[:,1] <= ymax)
    # return the boolean arr that indicates if the i-th fp is inside the current bb

    return np.logical_and(isinx, isiny)


def computeAR(l_ind, r_ind, l_bb_dim, r_bb_dim):
    #rect_dim : [xmin        ymin        xmax        ymax]
    ar_l = (l_bb_dim[l_ind,2] - l_bb_dim[l_ind,0]) / (l_bb_dim[l_ind,3] - l_bb_dim[l_ind,1])
    ar_r = (r_bb_dim[r_ind,2] - r_bb_dim[r_ind,0]) / (r_bb_dim[r_ind,3] - r_bb_dim[r_ind,1])

    return ar_l, ar_r


# l2r match : (for the left points in the bb-i, ask their matched right points to vote for the bb-j)
def bb_matching(res_pd_l, res_pd_r, p1, p2):
    n_l, n_r = res_pd_l.shape[0], res_pd_r.shape[0]
    if n_l == 0 or n_r == 0:
        return {}

    l_bb_dim = res_pd_l[:,:4]
    r_bb_dim = res_pd_r[:,:4]

    in_arr_l = []
    in_arr_r = []

    for j in range(n_r):
        in_r = insideBB(p2, r_bb_dim[j])
        in_arr_r.append(in_r)
    in_arr_r = np.array(in_arr_r).T

    #
    dic_l2r = {}

    # match from left to right
    for i in range(n_l):

        in_l = insideBB(p1, l_bb_dim[i])
        vote_res = np.sum(in_arr_r[in_l], axis=0)
        rbb_ind = np.argwhere(vote_res == np.amax(vote_res)).flatten().tolist()
        max_vote = vote_res[rbb_ind[0]]
        total_vote = vote_res.sum()

        # if none of the matching feats in the right voted for any bb; then there's no matching bb
        # discard directly; dont include in the dic_l2r
        if not np.any(vote_res):
            pass
        #
        else:
            ratio = max_vote / total_vote
            # if a tie in vote,
            if(len(rbb_ind)>1):
                ratio = -1 #1/len(rbb_ind)

            dic_l2r[i] = (ratio, rbb_ind, max_vote)


    ### Post processing and assignment:
    res_pairs = {}
    taken_r = []
    # sequence of adding : no-ties;
    dic_r2l = {}

    ############################ priority 1: no tie
    for key in dic_l2r:
        ratio = dic_l2r[key][0]
        rbb_ind = dic_l2r[key][1]
        min_ind = rbb_ind[0]

        #
        if ratio > 0:
            if min_ind not in dic_r2l:
               # key = l_ind
               dic_r2l[min_ind]=[key]
            else:
               dic_r2l[min_ind].append(key)

    # assignment of no-ties:
    for r_ind in dic_r2l:

        # if the r_ind has multiple left matches
        if len(dic_r2l[r_ind])>1:
            dt_lst = []
            # collect the ar, vote info
            for l_ind in dic_r2l[r_ind]:
                ## (ar_diff, #vote, l_ind)
                ar_l, ar_r = computeAR(l_ind, r_ind, l_bb_dim, r_bb_dim)
                ar_diff = -abs(ar_l - ar_r)
                max_vote = dic_l2r[l_ind][2]
                dt_lst.append((ar_diff, max_vote, l_ind))

            # sort tie-breaker (ar, #vote) in descending; ar priority over #vote
            dt_lst.sort(key=lambda x: (x[0], x[1]), reverse=True)
            l_win = dt_lst[0][2]

            ###
            # add the final decision to res_pairs{l_ind:r_ind}
            res_pairs[l_win] = r_ind
            taken_r.append(r_ind)

        # unique match:
        else:
            l_ind = dic_r2l[r_ind][0]
            res_pairs[l_ind] = r_ind
            taken_r.append(r_ind)

    ############################ priority 1: no tie


    ############################ priority 2: tie
    dic_r2l_tie = {}
    dt_lst_tie = []

    for key in dic_l2r:

        # the non-ties discarded were not deleted from dic_l2r
        if (dic_l2r[key][0] == -1):
            # compute ar, similar step as in no-tie case

            # dic_l2r[i] = (ratio, rbb_ind, max_vote)
            rbb_ind = dic_l2r[key][1]

            ar_l = (l_bb_dim[key,2] - l_bb_dim[key,0]) / (l_bb_dim[key,3] - l_bb_dim[key,1])
            ar_r = (r_bb_dim[rbb_ind,2] - r_bb_dim[rbb_ind,0]) / (r_bb_dim[rbb_ind,3] - r_bb_dim[rbb_ind,1])

            min_ind = dic_l2r[key][1][np.argmin(abs(ar_r - ar_l))]
            min_ar_diff = np.min(abs(ar_r - ar_l))

            # directly skip if the tie-break gives one assigned to no-ties
            if min_ind not in taken_r:
                pass
            elif min_ind not in dic_r2l_tie:
               dic_r2l_tie[min_ind]=[[min_ar_diff, key]]
            else:
               dic_r2l_tie[min_ind].append([min_ar_diff, key])


    # dic_r2l_tie: {r_ind : [[min_ar_diff1, key1], [min_ar_diff2, key2]]}
    for r_ind in dic_r2l_tie:

        # duplicate assignment
        if len(dic_r2l_tie[r_ind])>1:
            #[[min_ar_diff1, key1], [min_ar_diff2, key2]]
            dt_lst = copy.deepcopy(dic_r2l_tie[r_ind])
            dt_lst.sort()
            l_win = dt_lst[0][1]
            ###
            # add the final decision to res_pairs{l_ind:r_ind}
            res_pairs[l_win] = r_ind
            taken_r.append(r_ind)

        # unique tie-breaker
        else:
            l_ind = dic_r2l_tie[r_ind][0][1]
            res_pairs[l_ind] = r_ind
            taken_r.append(r_ind)

    ############################ priority 2: tie

    # print(taken_r)
    # print(dic_l2r)
    return res_pairs


#def stereo_inference(stop_frame, start_frame=0, sequence="00"):
def stereo_inference(im0, im1, res_pd_l, res_pd_r):
    # low confidence idtf/bb filtered out
    conf_thres = 0.6
    res_conf_l, res_conf_r = res_pd_l[res_pd_l[:,4] > conf_thres], res_pd_r[res_pd_r[:,4] > conf_thres]
    #res_conf_l, res_conf_r = res_pd_l, res_pd_r

    # matching results
    p1, p2 = lr_pair(im0, im1)
    res_pairs = bb_matching(res_conf_l, res_conf_r, p1, p2)

    # remove duplicates by sequential order
    temp = []
    nodup = {}
    for key, val in res_pairs.items():
        if val not in temp:
            temp.append(val)
            nodup[key] = val

    # remove the pair of mismatched class
    res_data = []   #a list of 2x6 arrays (len list = num of matching bb pairs)

    for key in nodup:
        l_vec = res_conf_l[key,:]
        r_vec = res_conf_r[nodup[key],:]
        if (l_vec[-1] != r_vec[-1]):
            pass
        else:
            res_data.append(np.array([l_vec, r_vec]))

    ################

    # a list of 2x6 arrays (len list = num of matching bb pairs)
    # print("frame -",i)
    # print(nodup)

    # print("-------------------------------------")

    return res_data




