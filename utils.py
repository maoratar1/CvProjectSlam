import numpy as np
import cv2
import tqdm

# Todo: 1. Change some functions name. 2. write ICP 3. Update Data Path

MOVIE_LEN = 3450

DATA_PATH = r'/Users/maoratar/opt/anaconda3/envs/Van_Ex1/VAN_ex/dataset/sequences/00/'


def read_images(frame_num, kernel_size=0):
    """
    :param frame_num:
    :param kernel_size:
    :param idx: Image's index in the Kitti dataset
    :return: left and right cameras photos
    """
    img_name = '{:06d}.png'.format(000000 + frame_num)
    img1 = cv2.imread(DATA_PATH + 'image_0/' + img_name, 1)
    img2 = cv2.imread(DATA_PATH + 'image_1/' + img_name, 1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if kernel_size != 0:
        # img1 = cv2.blur(img1, (kernel_size, kernel_size))
        # img2 = cv2.blur(img2, (kernel_size, kernel_size))
        img1 = cv2.GaussianBlur(img1, (kernel_size, kernel_size), 0)
        img2 = cv2.GaussianBlur(img2, (kernel_size, kernel_size), 0)

    return img1, img2


def read_cameras():
    """
    Reads First frame cameras intrinsic and extrinsic matrices
    :return:
    """
    print("\t\tLoading cameras matrices:\n\t\tK: Calibration camera\n\t\tM1: left camera extrinsic matrix\n"
          "\t\tM2: Right "
          "camera extrinsic matrix relative to the left camera")

    with open(DATA_PATH + 'calib.txt') as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
        l1 = [float(i) for i in l1]
        m1 = np.array(l1).reshape(3, 4)
        l2 = [float(i) for i in l2]
        m2 = np.array(l2).reshape(3, 4)
        k = m1[:, :3]
        m1 = np.linalg.inv(k) @ m1
        m2 = np.linalg.inv(k) @ m2
        return k, m1, m2


K, M1, M2 = read_cameras()


############## Images Matching ############################

def feature_detection_and_description(img1, img2, alg):
    """
    Computes KITTI key d2_points and theirs descriptors
    :param alg: Feature detecting and description algorithm
    :return: KITTI key d2_points and descriptors
    """
    img1_kpts, img1_dsc = alg.detectAndCompute(img1, None)
    img2_kpts, img2_dsc = alg.detectAndCompute(img2, None)
    return np.array(img1_kpts), np.array(img1_dsc), np.array(img2_kpts), np.array(img2_dsc)


def bf_matching(img1_dsc, img2_dsc, crossCheck=True, sort=True):
    """
    Find Matches between two KITTI descriptors
    :param metric: distance function for computes distance between two descriptors
    :param img1_dsc: image 1 descriptors
    :param img2_dsc: image 2 descriptors
    :return: array of matches
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    matches = bf.match(img1_dsc, img2_dsc)
    if sort:
        # Sort the matches from the best match to the worst - where best means it has the lowest distance
        matches = sorted(matches, key=lambda x: x.distance)
    return np.array(matches)


def detect_and_match(img1, img2):
    # Detects the image key d2_points and compute their descriptors
    img1_kpts, img1_dsc, img2_kpts, img2_dsc = feature_detection_and_description(img1, img2, cv2.SIFT_create())

    # Matches between the KITTI and plots the matching
    matches = bf_matching(img1_dsc, img2_dsc)
    return img1_kpts, img1_dsc, img2_kpts, img2_dsc, matches


def axes_of_matches(match, img1_kpts, img2_kpts):
    """
    Returns (x,y) values for each point from the two in the match object
    """
    img1_idx = match.queryIdx
    img2_idx = match.trainIdx
    x1, y1 = img1_kpts[img1_idx].pt
    x2, y2 = img2_kpts[img2_idx].pt
    return x1, y1, x2, y2


def rectified_stereo_pattern_rej(img1_kpts, img2_kpts, matches):
    """
    Apply the rectified stereo pattern rejection on image1 key d2_points and image 2 key d2_points
    :return: List of Inliers and outliers indexes of the 2 KITTI
    """
    inliers_matches_idx, outliers_matches_idx = [], []

    num_matches = len(matches)
    for i in range(num_matches):
        _, y1, _, y2 = axes_of_matches(matches[i], img1_kpts, img2_kpts)
        if abs(y2 - y1) <= 2:
            inliers_matches_idx.append(i)
        else:
            outliers_matches_idx.append(i)

    return inliers_matches_idx, outliers_matches_idx


def read_and_rec_match(frame_num):
    """
    Reads KITTI from pair idx, Finds matches with rectified test
    :param idx: Frame's index
    :return: key d2_points of the two KITTI and the matches
    """
    # Find matches in frame with rectified test
    left_img, right_img = read_images(frame_num)
    left0_kpts, left0_dsc, right0_kpts, _, pair0_matches = detect_and_match(left_img, right_img)
    pair0_rec_matches_idx, _ = rectified_stereo_pattern_rej(left0_kpts, right0_kpts, pair0_matches)
    return left0_kpts, left0_dsc, right0_kpts, pair0_matches, pair0_rec_matches_idx


def create_rec_dic(pair_matches, pair_rec_matches_idx):
    """
    Create dictionary in the form that for each key point of the left image:
                    {index in the list of key d2_points: index in the matches list}
    :param pair_matches: matches between left(query) and right(train) KITTI
    :param pair_rec_matches_idx: list of matches indexes (from pair matches list) that passed the rectified test
    :return: Dictionary
    """
    rec1_dic = {}
    for idx in pair_rec_matches_idx:
        kpt_id = pair_matches[idx].queryIdx
        rec1_dic[kpt_id] = idx
    return rec1_dic


def find_kpts_in_all_4_rec(left0_left1_matches, rec0_dic, rec1_dic):
    """
    Finds matching key d2_points in left0 and left1 that passed the rectified test.
    :param left0_left1_matches: matches between left0 and left1 KITTI
    :param rec0_dic: Dictionary which contains the key d2_points indexes that passed the rectified test in pair0
    :param rec1_dic: same as above for pair1
    :return: 3 lists:
                    1) q_pair0: list of indexes from pair0 matches list that passed the rectified test in pair 0
                                and are matched in left0 and left1
                    2) q_pair1: same as above for pair1
                    3) q_left0_left1: same as above for left0_left1 matches list
    """
    q_pair0, q_pair1, q_left0_left1 = [], [], []

    for i, match in enumerate(left0_left1_matches):
        left0_kpt_idx = match.queryIdx
        left1_kpt_idx = match.trainIdx

        # Explanation:
        # Check if "left0_kpt_idx" and "left1_kpt_idx" are key d2_points that passed the rectified test in each pair.
        # If so:  It adds to q_pair0 the key d2_points index in the list of pair 0 matches
        # so in index i, q_pair0[i] contains the index of the match in
        # pair 0 matches that match to pair 1 matches by the connector from
        # the matches between left0 and left1
        # for example if i'th match is <left0_kpt, left1_kpt> and left0_kpt and left1_kpt passed the rectified test
        # we get that pair0idx[i] = <left0_kpt, right0_kpt> (actually the pair0 index of this pair)
        #             pair1idx[i] = <left1_kpt, right1_kpt> (same as above)
        #             q_left0_left1[i] = <left0_kpt, left1_kpt> (same as above)
        if left0_kpt_idx in rec0_dic and left1_kpt_idx in rec1_dic:
            q_pair0.append(rec0_dic[left0_kpt_idx])
            q_pair1.append(rec1_dic[left1_kpt_idx])
            q_left0_left1.append(i)

    return q_pair0, q_pair1, q_left0_left1


def get_matches_coor(matches, img1_kpts, img2_kpts):
    """
    Returns 2 numpy arrays of matches d2_points in img1 and img2 accordingly
    """
    img1_matches_coor, img2_matches_coor = [], []
    for match in matches:
        img1_idx, img2_idx = match.queryIdx, match.trainIdx
        img1_matches_coor.append(img1_kpts[img1_idx].pt)
        img2_matches_coor.append(img2_kpts[img2_idx].pt)

    return np.array(img1_matches_coor), np.array(img2_matches_coor)


def linear_least_square(l_cam_mat, r_cam_mat, kp1_xy, kp2_xy):
    """
    Linear least square procedure.
    :param l_cam_mat: Left camera matrix
    :param r_cam_mat: Right camera matrix
    :param kp1_xy: (x,y) for key point 1
    :param kp2_xy: (x,y) for key point 2
    :return: Solution for the equation Ax = 0
    """
    # Compute the matrix A
    mat = np.array([kp1_xy[0] * l_cam_mat[2] - l_cam_mat[0],
                    kp1_xy[1] * l_cam_mat[2] - l_cam_mat[1],
                    kp2_xy[0] * r_cam_mat[2] - r_cam_mat[0],
                    kp2_xy[1] * r_cam_mat[2] - r_cam_mat[1]])

    # Calculate A's SVD
    u, s, vh = np.linalg.svd(mat, compute_uv=True)

    # Last column of V is the result as a numpy object
    return vh[-1]


def triangulate(l_mat, r_mat, kp1_xy_lst, kp2_xy_lst):
    """
    Apply triangulation procedure
    :param l_mat: Left camera matrix
    :param r_mat: Right camera matrix
    :return: List of 3d d2_points in the world
    """
    kp_num = len(kp1_xy_lst)
    res = []
    for i in range(kp_num):
        p4d = linear_least_square(l_mat, r_mat, kp1_xy_lst[i], kp2_xy_lst[i])
        p3d = p4d[:3] / p4d[3]
        res.append(p3d)
    return np.array(res)


def triangulation_outlier_rej(point_cloud0, point_cloud1):
    not_far_away_pts_pc0 = point_cloud0 <= 300
    non_negative_pts_pc0 = point_cloud0 > 0
    good_pts_pc0 = not_far_away_pts_pc0 and non_negative_pts_pc0

    not_far_away_pts_pc1 = point_cloud1 <= 300
    non_negative_pts_pc1 = point_cloud1 > 0
    good_pts_pc1 = not_far_away_pts_pc1 and non_negative_pts_pc1

    return good_pts_pc0 and good_pts_pc1


def icp_with_initial_estimate(point_cloud0, point_cloud1):  # Fixme: Implement
    """
    The point_clouds list are match each other by index.
    :param point_cloud0:
    :param point_cloud1:
    :return:
    """
    return 0


def icp(point_cloud0, point_cloud1):  # Fixme: implement
    return 0


def compute_trans_between_cur_to_next_4images_icp(left0_kpts, left0_dsc, right0_kpts,
                                                  pair0_matches, pair0_rec_matches_idx,
                                                  left1_kpts, left1_dsc, right1_kpts,
                                                  pair1_matches, pair1_rec_matches_idx):
    """
   Compute the transformation T between left 0 and left1 KITTI
   :return: numpy array with shape 3 X 4
   """

    # Find matches between left0 and left1
    left0_left1_matches = bf_matching(left0_dsc, left1_dsc)

    # Maybe it isn't needed, and we'll do the icp without search for matching in 4 pts but rather a simple ICP

    # Find key pts that match in all 4 KITTI
    rec0_dic = create_rec_dic(pair0_matches, pair0_rec_matches_idx)  # dict of {left kpt idx: pair rec id}
    rec1_dic = create_rec_dic(pair1_matches, pair1_rec_matches_idx)
    q_pair0_idx, q_pair1_idx, q_left0_left1_idx = find_kpts_in_all_4_rec(left0_left1_matches, rec0_dic, rec1_dic)

    # Frame triangulation
    left0_matches_coor, right0_matches_coor = get_matches_coor(pair0_matches[q_pair0_idx], left0_kpts,
                                                               right0_kpts)
    left1_matches_coor, right1_matches_coor = get_matches_coor(pair1_matches[q_pair1_idx], left1_kpts,
                                                               right1_kpts)

    pair0_p3d_pts = triangulate(K @ M1, K @ M2, left0_matches_coor, right0_matches_coor)
    pair1_p3d_pts = triangulate(K @ M1, K @ M2, left1_matches_coor, right1_matches_coor)

    common_pt_clouds_good_idx = triangulation_outlier_rej(pair0_p3d_pts, pair1_p3d_pts)

    point_cloud0 = pair0_p3d_pts[common_pt_clouds_good_idx]
    point_cloud1 = pair1_p3d_pts[common_pt_clouds_good_idx]

    # Find the best transformation between left0 and left1
    left1_cam_mat = icp(point_cloud0, point_cloud1)

    return left1_cam_mat


def compute_trans_between_cur_to_next_regular_icp(left0_kpts, left0_dsc, right0_kpts,
                                      pair0_matches, pair0_rec_matches_idx,
                                      left1_kpts, left1_dsc, right1_kpts,
                                      pair1_matches, pair1_rec_matches_idx):
    """
    Compute the transformation T between left 0 and left1 KITT
    :return: numpy array with shape 3 X 4
    """

    # Find matches between left0 and left1
    left0_left1_matches = bf_matching(left0_dsc, left1_dsc)

    # Frame triangulation
    left0_matches_coor, right0_matches_coor = get_matches_coor(pair0_matches, left0_kpts, right0_kpts)
    left1_matches_coor, right1_matches_coor = get_matches_coor(pair1_matches, left1_kpts, right1_kpts)

    pair0_p3d_pts = triangulate(K @ M1, K @ M2, left0_matches_coor, right0_matches_coor)
    pair1_p3d_pts = triangulate(K @ M1, K @ M2, left1_matches_coor, right1_matches_coor)

    common_pt_clouds_good_idx = triangulation_outlier_rej(pair0_p3d_pts, pair1_p3d_pts)

    point_cloud0 = pair0_p3d_pts[common_pt_clouds_good_idx]
    point_cloud1 = pair1_p3d_pts[common_pt_clouds_good_idx]

    # Find the best transformation between left0 and left1
    left1_cam_mat = icp_with_initial_estimate(point_cloud0, point_cloud1)

    return left1_cam_mat


def compose_transformations(first_ex_mat, second_ex_mat):
    """
    Compute the composition of two extrinsic camera matrices.
    first_cam_mat : A -> B
    second_cam_mat : B -> C
    composed mat : A -> C
    """
    # [R2 | t2] @ [ R1 | t1] = [R2 @ R1 | R2 @ t1 + t2]
    #             [000 | 1 ]
    hom1 = np.append(first_ex_mat, [np.array([0, 0, 0, 1])], axis=0)
    return second_ex_mat @ hom1


def convert_trans_from_rel_to_global(T_arr):
    relative_T_arr = []
    last = T_arr[0]

    for t in T_arr:
        last = compose_transformations(last, t)
        relative_T_arr.append(last)

    return relative_T_arr


def localization(first_left_ex_mat):
    """
    Compute the transformation of two consequence left KITTI in the whole movie
    :return:array of transformations where the i'th element is the transformation between i-1 -> i
    """
    T_arr = [first_left_ex_mat]

    # Find matches in pair0 with rectified test
    left0_kpts, left0_dsc, right0_kpts, pair0_matches, pair0_rec_matches_idx = read_and_rec_match(000000)
    for i in tqdm.tqdm(range(1, MOVIE_LEN)):
        left1_kpts, left1_dsc, right1_kpts, pair1_matches, pair1_rec_matches_idx = \
            read_and_rec_match(000000 + i)

        left1_ex_mat = compute_trans_between_cur_to_next_4images_icp(left0_kpts, left0_dsc, right0_kpts,
                                                         pair0_matches, pair0_rec_matches_idx,
                                                         left1_kpts, left1_dsc, right1_kpts,
                                                         pair1_matches, pair1_rec_matches_idx)
        left0_kpts, left0_dsc, right0_kpts, \
            pair0_matches, pair0_rec_matches_idx = left1_kpts, left1_dsc, \
                                                   right1_kpts, pair1_matches, pair1_rec_matches_idx

        T_arr.append(left1_ex_mat)

    return convert_trans_from_rel_to_global(T_arr)
