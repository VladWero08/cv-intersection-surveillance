import cv2
import os
import numpy as np

def preprocess_camera(camera: np.ndarray) -> np.ndarray:
    __camera = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)

    return __camera

DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_01_PATH = os.path.normpath(os.path.join(*[DIR, "..", "templates", "camera_01.jpg"]))
CAMERA_02_PATH = os.path.normpath(os.path.join(*[DIR, "..", "templates", "camera_02.jpg"]))
CAMERA_03_PATH = os.path.normpath(os.path.join(*[DIR, "..", "templates", "camera_03.jpg"]))
CAMERA_01 = preprocess_camera(cv2.imread(CAMERA_01_PATH))
CAMERA_02 = preprocess_camera(cv2.imread(CAMERA_02_PATH))
CAMERA_03 = preprocess_camera(cv2.imread(CAMERA_03_PATH))
CAMERAS = [CAMERA_01, CAMERA_02, CAMERA_03]

# references used for Homography projection
CAMERA_01_3D_REFERENCES = [(767, 944), (908, 995), (832, 922), (964, 967), (905, 901), (1015, 935), (942, 868), (1063, 906), (989, 852), (1098, 880), (1033, 839), (1128, 862), (1043, 827), (1093, 798), (902, 805), (959, 778), (847, 794), (910, 769), (526, 753), (467, 725), (423, 777), (376, 746), (369, 781), (329, 750), (320, 784), (291, 756), (279, 790), (254, 762), (235, 800), (213, 766)]
CAMERA_01_2D_REFERENCES = [(135, 251), (137, 238), (129, 250), (128, 237), (116, 250), (117, 237), (105, 250), (106, 237), (98, 251), (97, 238), (88, 251), (86, 235), (67, 252), (82, 251), (68, 272), (86, 272), (69, 282), (88, 281), (111, 333), (110, 352), (136, 335), (135, 352), (144, 335), (144, 352), (152, 333), (152, 351), (161, 333), (159, 351), (170, 334), (168, 351)]
CAMERA_02_3D_REFERENCES = [(1853, 600), (1849, 627), (1758, 600), (1748, 625), (1664, 598), (1648, 623), (1586, 592), (1545, 623), (1507, 598), (1472, 621), (1378, 593), (1337, 616), (1292, 594), 
(1246, 618), (1120, 590), (1055, 608), (770, 593), (889, 600), (691, 611), (803, 620), (645, 623), (769, 633), (1601, 947), (1661, 837), (1423, 921), (1504, 828), (1355, 807), (1289, 912), (1268, 806), (1196, 901)]
CAMERA_02_2D_REFERENCES = [(172, 253), (172, 235), (161, 253), (161, 237), (154, 251), (153, 235), (147, 254), (147, 235), (138, 251), (138, 235), (119, 252), (117, 235), (106, 251), (106, 235), (98, 251), (98, 237), (89, 250), (89, 235), (84, 251), (64, 252), (85, 263), (66, 263), (88, 283), (69, 281), (179, 351), (177, 334), (167, 351), (167, 333), (160, 349), (160, 333)]
CAMERA_03_3D_REFERENCES = [(1668, 697), (1568, 664), (1616, 716), (1513, 678), (1560, 736), (1456, 696), (1500, 756), (1390, 712), (1444, 773), (1343, 726), (1391, 788), (1291, 743), (1321, 808), (1223, 754), (1251, 841), (1153, 772), (1039, 882), (961, 813), (940, 898), (871, 828), (839, 918), (783, 843), (779, 919), (562, 948), (1812, 717), (1861, 681), (1851, 734), (1901, 693), (576, 999), (798, 976)]
CAMERA_03_2D_REFERENCES = [(194, 254), (192, 239), (183, 256), (184, 238), (175, 253), (175, 239), (163, 253), (162, 238), (155, 254), (154, 236), (147, 254), (147, 236), (137, 254), (137, 237), (128, 256), (128, 238), (108, 253), (107, 235), (96, 253), (96, 235), (88, 253), (88, 234), (83, 254), (65, 255), (215, 279), (200, 279), (215, 286), (198, 286), (88, 263), (68, 263)]

CAMERA_01_H = np.load(os.path.normpath(os.path.join(*[DIR, "..", "templates", "camera_01_homography.npy"]))) 
CAMERA_02_H = np.load(os.path.normpath(os.path.join(*[DIR, "..", "templates", "camera_02_homography.npy"]))) 
CAMERA_03_H = np.load(os.path.normpath(os.path.join(*[DIR, "..", "templates", "camera_03_homography.npy"]))) 


def get_similarity_sift(
    image_1: np.ndarray,
    image_2: np.ndarray
) -> float:
    """
    Computes the similarity between two images using SIFT features and
    Lowe's ratio test for good feature matches.

    Returns:
    --------
    similarity: float
        A similarity score between the two images, ranging from 0.0 to 1.0.
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image_1, None)
    kp2, des2 = sift.detectAndCompute(image_2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    if len(pts1) < 8:
        return 0

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    if mask is not None:
        return int(np.sum(mask))
    else:
        return 0

def get_camera_type_frame(frame: np.ndarray) -> int:
    """
    Given a frame from a video, compare its structural features to every
    camera type and choose the one that it matches the best.
    """
    best_similarity = -1
    best_camera_index = None

    __frame = preprocess_camera(frame)

    for i, camera in enumerate(CAMERAS):
        similarity = get_similarity_sift(__frame, camera)

        if similarity > best_similarity:
            best_similarity = similarity
            best_camera_index = i + 1 
            
    return best_camera_index


def get_fundamental_matrix(frame_A, frame_B):
    """
    Computes the Fundamental Matrix (F) between two images (frame_A and frame_B)
    using SIFT feature detection, FLANN matching, ratio test, and RANSAC for robustness.

    Args:
        frame_A (np.ndarray): The first image.
        frame_B (np.ndarray): The second image.

    Returns:
        tuple: A tuple containing the Fundamental Matrix (F),
               and the inlier corresponding points from frame_A and frame_B.
               Returns None if not enough matches are found or computation fails.
    """
    # initialize SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp_A, des_A = sift.detectAndCompute(frame_A, None)
    kp_B, des_B = sift.detectAndCompute(frame_B, None)

    # FLANN parameters for matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # perform K-Nearest Neighbor (KNN) matching
    matches = flann.knnMatch(des_A, des_B, k=2)

    # apply ratio test to filter good matches
    good_matches = []
    pts_A = []
    pts_B = []

    for m, n in matches:
        # ratio test as per Lowe's paper
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
            pts_A.append(kp_A[m.queryIdx].pt)
            pts_B.append(kp_B[m.trainIdx].pt)

    pts_A = np.float32(pts_A).reshape(-1, 1, 2)
    pts_B = np.float32(pts_B).reshape(-1, 1, 2)

    if len(pts_A) < 8: # Minimum 8 points for Fundamental Matrix
        print("Not enough good matches found to estimate Fundamental Matrix.")
        return None

    # estimate Fundamental Matrix using RANSAC
    F, mask = cv2.findFundamentalMat(pts_A, pts_B, cv2.RANSAC, 3, 0.99)

    if F is None:
        print("Error: Could not estimate Fundamental Matrix.")
        return None

    pts_A = pts_A[mask.ravel() == 1]
    pts_B = pts_B[mask.ravel() == 1]

    return F, pts_A, pts_B


def get_epipolar_line(F: np.ndarray, point_from_A: tuple) -> tuple:
    """
    Computes the epipolar line in Camera B corresponding to a given point in Camera A,
    using the Fundamental Matrix.

    Args:
        F (np.ndarray): The Fundamental Matrix relating Camera A and Camera B.
        point_from_A (tuple): The (x, y) coordinates of the point in Camera A.

    Returns:
        tuple: The coefficients (a, b, c) of the epipolar line (ax + by + c = 0) in Camera B.
               Returns None if computation fails.
    """ 
    # convert point_from_A to numpy array for epiline computation
    # it alsoo needs to be (1,1,2) for computeCorrespondEpilines
    point_from_A_np = np.float32([[point_from_A]]) 

    # compute epipolar line in Camera B corresponding to point_from_
    lines_B = cv2.computeCorrespondEpilines(point_from_A_np, 1, F)
    lines_B = lines_B.reshape(-1, 3)

    if len(lines_B) == 0:
        print("Could not compute epipolar line for the given point.")
        return None

    # return the coefficients of 
    # the epipolar line in Camera B
    return tuple(lines_B[0])


def get_homography_matrix(src_points_3D: np.ndarray, dst_points_2D: np.ndarray, save_path: str) -> np.ndarray:
    """
    Computes a homography matrix from 3D reference points to 2D image points
    and optionally saves it to a file.

    Args:
        src_points_3D (np.ndarray): Coordinates of points in the 3D reference system.
        dst_points_2D (np.ndarray): Corresponding 2D pixel coordinates in the image.
        save_path (str): The file path to save the computed homography matrix (.npy format).

    Returns:
        np.ndarray: The computed 3x3 homography matrix. Returns None if computation fails.
    """
    src_pts = np.float32(src_points_3D).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_points_2D).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is not None:
        np.save(save_path, H)
        print(f"Homogenity saved to {save_path}!")
    else:
        print(f"Failed to compute homography for {save_path}!")

    return H


def project_from_camera_A_to_camera_B(bbox: tuple, camera_A_H: np.ndarray, camera_B_H: np.ndarray) -> tuple:
    """
    Projects a bounding box from Camera A's coordinate system to Camera B's coordinate system
    using pre-computed homography matrices.

    Args:
        bbox (tuple): The bounding box in Camera A as (x, y, width, height).
        camera_A_H (np.ndarray): Homography matrix from Camera A's 2D image to the 3D reference.
        camera_B_H (np.ndarray): Homography matrix from Camera B's 2D image to the 3D reference.

    Returns:
        tuple: The projected bounding box in Camera B as (x_min_B, y_min_B, x_max_B, y_max_B).
    """
    x_min_A = bbox[0]
    x_max_A = bbox[0] + bbox[2]
    y_min_A = bbox[1] 
    y_max_A = bbox[1] + bbox[3]

    coordinates_A = np.float32([
        [x_min_A, y_max_A],
        [x_max_A, y_max_A],
        [x_max_A, y_min_A],
        [x_min_A, y_min_A]
    ]).reshape(-1, 1, 2)

    # project bbox from 3D to reference system
    coordinates_2D = cv2.perspectiveTransform(coordinates_A, camera_A_H)
    # inverse homography for camera B
    H_ref_to_camera_B = np.linalg.inv(camera_B_H)
    # project to camera's B coordinate system
    coordinates_B = cv2.perspectiveTransform(coordinates_2D, H_ref_to_camera_B)
    coordinates_B = coordinates_B.reshape(-1, 2).astype(int)

    x_min_B = np.min(coordinates_B[:, 0])
    y_min_B = np.min(coordinates_B[:, 1])
    x_max_B = np.max(coordinates_B[:, 0])
    y_max_B = np.max(coordinates_B[:, 1])
    
    return (x_min_B, y_min_B, x_max_B, y_max_B)



def draw_epipolar_line_on_frame(frame_B, line_coeffs, color=(0, 255, 0), thickness=2):
    """
    Draws the epipolar line on frame_B using the line coefficients (a, b, c).

    Args:
        frame_B (ndarray): Image from Camera B.
        line_coeffs (tuple): (a, b, c) for the epipolar line ax + by + c = 0.
        color (tuple): Line color in BGR.
        thickness (int): Line thickness.
    
    Returns:
        ndarray: Image with the epipolar line drawn.
    """
    a, b, c = line_coeffs
    h, w = frame_B.shape[:2]

    # Compute intersection with image borders
    if b != 0:
        # y = (-a*x - c) / b
        pt1 = (0, int(-c / b))                    # x = 0
        pt2 = (w - 1, int((-a * (w - 1) - c) / b)) # x = w-1
    else:
        # Vertical line: x = -c / a
        x = int(-c / a)
        pt1 = (x, 0)
        pt2 = (x, h - 1)

    # Draw the line
    frame_with_line = frame_B.copy()
    cv2.line(frame_with_line, pt1, pt2, color, thickness)

    return frame_with_line
