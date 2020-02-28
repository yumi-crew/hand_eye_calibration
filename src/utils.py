import numpy as np
import zivid
import cv2
from homogeneousTransformations import HTransf as HT
import glob


def load_zdf(file: str):
    with zivid.Application() as app:
        image = zivid.Frame(file)
        pointCloud = image.get_point_cloud().to_array()
    return pointCloud


def load_t4s(path: str):
    files = sorted(glob.glob(f'{path}/*.t4'))
    poses = [HT.from_matrix(np.fromfile(pose_file, dtype=float, count=-1, sep=' ').reshape(4, 4))
             for pose_file in files]
    return files, poses


def load_quat_poses(path: str):
    files = sorted(glob.glob(f'{path}/*.txt'))
    poses = [HT.from_quat_pose(np.fromfile(
        pose_file, dtype=float, count=-1, sep=' ')) for pose_file in files]
    return files, poses


def calibration_pts(nx: int, ny: int, square_size: float, pnt_clds: np.ndarray):
    rgbs = []
    corners = []
    XYZs = []
    obj_pts = np.zeros((len(pnt_clds), nx*ny, 3))
    # object origin: top left
    # obj_pts[:,:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)*square_size
    # object origin: bottom left
    coord_grid = np.array(np.meshgrid(
        np.arange(0, nx, 1), np.arange(ny, 0, -1)))*square_size
    obj_pts[:, :, :2] = np.array(
        [coord_grid[0].T, coord_grid[1].T]).T.reshape(-1, 2)

    for img_nb, point_cloud in enumerate(pnt_clds):
        rgb = np.dstack(
            [point_cloud['r'], point_cloud['g'], point_cloud['b']])
        xyz = np.dstack(
            [point_cloud['x'], point_cloud['y'], point_cloud['z']])
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        ret, corners_in_image = cv2.findChessboardCorners(
            gray, (nx, ny))
        assert(ret == True), f'failed on finding corners, {img_nb}'
        corners_in_image = corners_in_image.reshape(-1, 2)
        cv2.drawChessboardCorners(
            rgb, (nx, ny), corners_in_image, 1)
        rgbs.append(rgb)
        corners.append(corners_in_image)
        XYZs.append(xyz)

    img_pts = np.array(corners).reshape(
        len(pnt_clds), -1, 2)
    img_size = gray.T.shape

    return (img_pts, obj_pts, XYZs, corners, rgbs,  img_size)


def line_line_intersect(p1: float, p2: float, p3: float, p4: float):
    # finds the point of intersection of two lines l1=(p1, p2), l2=(p3,p4)
    denominator = (p1[0]-p2[0])*(p3[1]-p4[1])-(p1[1]-p2[1])*(p3[0]-p4[0])
    point = [((p1[0]*p2[1]-p1[1]*p2[0])*(p3[0]-p4[0])-(p1[0]-p2[0])*(p3[0]*p4[1]-p3[1]*p4[0]))/denominator,
             ((p1[0]*p2[1]-p1[1]*p2[0])*(p3[1]-p4[1])-(p1[1]-p2[1])*(p3[0]*p4[1]-p3[1]*p4[0]))/denominator]
    return point


def center_pts(corners: list, nx: int, ny: int, num_imgs: int):
    # the center point of each square is the average of the four surrounding corners
    center_points = np.zeros(
        (num_imgs, (nx-1)*(ny-1), 2))
    # corners is a list of all the corners for each image
    for img_nb, board_corners in enumerate(corners):
        cps = []  # list to temporarily store the center points.
        for y in range(ny-1):
            for x in range(nx-1):
                idx = y*(nx)+x
                c0 = idx
                c1 = idx+1
                c2 = c0+(nx)
                c3 = c2+1
                # point average
                # cps.append((corners[c0]+corners[c1] +
                #             corners[c2]+corners[c3])/4)
                # intersection point of diagonals
                cps.append(line_line_intersect(
                    board_corners[c0], board_corners[c3], board_corners[c2], board_corners[c1]))
        center_points[img_nb, :, :] = np.array(cps).reshape(-1, 2)

    return center_points


def pnt_cld_transf(pnt_cld1: np.ndarray, pnt_cld2: np.ndarray):
    c1 = np.mean(pnt_cld1, axis=0)
    c2 = np.mean(pnt_cld2, axis=0)

    U, Z, V = np.linalg.svd((pnt_cld2-c2).T @ (pnt_cld1-c1))

    S = np.diag([1, 1, np.linalg.det(V.T@U.T)])
    R = V.T@S@U.T
    t = c1.T - R@c2.T

    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = t
    return HT.from_matrix(T)


def park_martin(A: list, B: list):
    '''Calculates the transformation between the robot end-effector frame and the camera frame.
        Using lists containing pairs of relative motions of camera and robot.
        Method: AX = XB in the style of Park and Martin.

        Parameters: A, B lists of HTransf objects.
        Returns: HTransf object representing the transformation
        between robot end-effector frame and the camera frame
        '''
    K_a = np.zeros((3, len(A)))
    K_b = np.zeros((3, len(B)))

    for i in range(len(A)):
        angle_a, axis_a = A[i].angle_axis()
        angle_b, axis_b = B[i].angle_axis()
        K_a[:, i] = (angle_a*axis_a).flatten()
        K_b[:, i] = (angle_b*axis_b).flatten()
    # calcualte the optimal least squares solution for the
    # rotation matrix between camrea frame and end-effector frame
    H = K_b @ K_a.T
    U, Z, V = np.linalg.svd(H)

    # umeyama correction, to correct for reflection matrices
    S = np.diag([1, 1, np.linalg.det(V.T@U.T)])
    R_optimal = HT.from_matrix(V.T@S@U.T)

    # calculate the best fit translation (dependent on the rotation)
    C = np.zeros((3*len(A), 3))
    d = np.zeros((3*len(B), 1))
    for i in range(0, len(A)*3, 3):
        C[i:i+3, :] = A[int(i/3)].rot_matrix - np.identity(3)
        d[i:i+3, 0] = R_optimal @ B[int(i/3)].get_translation() - \
            A[int(i/3)].get_translation()
        C[i:i+3, :] = np.identity(3) - A[int(i/3)].rot_matrix
        d[i:i+3, 0] = A[int(i/3)].get_translation() - \
            R_optimal @ B[int(i/3)].get_translation()

    t_optimal = np.linalg.lstsq(C, d, rcond=None)[0]

    a, k = R_optimal.angle_axis()
    return HT(a, k, t_optimal)


def calib_err_2D(hec, rob_poses: list, board_poses: list, eye_in_hand=True):
    '''Uses the variation in the constant transformation between
        - robot base frame and object points for eye-in-hand calibration
        - end-effector frame and object points for eye-in-base calibration
        to calculate an estimate for the accuracy of the calibration.
    '''
    if eye_in_hand:
        T_bo = np.array([(rob_pose @ hec @ board_pose).matrix for rob_pose, board_pose in
                         zip(rob_poses, board_poses)]
                        )
    else:
        T_bo = np.array([(rob_pose.inv() @ hec @ board_pose).matrix for rob_pose, board_pose in
                         zip(rob_poses, board_poses)]
                        )

    mean_r = np.mean(np.array([HT.from_matrix(T).rvec()
                               for T in T_bo]), axis=0)
    mean_t = np.mean(T_bo[:, :3, 3], axis=0)
    mean_board_pose = HT.from_vecs(mean_r, mean_t)

    trans_err = np.zeros((len(rob_poses), 3))
    rot_err = np.zeros((len(rob_poses)))
    for i in range(len(rob_poses)):
        Terr = HT.from_matrix(mean_board_pose.inv() @ T_bo[i])
        trans_err[i, :3] = Terr.t
        rot_err[i] = np.rad2deg(Terr.angle_axis()[0])

    return (np.mean(np.linalg.norm(trans_err, axis=1)), np.mean(np.abs(rot_err)))


def calib_err_3D(hec, rob_poses: list, board_pts: list, eye_in_hand=True):
    '''Uses the variation in the constant transformation between
        - robot base frame and object points for eye-in-hand calibration
        - end-effector frame and object points for eye-in-base calibration
        to calculate an estimate for the accuracy of the calibration.
    '''
    if eye_in_hand:
        T_const = np.array(
            [rob_pose @ hec @ board_pnt.T for
             rob_pose, board_pnt in zip(rob_poses, board_pts)]
        ).transpose(0, 2, 1)
    else:
        T_const = np.array(
            [rob_pose.inv() @ hec @ board_pnt.T for
             rob_pose, board_pnt in zip(rob_poses, board_pts)]
        ).transpose(0, 2, 1)

    mean_XYZ = np.mean(T_const, axis=0)

    trans_err = np.zeros((len(rob_poses), 3))
    rot_err = np.zeros((len(rob_poses)))
    for i in range(len(rob_poses)):
        Terr = pnt_cld_transf(mean_XYZ[:, : 3], T_const[i, :, : 3])
        trans_err[i, : 3] = Terr.t
        rot_err[i] = np.rad2deg(Terr.angle_axis()[0])

    return (np.mean(np.linalg.norm(trans_err, axis=1)), np.mean(np.abs(rot_err)))


def interpolate_2D(f: list, xy: list, x: float, y: float):
    # algorithm found at https://en.wikipedia.org/wiki/Bilinear_interpolation
    f1, f2, f3, f4 = f
    x1, x2, y1, y2 = xy

    A = np.array([[1, x1, y1, x1*y1],
                  [1, x1, y2, x1*y2],
                  [1, x2, y1, x2*y1],
                  [1, x2, y2, x2*y2]])
    b = np.array([f1, f2, f3, f4]).T

    a0, a1, a2, a3 = np.linalg.inv(A) @ b
    return a0 + a1 * x + a2 * y + a3 * x * y


def generate_poses(n: int, noise_list: list):

    # groud truth hand eye calibration matrix and chessboard pose
    hec = HT.from_matrix(np.array([[1, 0, 0, 50],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 100],
                                   [0, 0, 0, 1]]))
    cb_pose = HT.from_matrix(np.array([[0, -1, 0, 200],
                                       [1, 0, 0, 70],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]]))

    Ai = []
    Bi = []

    nx = 20
    ny = 13
    obj_pts = np.zeros((nx*ny, 4))
    obj_pts[:, 3] = 1
    # obj_pts[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)*20.0
    obj_pts[:, : 2] = np.mgrid[0: nx, 0: ny].T.reshape(-1, 2)*20.0
    obj_pts[:, : 3] = obj_pts[:, : 3]-np.mean(obj_pts[:, : 3], axis=0)
    Ry_180 = HT(np.pi, np.array([0, 1, 0]))

    # chessboard_pts = cb_pose @ obj_pts.T

    for i in range(n):
        a_rvec = np.pi*np.random.randn(3)
        a_tvec = 300*np.sqrt(3)*np.random.randn(3)
        a_tvec[2] = np.abs(a_tvec[2])
        A_i = HT.from_vecs(a_rvec, a_tvec)
        if A_i.matrix[2, 2] > 0:
            A_i = A_i @ Ry_180

        Ai.append(A_i)

        # temp_Bi = (Ai[i] @ hec).inv() @ cb_pose
        # b_rvec = temp_Bi.rvec()
        # b_tvec = temp_Bi.t# + noise * np.random.randn(3)
        # Bi.append(HT.from_vecs(b_rvec, b_tvec))
        Bi.append((Ai[i] @ hec).inv() @ cb_pose)
    bpts = []
    bplanes = []

    for noise in noise_list:

        board_points = []
        board_poses_plane = []
        for i in range(n):
            point_noise = np.zeros((4, nx*ny))
            point_noise[: 3, :] = noise*np.random.randn(3, nx*ny)
            chessboard_pts = cb_pose @ (obj_pts + point_noise.T).T

            board_points.append(
                (((Ai[i] @ hec).inv() @ chessboard_pts).T))  # + point_noise).T)

            board_poses_plane.append(plane_fit(board_points[i], nx, ny)[0])

        bpts.append(board_points)
        bplanes.append(board_poses_plane)

    return Ai, Bi, bpts, bplanes


def plane_fit(pnt_cld: np.ndarray, nx: int, ny: int):
    U, Z, V = np.linalg.svd(pnt_cld)
    P = V.T[:, 3]
    if P[3] < 0:
        P = -P

    x_ctrl = (pnt_cld[nx-2]-pnt_cld[0]) / \
        np.linalg.norm(pnt_cld[nx-2]-pnt_cld[0])

    # best fit x:
    X = pnt_cld  # np.array(pnt_cld[:nx-1])
    _, _, v = np.linalg.svd(X[:, : 3] - np.mean(X[:, : 3], axis=0))
    x = v[0]/np.linalg.norm(v[0])
    # ensure x is pointing in the correct direction
    if(np.isclose(x, -x_ctrl[0: 3], rtol=0.1).any()):
        x = -x

    # z-axis normal to chessboard plane, and unit vector
    z = P[0: 3]/np.linalg.norm(P[0: 3])
    # z = -v[2]/np.linalg.norm(v[2])
    # if z[2] > 0:
    #     z = -z
    # ensure x is orthogonal to z

    x = x - ((x@z)/np.linalg.norm(z)**2) * z
    x = x/np.linalg.norm(x)
    # y is orthogonal to x and z, xyz is now an orthogonal basis in R3
    y = np.cross(z, x)
    # centroid selected as origin
    t = np.mean(pnt_cld, axis=0)
    # if(t[2]<0):
    #     x=-x
    #     z=-z

    T = np.zeros((4, 4))
    T[: 3, 0] = x[0: 3]
    T[: 3, 1] = y[0: 3]
    T[: 3, 2] = z[0: 3]
    T[:, 3] = t
    T = HT.from_matrix(T)
    return T, P


if __name__ == '__main__':
    pass
