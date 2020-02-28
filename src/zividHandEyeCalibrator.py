import numpy as np
import zivid
import cv2
from homogeneousTransformations import HTransf as HT
import glob
import itertools
import concurrent.futures
import multiprocessing
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import random
from scipy import interpolate
from utils import *

np.set_printoptions(suppress=True)


class ZividHEcalibrator(object):
    '''
            Description of class functionality
    '''

    def __init__(self, nx: int, ny: int, sqrSize: float, eye_in_hand: bool = True):
        self.app = zivid.Application()
        self.nx = nx
        self.ny = ny
        self.square_size = sqrSize
        self.images = []
        self.num_imgs = 0
        self.point_clouds = []
        self.XYZs = []
        self.chessboard_poses = []
        self.camera_poses = []
        self.robot_poses = []
        self.rgbs = []
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_files = []
        self.pose_files = []
        self.corners = []
        self.center_points = []
        self.object_planes = []
        self.board_points = []
        self.rvecs = []
        self.tvecs = []
        self.HE_calib = None
        self.method = ''
        self.eye_in_hand = eye_in_hand

    def load_robot_poses(self, path: str, rot_repr='SO(3)'):
        print(f'Loading robot poses from ./{path}')
        if rot_repr == 'SO(3)':
            self.pose_files, self.robot_poses = load_t4s(path)
        elif rot_repr == 'quaternion':
            self.pose_files, self.robot_poses = load_quat_poses(path)
        if np.mean(self.robot_poses[0].t) < 1:
            print('converting from m to mm')
            for idx, _ in enumerate(self.robot_poses):
                self.robot_poses[idx] = self.robot_poses[idx].to_mm()

    # NON PARALELLIZED METHOD FOR LOADING IMGS

    def load_zdfs(self, path: str):
        print(f'Loading .zdf images from ./{path}')
        self.image_files = sorted(glob.glob(f'{path}/*.zdf'))
        self.images = [zivid.Frame(file) for file in self.image_files]
        self.point_clouds = [image.get_point_cloud().to_array()
                             for image in self.images]
        self.num_imgs = len(self.point_clouds)

    # PARALLELIZED METHODS FOR LOADING IMAGES
    def load_zdfs_parallel(self, folder_path: str):
        print(f'Loading .zdf images from ./{folder_path}')
        self.image_files = sorted(glob.glob(f'{folder_path}/*.zdf'))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            imageData = [executor.submit(load_zdf, file)
                         for file in self.image_files]
            self.point_clouds = [image.result() for image in imageData]
        self.num_imgs = len(self.point_clouds)

    def calculate_chessboard_poses_2D(self):
        '''
        Calculates the poses of the calibration objects in self.point_clouds using
        opencv camera calibration, and strores them in the list self.chessboard_poses.
        If the intrinsic parameters of the camera are known,
        and set with the method self.setIntrinsics(...) cv2.solvePnP(...) is used, else
        cv2.CalibrateCamera(...) is used to calibrate the intrinsic and extrinsic camera
        parameters.
        '''
        assert(len(self.point_clouds) > 0), 'No calibration images loaded'
        print('Calculating chessboard poses')
        self.method = '2D'
        img_pts, obj_pts, self.XYZs, self.corners, self.rgbs, img_size = calibration_pts(
            self.nx, self.ny, self.square_size, self.point_clouds)

        if self.camera_matrix is not None and self.dist_coeffs is not None:
            print('solvepnp')
            for i in range(len(self.point_clouds)):
                _, rvec, tvec = \
                    cv2.solvePnP(obj_pts[i].astype('float32'), img_pts[i].astype(
                        'float32'), self.camera_matrix, self.dist_coeffs)
                self.rvecs.append(rvec)
                self.tvecs.append(tvec)
        else:
            _, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = \
                cv2.calibrateCamera(obj_pts.astype(
                    'float32'), img_pts.astype('float32'), img_size, None, None)

        self.chessboard_poses = [HT.from_vecs(r, t) for (
            r, t) in zip(self.rvecs, self.tvecs)]

    def calculate_chessboard_poses_3D(self):
        '''
        Calculates the poses of the calibration objects in self.point_clouds,
        using the 3D data from the zivid structured light scanner
        '''
        assert(len(self.point_clouds) > 0), 'No calibration images loaded'
        print('Calculating chessboard poses (3D)')
        self.method = '3D'
        img_pts, obj_pts, self.XYZs, self.corners, self.rgbs, img_size = calibration_pts(
            self.nx, self.ny, self.square_size, self.point_clouds)

        self.center_points = center_pts(
            self.corners, self.nx, self.ny, self.num_imgs)

        # draw the centerpoints
        for imgNb, rgb in enumerate(self.rgbs):
            for point in self.center_points[imgNb]:
                cv2.circle(rgb, tuple(point.astype('int')),
                           radius=3, color=(255, 0, 0), thickness=-1)
            cv2.circle(rgb, tuple(
                self.center_points[imgNb][-(self.nx-1)].astype('int')), radius=3, color=(0, 0, 255), thickness=-1)
            # cv2.imshow(f'hei{imgNb}', cv2.resize(rgb, (0,0), fx=0.5, fy=0.5))
        # cv2.waitKey(0)

        # use the xyz coordinates of the centerpoints to estimate the pose of the chessboard in the camera frame.
        # self.XYZs[imgNb][px, py] -> [x, y, z]
        # find the best fint plane P to each set of center points A, so that AP = 0

        for xyz, cps in zip(self.XYZs, self.center_points):
            xyz = xyz.transpose(1, 0, 2)
            A = np.zeros((cps.shape[0], 4))
            A[:, 3] = 1

            for idx, row in enumerate(cps):
                # A[idx, :3] = xyz[row[0], row[1]]

                px, py = row[0], row[1]

                x1 = np.floor(px).astype('int')
                x2 = np.ceil(px).astype('int')
                y1 = np.floor(py).astype('int')
                y2 = np.ceil(py).astype('int')

                if x1 == x2:
                    x2 += 1
                if y1 == y2:
                    y2 += 1

                X, Y, Z = np.array([xyz[x1, y1],
                                    xyz[x2, y1],
                                    xyz[x1, y2],
                                    xyz[x2, y2]]).T
                # interpolation:
                A[idx, 0] = interpolate_2D(X, [x1, x2, y1, y2], px, py)
                A[idx, 1] = interpolate_2D(Y, [x1, x2, y1, y2], px, py)
                A[idx, 2] = interpolate_2D(Z, [x1, x2, y1, y2], px, py)

            A = A[~np.isnan(A).any(axis=1)]
            self.board_points.append(A)

            T, P = plane_fit(A, self.nx, self.ny)
            self.chessboard_poses.append(T)
            self.object_planes.append(P)
            self.tvecs.append(T.t)
            self.rvecs.append(T.rvec())

    def draw_coord_sys(self, imgNb: int):
        '''
        Draws coordinate systems on the chessboard in the list self.rgbs corresponding to the index imgNb.
        Requres the camera intrisics to be set in self.camera_matrix and self.dist_coeffs, 
        use set_intrinsics() method
        Parameters: imgNb: int
        returns: img: ndarray
        '''
        # axis = (np.float32([[1,1,0], [0,2,0], [0,1,1], [0,0,0]])*self.square_size)
        axis = (np.float32([[2, 0, 0], [0, 2, 0], [
                0, 0, 2], [0, 0, 0]])*self.square_size)
        imgpts, jac = cv2.projectPoints(
            axis, self.rvecs[imgNb], self.tvecs[imgNb], self.camera_matrix, self.dist_coeffs)
        btmLftCrn = (self.ny-1)*self.nx
        corner = tuple(self.corners[imgNb][btmLftCrn].ravel())
        origin = tuple(imgpts[3].ravel())
        img = self.rgbs[imgNb]
        # cv2.circle(img, tuple(imgpts[3].ravel()), radius=10, color=(255,0,0), thickness=-1) #origin
        img = cv2.line(img, origin, tuple(imgpts[0].ravel()), (0, 0, 255), 3)
        img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (0, 255, 0), 3)
        img = cv2.line(img, origin, tuple(imgpts[2].ravel()), (255, 0, 0), 3)
        return img

    def viz_cam_pose(self, focus: str = 'objectCentric', axLen: int = 50):
        '''
        Visualizes each camera pose in the object coordinate system.
        x-axis:red, y-axis:green, z-axis:blue
        '''
        if focus == 'cameraCentric':
            name = 'Object Poses'
            f1 = 'Camrea Frame'
            f2 = 'O'
        else:
            name = 'Camera Poses'
            f1 = 'Object Frame'
            f2 = 'C'
        fig = plt.figure(name)
        ax = plt.axes(projection='3d')
        # object frame: draw coordinate frame and chessboard
        x, y, z = np.zeros((3, 3))
        u, v, w = np.array([[axLen, 0, 0], [0, axLen, 0], [0, 0, axLen]])
        ax.quiver(x, y, z, u, v, w, color=np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        ax.text(0+10, 0+10, 0, s=f1)
        boardColor = np.array([0, 0, 0])
        ax.plot3D(np.array([0, self.square_size*self.nx]),
                  np.array([0, 0]), np.array([0, 0]), color=boardColor)
        ax.plot3D(np.array([0, self.square_size*self.nx]), np.array([self.square_size *
                                                                     self.ny, self.square_size*self.ny]), np.array([0, 0]), color=boardColor)
        ax.plot3D(np.array([0, 0]), np.array(
            [0, self.square_size*self.ny]), np.array([0, 0]), color=boardColor)
        ax.plot3D(np.array([self.square_size*self.nx, self.square_size*self.nx]),
                  np.array([0, self.square_size*self.ny]), np.array([0, 0]), color=boardColor)
        # camera frames:
        for idx, pose in enumerate(self.chessboard_poses):
            # chessboardPoses: HTransf: camera -> chessboard
            if focus == 'objectCentric':
                pose = pose.inv()  # HTransf: chessboard -> camera
            campos = pose.matrix[0:3, 3]
            x, y, z = np.array([campos, campos, campos]).T
            dirs = pose.rot_matrix*axLen
            u, v, w = dirs
            ax.quiver(x, y, z, u, v, w, color=np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            ax.text(x[0]+10, y[0]+10, z[0]+10, s=f'{f2}{idx}')
        plt.show(block=False)

    def viz_rob_pose(self, axLen: int = 50):
        # visualizes the end-effector pose in the robot base frame
        fig = plt.figure('Robot Poses')
        ax = plt.axes(projection='3d')
        # Base frame:
        x, y, z = np.zeros((3, 3))
        u, v, w = np.array([[axLen, 0, 0], [0, axLen, 0], [0, 0, axLen]])
        ax.quiver(x, y, z, u, v, w, color=np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        ax.text(0+10, 0+10, 0, s='Base frame')
        # end-effector frames:
        for idx, pose in enumerate(self.robot_poses):
            robpos = pose.matrix[0:3, 3]
            x, y, z = np.array([robpos, robpos, robpos]).T
            u, v, w = pose.rot_matrix*axLen
            ax.quiver(x, y, z, u, v, w, color=np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            ax.text(x[0]+10, y[0]+10, z[0]+10, s=f'EF{idx}')
        plt.show(block=False)

    def viz_HE_transf(self, X_HE: list, showTranslation: bool = True, inCamera=False):
        # visualizes the hand-eye transformation from end-effector frame to camera frame.
        fig = plt.figure('Camera frame in end-effector coordinates')
        ax = plt.axes(projection='3d')
        axLen = 50
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_zlim(-200, 200)
        if inCamera:
            f1 = 'C'
            f2 = 'EF'
        else:
            f1 = 'EF'
            f2 = 'c'
        # base frame:
        x, y, z = np.zeros((3, 3))
        u, v, w = np.array([[axLen, 0, 0], [0, axLen, 0], [0, 0, axLen]])
        ax.quiver(x, y, z, u, v, w, color=np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        ax.text(10, 10, 10, s=f1)
        for idx, X in enumerate(X_HE):
            if inCamera:
                X = X.inv()
            pos = X.matrix[0:3, 3]
            if showTranslation:
                x, y, z = np.array([pos, pos, pos]).T
            u, v, w = X.rot_matrix*axLen
            ax.quiver(x, y, z, u, v, w, color=np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            ax.text(x[0]+10, y[0]+10, z[0]+10, s=f'{f2}{idx}')
        plt.show(block=False)

    def set_intrinsics(self, pathToK: str, pathToDist: str):
        '''Sets the intrinsic camera parameters from text files
        Parameters:
                pathToK: str, path to the text file containing the camera matrix
                pathToDist: str, path to the text file containing the distortion coeffs.
        '''
        self.camera_matrix = np.fromfile(
            pathToK, dtype=float, count=-1, sep=' ').reshape(3, 3)
        self.dist_coeffs = np.fromfile(
            pathToDist, dtype=float, count=-1, sep=' ')

    def calculate_relative_poses(self, pose_pairs: int = -1, use_board_pts: bool = True):
        '''Calculates the relative transformation between all combinations of end-effector poses
        and all corresponding combinations of camera/object poses (eye-in-hand/eye-in-base).

        Returns:
                Ai : list of HTransf objects, transformations between end-effector poses
                Bi : list of HTransf objects, transformations between camera poses
        '''
        # print('Calculating transformations between all available pose-combinations')
        assert(len(self.chessboard_poses) > 0), 'No object poses loaded'
        assert(len(self.robot_poses) > 0), 'No end-effector poses loaded'
        assert(len(self.chessboard_poses) == len(self.robot_poses)
               ), 'Mismatch in number of object and end-effector poses'
        Ai = []
        Bi = []
        if pose_pairs < 3 or pose_pairs > self.num_imgs:
            pose_pairs = len(self.point_clouds)
        combinations = itertools.combinations(list(range(pose_pairs)), 2)
        if self.eye_in_hand:
            if self.method == '3D' and use_board_pts:
                # use the xyz-object points on the chessboard to calculate relative camera poses
                for first, second in combinations:
                    Ai.append(self.robot_poses[second].inv()
                              @ self.robot_poses[first])
                    assert(self.board_points[first][:, :3].shape == self.board_points[second]
                           [:, :3].shape), f'Nan points in image {first} or {second}'
                    Bi.append(pnt_cld_transf(
                        self.board_points[second][:, :3], self.board_points[first][:, :3]))
            else:
                for first, second in combinations:
                    Ai.append(self.robot_poses[second].inv()
                              @ self.robot_poses[first])
                    Bi.append(self.chessboard_poses[second]
                              @ self.chessboard_poses[first].inv())
        else:
            # cam-in-base
            if self.method == '3D' and use_board_pts:
                # use the xyz-object points on the chessboard to calculate relative camera poses
                for first, second in combinations:
                    Ai.append(self.robot_poses[second]
                              @ self.robot_poses[first].inv())
                    assert(self.board_points[first][:, :3].shape == self.board_points[second]
                           [:, :3].shape), f'Nan points in image {first} or {second}'
                    Bi.append(pnt_cld_transf(
                        self.board_points[second][:, :3], self.board_points[first][:, :3]))
            else:
                for first, second in combinations:
                    Ai.append(self.robot_poses[second]
                              @ self.robot_poses[first].inv())
                    Bi.append(self.chessboard_poses[second]
                              @ self.chessboard_poses[first].inv())

        return Ai, Bi

    def HE_calibration(self, A: list, B: list):
        self.HE_calib = park_martin(A, B)
        return self.HE_calib

    def calib_error(self):
        '''
        calculates the position of the calibration object in the robot base-frame,
        based on each the end-effector pose, camera pose and the hand eye calibration.
        The mean values for position and rotation angle as well as the standard deviations
        of the position and rotation estimates are printed.
        '''

        if self.method == '2D':
            trans_err, rot_err = calib_err_2D(
                self.HE_calib, self.robot_poses, self.chessboard_poses, eye_in_hand=self.eye_in_hand)
        else:
            trans_err, rot_err = calib_err_3D(
                self.HE_calib, self.robot_poses, self.board_points, eye_in_hand=self.eye_in_hand)
        print(trans_err, rot_err)

    def err_iter(self, plot=False):
        rot_errs = []
        trans_errs = []

        for i in range(3, self.num_imgs+1):
            print(i)

            if self.method == '2D':
                Ai, Bi = self.calculate_relative_poses(
                    pose_pairs=i, use_board_pts=False)
                hec = self.HE_calibration(Ai, Bi)
                trans_err, rot_err = calib_err_2D(
                    hec, self.robot_poses, self.chessboard_poses, eye_in_hand=self.eye_in_hand)
            else:
                Ai, Bi = self.calculate_relative_poses(
                    pose_pairs=i, use_board_pts=True)
                hec = self.HE_calibration(Ai, Bi)
                trans_err, rot_err = calib_err_3D(
                    hec, self.robot_poses, self.board_points, eye_in_hand=self.eye_in_hand)
            trans_errs.append(trans_err)
            rot_errs.append(rot_err)

        if plot:
            f1 = plt.figure('Translation')
            plt.plot(range(3, self.num_imgs+1), trans_errs)
            f2 = plt.figure('Rotation')
            plt.plot(range(3, self.num_imgs+1), rot_errs)
            plt.show(block=False)
        print(hec)
        print(trans_err, rot_err)
        return trans_errs, rot_errs


if __name__ == '__main__':

    '''example taken from You Cheung Shiu et al.
       "calibration of wrist-mounted robotic sensors by solving 
       homogeneous transformation equations of the form AX=XB"
    '''
    X_act = HT(a=0.2, k=np.array([1, 0, 0]), t=np.array([10, 50, 100]))
    A1 = HT(a=3.0, k=np.array([0, 0, 1]), t=np.array([0, 0, 0]))
    A2 = HT(a=1.5, k=np.array([0, 1, 0]), t=np.array([-400, 0, 400]))
    B1 = X_act.inv() @ A1 @ X_act
    B2 = X_act.inv() @ A2 @ X_act
    # print(A1, A2, B1, B2)
    A11 = np.array([[-0.989992, -0.141120, 0.0, 0.0],
                    [0.141120, -0.989992, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1]])
    B11 = np.array([[-0.989992, -0.138307, 0.028036, -26.9559],
                    [0.138307, -0.911449, 0.387470, -96.1332],
                    [-0.028036, 0.387470, 0.921456, 19.4872],
                    [0.0, 0.0, 0.0, 1]])
    A22 = np.array([[0.070737, 0.0, 0.997495, -400.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-0.997495, 0.0, 0.070737, 400.0],
                    [0.0, 0.0, 0.0, 1]])
    B22 = np.array([[0.070737, 0.198172, 0.997612, -309.543],
                    [-0.198172, 0.963323, -0.180936, 59.0244],
                    [-0.977612, -0.180936, 0.107415, 291.177],
                    [0.0, 0.0, 0.0, 1]])

    calibrator = ZividHEcalibrator(sqrSize=None, nx=None, ny=None)
    print('Fasit:\n', X_act)
    T1 = calibrator.HE_calibration([HT.from_matrix(A11), HT.from_matrix(A22)], [
        HT.from_matrix(B11), HT.from_matrix(B22)])
    T2 = calibrator.HE_calibration([A1, A2], [B1, B2])
    calibrator.viz_HE_transf([T1])
    print(T1, T2)
    plt.show()
