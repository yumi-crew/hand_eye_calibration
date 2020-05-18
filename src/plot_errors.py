from homogeneousTransformations import HTransf as HT
from zividHandEyeCalibrator import ZividHEcalibrator
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *

np.set_printoptions(suppress=True)
if __name__ == '__main__':
   
    folder = 'datasets/yumi_27-02-20'

    sq = 5
    nx = 8
    ny = 5

    calibrator2D = ZividHEcalibrator(sqrSize=sq, nx=nx, ny=ny, eye_in_hand=False)

    calibrator2D.load_zdfs(folder)
    calibrator2D.load_robot_poses(folder, rot_repr='quaternion')
    calibrator2D.calculate_chessboard_poses_2D()
    t_errs2D, rot_errs2D = calibrator2D.err_iter()

    calibrator3D = ZividHEcalibrator(sqrSize=sq, nx=nx, ny=ny, eye_in_hand=False)
    calibrator3D.load_zdfs(folder)
    calibrator3D.load_robot_poses(folder, rot_repr='quaternion')
    calibrator3D.calculate_chessboard_poses_3D()
    t_errs3D, rot_errs3D = calibrator3D.err_iter()
    calibrator3D.method = '2D'
    t_errs3D_plane_fit, rot_errs3D_plane_fit = calibrator3D.err_iter()
 ##### plt #####
    f1 = plt.figure('Translation error')
    plt.plot(range(3, calibrator2D.num_imgs+1), t_errs3D, label='3D')
    plt.plot(range(3, calibrator2D.num_imgs+1),
             t_errs3D_plane_fit, label='3D plane fit')
    plt.plot(range(3, calibrator2D.num_imgs+1), t_errs2D, label='openCV')
    plt.xlabel('Pose pairs')
    plt.ylabel('Translation estimation error [mm]')
    plt.legend()

    f2 = plt.figure('Rotation error')
    plt.plot(range(3, calibrator3D.num_imgs+1), rot_errs3D, label='3D')
    plt.plot(range(3, calibrator3D.num_imgs+1),
             rot_errs3D_plane_fit, label='3D plane fit')
    plt.plot(range(3, calibrator3D.num_imgs+1), rot_errs2D, label='openCV')
    plt.xlabel('Pose pairs')
    plt.ylabel('Rotation estimation error [deg]')
    plt.legend()
    plt.show()
