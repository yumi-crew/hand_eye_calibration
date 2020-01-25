from homogeneousTransformations import HTransf as HT
from zividHandEyeCalibrator import ZividHEcalibrator
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *

np.set_printoptions(suppress=True)
if __name__ == '__main__':
    # 13.9
    folder = 'datasets/captures02-12-19'
    #folder = 'datasets/dataset_zivid'
    sq = 13.8
    nx = 20
    ny = 13
    calibrator = ZividHEcalibrator(sqrSize=sq, nx=nx, ny=ny)

    calibrator.load_zdfs(folder)
    calibrator.load_robot_poses(folder, rot_repr='quaternion')

    calibrator.calculate_chessboard_poses_3D()
    calibrator.viz_cam_pose(focus='objectCentric')
    calibrator.viz_rob_pose()
    calibrationMatrices = []
    
    Ai, Bi = calibrator.calculate_relative_poses(
        pose_pairs=-1, use_board_pts=True)
    calibrationMatrices.append(calibrator.HE_calibration(Ai, Bi))

    print(f'HE calibration:\n{calibrationMatrices[-1]}')
    calibrator.viz_HE_transf(calibrationMatrices, showTranslation=True)
    print(calibrator.method)
    calibrator.calib_error()
    calibrator.err_iter(plot=True)

    # show boards with coordinate systems:
    # for imgNb in range(0, len(calibrator.point_clouds)):
    #     img = calibrator.draw_coord_sys(imgNb)
    #     cv2.imshow(f'board {imgNb}', cv2.resize(img, (0, 0), fx=0.5, fy=0.5))

    plt.show()
