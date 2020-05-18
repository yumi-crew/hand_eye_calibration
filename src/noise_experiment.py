import numpy as np
from homogeneousTransformations import HTransf as HT
from zividHandEyeCalibrator import ZividHEcalibrator
from utils import *
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True)
if __name__ == '__main__':

    noise = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]

    logdir = 'noise_exps/noise_exp5'

    calibrator = ZividHEcalibrator(15.0, 20, 13)
    calibrator.num_imgs = 50

    A, B, board_points, board_poses_plane = generate_poses(
        calibrator.num_imgs, noise)
    calibrator.robot_poses = A
    calibrator.chessboard_poses = B
    with open(f'{logdir}/log.txt', 'w') as log:
        log.write('     t_err,      rot_err\n')
        for i, n in enumerate(noise):
            calibrator.board_points = board_points[i]
            calibrator.chessboard_poses = B
            calibrator.method = '3D'
            t_errs3D, rot_errs3D = calibrator.err_iter()
            log.write(
                f'noise={n}, {t_errs3D[-1]}, {rot_errs3D[-1]},    (3D)\n')
            calibrator.chessboard_poses = board_poses_plane[i]
            calibrator.method = '2D'
            t_errs3D_plane, rot_errs3D_plane = calibrator.err_iter()
            log.write(
                f'noise={n}, {t_errs3D_plane[-1]}, {rot_errs3D_plane[-1]},    (3D planar)\n')
            f1 = plt.figure(f'Translation error, \u03C3 = {n} [mm]')
            plt.plot(range(3, calibrator.num_imgs+1), t_errs3D, label='3D')
            plt.plot(range(3, calibrator.num_imgs+1),
                     t_errs3D_plane, label='3D plane fit')

            plt.xlabel('Pose pairs')
            plt.ylabel('Translation estimation error [mm]')
            plt.legend()
            plt.savefig(f'{logdir}/Terr_tn{n}.png')

            f2 = plt.figure(f'Rotation error, \u03C3 = {n} [deg]')
            plt.plot(range(3, calibrator.num_imgs+1), rot_errs3D, label='3D')
            plt.plot(range(3, calibrator.num_imgs+1),
                     rot_errs3D_plane, label='3D plane fit')

            plt.xlabel('Pose pairs')
            plt.ylabel('Rotation estimation error [deg]')
            plt.legend()
            plt.savefig(f'{logdir}/Rerr_tn{n}.png')
    #plt.show()
