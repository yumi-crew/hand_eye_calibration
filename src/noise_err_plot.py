import numpy as np
import matplotlib.pyplot as plt

t_errs_3D = [0.02581530, 0.0559382, 0.0697547, 0.0771030851, 0.11530405,
             0.2295124, 0.470901409, 0.5921750, 0.850088, 1.236484, 2.25496813]
t_errs_planar = [0.01683837, 0.05195553, 0.06480726, 0.060774, 0.1008311,
                 0.1788238, 0.3832746, 0.6876377, 0.6444923, 0.77810277, 1.9951203]
r_errs_3D = [0.00496178, 0.010198554, 0.0172963, 0.0200701, 0.0288756,
             0.0503679, 0.1179821, 0.16360833, 0.214646, 0.2799039, 0.56434963]
r_errs_planar = [0.002617398, 0.01257257, 0.0221942, 0.0282148, 0.0344485,
                 0.0709794, 0.160879, 0.21507925, 0.2922264, 0.388456, 0.839990]
noise = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]

f1 = plt.figure('noise translation error relationship')
plt.scatter(noise[0:5], t_errs_3D[0:5])
plt.plot(noise[0:5], np.poly1d(np.polyfit(
    noise[0:5], t_errs_3D[0:5], 1))(noise[0:5]), label='3D')
plt.scatter(noise[0:5], t_errs_planar[0:5])
plt.plot(noise[0:5], np.poly1d(np.polyfit(
    noise[0:5], t_errs_planar[0:5], 1))(noise[0:5]), label='3D plane fit')
plt.legend()
plt.xlabel('Noise standard deviation [mm]')
plt.ylabel('Translation estimation error [mm]')


f2 = plt.figure('noise rotation error relationship')
plt.scatter(noise[0:5], r_errs_3D[0:5])
plt.plot(noise[0:5], np.poly1d(np.polyfit(
    noise[0:5], r_errs_3D[0:5], 1))(noise[0:5]), label='3D')
plt.scatter(noise[0:5], r_errs_planar[0:5])
plt.plot(noise[0:5], np.poly1d(np.polyfit(
    noise[0:5], r_errs_planar[0:5], 1))(noise[0:5]), label='3D plane fit')
plt.legend()
plt.xlabel('Noise standard deviation [mm]')
plt.ylabel('Rotation estimation error [deg]')
plt.show()
