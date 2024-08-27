import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transform_manager import TransformManager
from scipy.spatial.transform import Rotation

T_D435_2_LiDARup = np.eye(4)
T_D435_2_LiDARup = \
[[-8.90117938e-09, -3.66518682e-09,  1.00000000e+00,  1.01875613e-01],
 [-1.00000000e+00,  5.23598776e-07, -8.90117743e-09,  5.81658210e-02],
 [-5.23598776e-07, -1.00000000e+00, -3.66519148e-09, -7.01829509e-02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
T_LiDARup_2_D435 = np.linalg.inv(T_D435_2_LiDARup)

T_LiDARdown_2_LiDARup = np.eye(4)
T_LiDARdown_2_LiDARup = \
[[ 9.99915501e-01, -1.29996338e-02,  4.29963666e-07,  9.20501009e-02],
 [-1.29996338e-02, -9.99915501e-01, -5.58984243e-09, -9.36340880e-03],
 [ 4.30000000e-07, -1.22464679e-16, -1.00000000e+00, -1.01585960e-01],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
T_LiDARup_2_LiDARdown = np.linalg.inv(T_LiDARdown_2_LiDARup)

T_LiDARup_2_IMUup = np.eye(4)
T_LiDARup_2_IMUup = \
[[ 1.00000000e+00,  7.15584997e-11, -1.04719755e-10, -1.10000000e-02],
 [-7.15584993e-11,  1.00000000e+00,  3.66519143e-09, -2.32899998e-02],
 [ 1.04719755e-10, -3.66519143e-09,  1.00000000e+00,  4.41200001e-02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
T_IMUup_2_LiDARup = np.linalg.inv(T_LiDARup_2_IMUup)

T_IMU_2_LiDARup = np.eye(4)
T_IMU_2_LiDARup = \
[[ 1.00000000e+00,  1.06465084e-09, -1.91986049e-11,  4.10999995e-04],
 [-1.06465084e-09,  1.00000000e+00,  1.58824962e-08, -6.73500172e-03],
 [ 1.91986218e-11, -1.58824962e-08,  1.00000000e+00, -1.08357000e-01],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
T_LiDARup_2_IMU = np.linalg.inv(T_IMU_2_LiDARup)

# Print in IMU_mti
print("IMU_mti to Mid-360 up: ", T_IMU_2_LiDARup)
print("IMU_mti to Mid-360 down: ", T_LiDARup_2_LiDARdown@T_IMU_2_LiDARup)
print("IMU_mti to D435_infra1: ", T_LiDARup_2_D435@T_IMU_2_LiDARup)
print("Mid-360 down to D435_infra1: ", T_LiDARup_2_D435@T_LiDARdown_2_LiDARup)

tm = TransformManager()
tm.add_transform("D435_infra1", "Mid-360 up", T_D435_2_LiDARup)
# tm.add_transform("D435_infra1", "Mid-360 down", T_D435_2_LiDARdown)
tm.add_transform("Mid-360 down", "Mid-360 up", T_LiDARdown_2_LiDARup)
tm.add_transform("Mid-360 up", "IMU Mid-360", T_LiDARup_2_IMUup)
tm.add_transform("IMU_mti", "Mid-360 up", T_IMU_2_LiDARup)

plt.figure(figsize=(12, 12))

ax = make_3d_axis(ax_s=1)
ax = tm.plot_frames_in("IMU_mti", ax=ax, s=0.07)
#ax.plot(*origin_c_in_b[:3],"rx")
#ax.plot(*origin_b_in_d[:3],"bx")
#ax.view_init(30, 20)
ax.set_xlim3d(-0.2, 0.2)
ax.set_ylim3d(-0.2, 0.2)
ax.set_zlim3d(-0.2, 0.2)
plt.show()