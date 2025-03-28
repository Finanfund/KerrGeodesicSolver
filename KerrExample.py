import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from KerrParameters import KerrParameters_1 as KerrParameters
from FullKerr.KerrGeoFast import KerrGeodesic_1 as KerrGeodesic
from GeodesicVisualization import GeodesicVisualization
import time

start_time = time.perf_counter()  # 开始计时

# 定义初始参数
a = 0.1
delta_lambda = 0.01
max_lambda = 100000  # 定义最大 lambda
N = int(max_lambda / delta_lambda)
# N = 10000
print(f"N: {N}")
z1 = 0.7
r1_initial = 6
r2_initial = 6
M = 1.0  # 黑洞质量，暂时只能为1

# 定义动画速度
skip_lambda = 1  # 每帧跳过的 lambda 值
skip_steps = int(skip_lambda / delta_lambda)

# 定义模式变量，0表示运行所有功能，1表示绘制分量演化，2表示绘制3D轨迹，3表示创建3D动画
mode = 3

# 获取 KerrParameters 的参数
def initialize_parameters(a, z1, r1_initial, r2_initial, M):
    try:
        bh_parameters = KerrParameters(a, z1, r1_initial, r2_initial, M=M)
        parameters = bh_parameters.parameter()
    except Exception as e:
        print(f"初始化 KerrParameters 时出错: {e}")
        exit()
    return parameters

# 初始化 KerrGeodesic 对象
def initialize_kerr(parameters):
    try:
        E, L, C, r1, r2, r3, r4, z1, z2 = [
            parameters[key] for key in ["E", "L", "C", "r1", "r2", "r3", "r4", "z1", "z2"]
        ]
    except KeyError as e:
        print(f"参数缺失: {e}")
        exit()

    try:
        kerr = KerrGeodesic(
            a=a,
            r1=r1,
            r2=r2,
            z1=z1,
            delta_lambda=delta_lambda,
            N=N
        )
    except Exception as e:
        print(f"初始化 KerrGeodesic 时出错: {e}")
        exit()
    return kerr

# 获取绘图结果
def get_results(kerr):
    try:
        results = kerr.get_results()
    except Exception as e:
        print(f"获取结果时出错: {e}")
        exit()
    return results

parameters = initialize_parameters(a, z1, r1_initial, r2_initial, M=M)
kerr = initialize_kerr(parameters)
results = get_results(kerr)

lambda_array = kerr.lambda_array
r_lambda = results['r_lambda']
z_lambda = results['z_lambda']
phi_lambda = results['phi_lambda']
t = results['t_lambda']
sin_theta = np.sqrt(1 - z_lambda**2)

# 将球坐标转换为笛卡尔坐标
x = r_lambda * np.cos(phi_lambda) * sin_theta
y = r_lambda * np.sin(phi_lambda) * sin_theta
Z = r_lambda * z_lambda

# 球坐标初始方向转换旋转矩阵
theta = - np.pi / 2 + np.arccos(z1)
phi_ini =  0

# # 旋转矩阵实现
# def rotation_matrix(theta, phi):
#     R_z = np.array([
#         [np.cos(phi), -np.sin(phi), 0],
#         [np.sin(phi), np.cos(phi), 0],
#         [0, 0, 1]
#     ])
#     R_y = np.array([
#         [np.cos(-theta), 0, np.sin(-theta)],
#         [0, 1, 0],
#         [np.sin(theta), 0, np.cos(-theta)]
#     ])
#     return R_y @ R_z

# # 应用旋转矩阵
# R = rotation_matrix(theta, phi_ini)
# rotated_coords = np.dot(R, np.array([x, y, Z]))
# x, y, Z = rotated_coords

# parameters = initialize_parameters(0, z1, r1_initial, r2_initial, M=M)
# kerr = initialize_kerr(parameters)
# results = get_results(kerr)

# lambda_array = kerr.lambda_array
# r2_lambda = results['r_lambda']
# z2_lambda = results['z_lambda']
# phi2_lambda = results['phi_lambda']
# t2 = results['t_lambda']
# sin_theta = np.sqrt(1 - z2_lambda**2)

# # 将球坐标转换为笛卡尔坐标
# x2 = r2_lambda * np.cos(phi2_lambda) * sin_theta
# y2 = r2_lambda * np.sin(phi2_lambda) * sin_theta
# Z2 = r2_lambda * z2_lambda
end_time = time.perf_counter()  # 结束计时
print(f"运行时间: {end_time - start_time:.2f} 秒")
# 根据 mode 选择要运行的功能
visualizer = GeodesicVisualization()
if mode == 0 or mode == 1:
    visualizer.plot_evolution(lambda_array,t,r_lambda,z_lambda,phi_lambda)
if mode == 0 or mode == 2:
    visualizer.plot_3d_trajectory(x, y, Z, parameters['r1'], 1.2)
if mode == 0 or mode == 3:
    visualizer.animate_3d_trajectory(x, y, Z, skip_steps, parameters['r1'], 1.2)
# if mode == 0 or mode == 4:
#     visualizer.plot_evolution_cartesian(lambda_array, x, y, Z, t, 1.0)
# if mode == 0 or mode == 5:
#     visualizer.plot_trajectory_difference(lambda_array, t, x, y, Z, t2, x2, y2, Z2)
# if mode == 0 or mode == 6:
#     # visualizer.plot_geodesic_difference_rtphi(t, r_lambda, z_lambda, phi_lambda, 0*t2, 0*r2_lambda, 0*z2_lambda, 0*phi2_lambda)
#     visualizer.plot_geodesic_difference_rtphi(t, r_lambda, z_lambda, phi_lambda, t2, r2_lambda, z2_lambda, phi2_lambda)