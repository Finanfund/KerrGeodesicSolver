import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from BlackHoleOrbitParameter import BlackHoleOrbitParameter
from KerrGeodesic import KerrGeodesic
from GeodesicVisualization import GeodesicVisualization

# 定义初始参数
a = 0.6
delta_lambda = 0.003
max_lambda = 50.0  # 定义最大 lambda
N = int(max_lambda / delta_lambda)
N = 50000
print(f"N: {N}")
z1 = 0.7
r1_initial = 7.0
r2_initial = 6.0
M = 1.0  # 黑洞质量，暂时只能为1

# 定义动画速度
skip_lambda = 0.1  # 每帧跳过的 lambda 值
skip_steps = int(skip_lambda / delta_lambda)

# 定义模式变量，0表示运行所有功能，1表示绘制分量演化，2表示绘制3D轨迹，3表示创建3D动画
mode = 0

# 获取 BlackHoleOrbitParameter 的参数
def initialize_parameters():
    try:
        bh_parameters = BlackHoleOrbitParameter(a, z1, r1_initial, r2_initial, M=M)
        parameters = bh_parameters.parameter()
    except Exception as e:
        print(f"初始化 BlackHoleOrbitParameter 时出错: {e}")
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
            E=E,
            L=L,
            C=C,
            r1=r1,
            r2=r2,
            r3=r3,
            r4=r4,
            z1=z1,
            z2=z2,
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

parameters = initialize_parameters()
kerr = initialize_kerr(parameters)
results = get_results(kerr)

lambda_array = kerr.lambda_array
r_lambda = results['r_lambda']
z_lambda = results['z_lambda']
phi_lambda = results['phi_lambda']
sin_theta = np.sqrt(1 - z_lambda**2)

# 将球坐标转换为笛卡尔坐标
x = r_lambda * np.sin(phi_lambda) * sin_theta
y = r_lambda * np.cos(phi_lambda) * sin_theta
z = r_lambda * z_lambda

# 根据 mode 选择要运行的功能
visualizer = GeodesicVisualization()
if mode == 0 or mode == 1:
    visualizer.plot_evolution(lambda_array, results, 1.0)
if mode == 0 or mode == 2:
    visualizer.plot_3d_trajectory(x, y, z, r1_initial, 1.2)
if mode == 0 or mode == 3:
    visualizer.animate_3d_trajectory(x, y, z, skip_steps, r1_initial, 1.2)
