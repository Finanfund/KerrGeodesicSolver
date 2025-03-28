# GeodesicVisualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

class GeodesicVisualization:
    
    @staticmethod
    def plot_evolution(lambda_array, t_array, r_array, z_array, phi_array, figsize=1.0):
        """
        绘制 t(λ), r(λ), z(λ), φ(λ) 随 λ 的演化图。

        参数：
        lambda_array (array-like): λ 值的数组。
        t_array (array-like): t 的演化数据。
        r_array (array-like): r 的演化数据。
        z_array (array-like): z 的演化数据。
        phi_array (array-like): φ 的演化数据。
        figsize (float): 图像大小的缩放因子。
        """
        plt.figure(figsize=(12 * figsize, 8 * figsize))

        # t(λ) 演化
        plt.subplot(2, 2, 1)
        plt.plot(lambda_array, t_array, label='t(λ)')
        plt.xlabel('λ')
        plt.ylabel('t(λ)')
        plt.title('t(λ) vs λ')
        plt.grid(True)

        # r(λ) 演化
        plt.subplot(2, 2, 2)
        plt.plot(lambda_array, r_array, label='r(λ)', color='orange')
        plt.xlabel('λ')
        plt.ylabel('r(λ)')
        plt.title('r(λ) vs λ')
        plt.grid(True)

        # z(λ) 演化
        plt.subplot(2, 2, 3)
        plt.plot(lambda_array, z_array, label='z(λ)', color='green')
        plt.xlabel('λ')
        plt.ylabel('z(λ)')
        plt.title('z(λ) vs λ')
        plt.grid(True)

        # φ(λ) 演化
        plt.subplot(2, 2, 4)
        plt.plot(lambda_array, phi_array, label='φ(λ)', color='red')
        plt.xlabel('λ')
        plt.ylabel('φ(λ)')
        plt.title('φ(λ) vs λ')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_3d_trajectory(x, y, z, r1_initial, figsize=1.0):
        """
        绘制 Kerr 测地线的 3D 轨迹。

        参数：
        x, y, z (array-like): 轨迹的坐标。
        r1_initial (float): 用于设置坐标轴范围的初始半径。
        figsize (float): 图像大小的缩放因子。
        """
        fig = plt.figure(figsize=(10 * figsize, 8 * figsize))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制轨迹
        ax.plot(x, y, z, label='Kerr Geodesic Trajectory', lw=0.8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Kerr Geodesic Trajectory')
        ax.legend()

        # 根据 r1_initial 值设置坐标轴范围
        ax.set_xlim([-r1_initial * 1.2, r1_initial * 1.2])
        ax.set_ylim([-r1_initial * 1.2, r1_initial * 1.2])
        ax.set_zlim([-r1_initial * 1.2, r1_initial * 1.2])

        # 添加交互
        ax.view_init(elev=30, azim=45)  # 设置初始视角
        plt.tight_layout()
        plt.show()

    @staticmethod
    def animate_3d_trajectory(x, y, z, skip_steps, r1_initial, figsize=1.0):
        """
        创建 Kerr 测地线的 3D 轨迹动画。

        参数：
        x, y, z (array-like): 轨迹的坐标。
        skip_steps (int): 每帧跳过的步数。
        r1_initial (float): 用于设置坐标轴范围的初始半径。
        figsize (float): 图像大小的缩放因子。
        """
        fig = plt.figure(figsize=(10 * figsize, 8 * figsize))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Kerr Geodesic Trajectory')

        # 根据 r1_initial 值设置坐标轴范围
        ax.set_xlim([-r1_initial * 1.1, r1_initial * 1.1])
        ax.set_ylim([-r1_initial * 1.1, r1_initial * 1.1])
        ax.set_zlim([-r1_initial * 1.1, r1_initial * 1.1])

        # 初始化绘图
        line, = ax.plot([], [], [], label='Kerr Geodesic Trajectory', lw=0.8)
        ax.legend()

        # 动画初始化函数
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            return line,

        # 动画更新函数
        def update(frame):
            frame = frame * skip_steps  # 每帧跳过指定数量的 λ 值
            if frame >= len(x):
                frame = len(x) - 1
            line.set_data(x[:frame + 1], y[:frame + 1])
            line.set_3d_properties(z[:frame + 1])
            return line,

        # 创建动画
        ani = FuncAnimation(fig, update, frames=len(x) // skip_steps, init_func=init, blit=False, interval=10, repeat=False)

        # 添加交互
        ax.view_init(elev=30, azim=45)  # 设置初始视角
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_evolution_cartesian(lambda_array, t, x, y, z, figsize=1.0):
        """
        绘制 t(λ), x(λ), y(λ), z(λ) 随 λ 的演化图。

        参数：
        lambda_array (array-like): λ 值的数组。
        x, y, z, t (array-like): x, y, z, t 的演化数据。
        figsize (float): 图像大小的缩放因子。
        """
        plt.figure(figsize=(12 * figsize, 8 * figsize))

        # t(λ) 演化
        plt.subplot(2, 2, 1)
        plt.plot(lambda_array, t, label='t(λ)')
        plt.xlabel('λ')
        plt.ylabel('t(λ)')
        plt.title('t(λ) vs λ')
        plt.grid(True)

        # x(λ) 演化
        plt.subplot(2, 2, 2)
        plt.plot(lambda_array, x, label='x(λ)', color='blue')
        plt.xlabel('λ')
        plt.ylabel('x(λ)')
        plt.title('x(λ) vs λ')
        plt.grid(True)

        # y(λ) 演化
        plt.subplot(2, 2, 3)
        plt.plot(lambda_array, y, label='y(λ)', color='orange')
        plt.xlabel('λ')
        plt.ylabel('y(λ)')
        plt.title('y(λ) vs λ')
        plt.grid(True)

        # z(λ) 演化
        plt.subplot(2, 2, 4)
        plt.plot(lambda_array, z, label='z(λ)', color='green')
        plt.xlabel('λ')
        plt.ylabel('z(λ)')
        plt.title('z(λ) vs λ')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_trajectory_difference(lambda_array, t1, x1, y1, z1, t2, x2, y2, z2, figsize=1.0):
        """
        绘制两条轨迹的差值。

        参数：
        lambda_array (array-like): λ 值的数组。
        t1, x1, y1, z1 (array-like): 第一条轨迹的 t, x, y, z 数据。
        t2, x2, y2, z2 (array-like): 第二条轨迹的 t, x, y, z 数据。
        figsize (float): 图像大小的缩放因子。
        """
        plt.figure(figsize=(12 * figsize, 8 * figsize))

        # t(λ) 差值
        plt.subplot(2, 2, 1)
        t_diff = [a - b for a, b in zip(t1, t2)]
        plt.plot(lambda_array, t_diff, label='Δt(λ)')
        plt.xlabel('λ')
        plt.ylabel('Δt(λ)')
        plt.title('Δt(λ) vs λ')
        plt.grid(True)

        # x(λ) 差值
        plt.subplot(2, 2, 2)
        x_diff = [a - b for a, b in zip(x1, x2)]
        plt.plot(lambda_array, x_diff, label='Δx(λ)', color='blue')
        plt.xlabel('λ')
        plt.ylabel('Δx(λ)')
        plt.title('Δx(λ) vs λ')
        plt.grid(True)

        # y(λ) 差值
        plt.subplot(2, 2, 3)
        y_diff = [a - b for a, b in zip(y1, y2)]
        plt.plot(lambda_array, y_diff, label='Δy(λ)', color='orange')
        plt.xlabel('λ')
        plt.ylabel('Δy(λ)')
        plt.title('Δy(λ) vs λ')
        plt.grid(True)

        # z(λ) 差值
        plt.subplot(2, 2, 4)
        z_diff = [a - b for a, b in zip(z1, z2)]
        plt.plot(lambda_array, z_diff, label='Δz(λ)', color='green')
        plt.xlabel('λ')
        plt.ylabel('Δz(λ)')
        plt.title('Δz(λ) vs λ')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_geodesic_difference_rtphi(t1, r1, z1, phi1, t2, r2, z2, phi2, figsize=1.0):
        """
        绘制两组 t, r, z, φ 轨迹的差值。

        参数：
        t1, r1, z1, phi1 (array-like): 第一组轨迹的 t, r, z, φ 数据。
        t2, r2, z2, phi2 (array-like): 第二组轨迹的 t, r, z, φ 数据。
        figsize (float): 图像大小的缩放因子。
        """
        # 找出 r1(t1), r2(t2) 的函数关系
        r1_func = interp1d(t1, r1, kind='linear', fill_value='extrapolate')
        z1_func = interp1d(t1, z1, kind='linear', fill_value='extrapolate')
        phi1_func = interp1d(t1, phi1, kind='linear', fill_value='extrapolate')

        r2_func = interp1d(t2, r2, kind='linear', fill_value='extrapolate')
        z2_func = interp1d(t2, z2, kind='linear', fill_value='extrapolate')
        phi2_func = interp1d(t2, phi2, kind='linear', fill_value='extrapolate')

        # 生成公共 t 轴
        t_min = max(min(t1), min(t2))
        t_max = min(max(t1), max(t2))
        t_common = np.linspace(t_min, t_max, num=1000)

        # 计算差值
        r_diff = r1_func(t_common) - r2_func(t_common)
        z_diff = z1_func(t_common) - z2_func(t_common)
        phi_diff = phi1_func(t_common) - phi2_func(t_common)

        # 绘制差值图
        plt.figure(figsize=(15 * figsize, 8 * figsize))

        # Δr vs t_common
        plt.subplot(3, 1, 1)
        plt.plot(t_common, r_diff, label='Δr', color='blue')
        plt.xlabel('t')
        plt.ylabel('Δr')
        plt.title('Δr vs t')
        plt.grid(True)

        # Δz vs t_common
        plt.subplot(3, 1, 2)
        plt.plot(t_common, z_diff, label='Δz', color='green')
        plt.xlabel('t')
        plt.ylabel('Δz')
        plt.title('Δz vs t')
        plt.grid(True)

        # Δφ vs t_common
        plt.subplot(3, 1, 3)
        plt.plot(t_common, phi_diff, label='Δφ', color='red')
        plt.xlabel('t')
        plt.ylabel('Δφ')
        plt.title('Δφ vs t')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
