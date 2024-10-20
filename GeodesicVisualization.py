# GeodesicVisualization.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class GeodesicVisualization:
    
    # 绘制 t(λ), r(λ), z(λ), φ(λ) 的演化图
    @staticmethod
    def plot_evolution(lambda_array, results):
        plt.figure(figsize=(12, 8))

        # t(λ) 演化
        plt.subplot(2, 2, 1)
        plt.plot(lambda_array, results['t_lambda'], label='t(λ)')
        plt.xlabel('λ')
        plt.ylabel('t(λ)')
        plt.title('t(λ) vs λ')
        plt.grid(True)

        # r(λ) 演化
        plt.subplot(2, 2, 2)
        plt.plot(lambda_array, results['r_lambda'], label='r(λ)', color='orange')
        plt.xlabel('λ')
        plt.ylabel('r(λ)')
        plt.title('r(λ) vs λ')
        plt.grid(True)

        # z(λ) 演化
        plt.subplot(2, 2, 3)
        plt.plot(lambda_array, results['z_lambda'], label='z(λ)', color='green')
        plt.xlabel('λ')
        plt.ylabel('z(λ)')
        plt.title('z(λ) vs λ')
        plt.grid(True)

        # φ(λ) 演化
        plt.subplot(2, 2, 4)
        plt.plot(lambda_array, results['phi_lambda'], label='φ(λ)', color='red')
        plt.xlabel('λ')
        plt.ylabel('φ(λ)')
        plt.title('φ(λ) vs λ')
        plt.grid(True)

        # 调整布局并显示图像
        plt.tight_layout()
        plt.show()

    # 绘制三维轨迹图
    @staticmethod
    def plot_3d_trajectory(x, y, z, r2_initial):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制轨迹
        ax.plot(x, y, z, label='Kerr Geodesic Trajectory', lw=0.8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Kerr Geodesic Trajectory')
        ax.legend()

        # 根据 r2 的值设置坐标轴范围
        ax.set_xlim([-r2_initial * 1.2, r2_initial * 1.2])
        ax.set_ylim([-r2_initial * 1.2, r2_initial * 1.2])
        ax.set_zlim([-r2_initial * 1.2, r2_initial * 1.2])

        # 添加交互
        ax.view_init(elev=30, azim=45)  # 设置初始视角
        plt.tight_layout()
        plt.show()

    # 创建三维轨迹动画
    @staticmethod
    def animate_3d_trajectory(x, y, z, skip_steps, r2_initial):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Kerr Geodesic Trajectory')

        # 根据 r2 的值设置坐标轴范围
        ax.set_xlim([-r2_initial * 1.2, r2_initial * 1.2])
        ax.set_ylim([-r2_initial * 1.2, r2_initial * 1.2])
        ax.set_zlim([-r2_initial * 1.2, r2_initial * 1.2])

        # 初始化绘制
        line, = ax.plot([], [], [], label='Kerr Geodesic Trajectory', lw=0.8)
        ax.legend()

        # 动画初始化函数
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            return line,

        # 动画更新函数
        def update(frame):
            frame = frame * skip_steps  # 每帧跳过指定数量的 lambda 值
            if frame >= len(x):
                frame = len(x) - 1
            line.set_data(x[:frame + 1], y[:frame + 1])
            line.set_3d_properties(z[:frame + 1])
            return line,

        # 创建动画，加速动画通过减小间隔和减少帧数实现
        ani = FuncAnimation(fig, update, frames=len(x) // skip_steps, init_func=init, blit=False, interval=10, repeat=False)

        # 添加交互
        ax.view_init(elev=30, azim=45)  # 设置初始视角
        plt.tight_layout()
        plt.show()
