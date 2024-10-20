import matplotlib.pyplot as plt
import numpy as np
from BlackHoleOrbitParameter import BlackHoleOrbitParameter
from KerrGeodesic import KerrGeodesic

# 定义初始参数
a = 0.99
delta_lambda = 0.005
N = 1000
z1 = 0.6
r1_initial = 100.0
r2_initial = 400.0
M = 1.0  # 黑洞质量

# 获取 BlackHoleOrbitParameter 的参数
try:
    bh_parameters = BlackHoleOrbitParameter(a, z1, r1_initial, r2_initial, M=M)
    parameters = bh_parameters.parameter()
except Exception as e:
    print(f"初始化 BlackHoleOrbitParameter 时出错: {e}")
    exit()

# 将参数赋值到对应的变量
try:
    E, L, C, r1, r2, r3, r4, z1, z2 = [
        parameters[key] for key in ["E", "L", "C", "r1", "r2", "r3", "r4", "z1", "z2"]
    ]
except KeyError as e:
    print(f"参数缺失: {e}")
    exit()

# 初始化 KerrGeodesic 对象
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

# 获取结果
try:
    results = kerr.get_results()
except Exception as e:
    print(f"获取结果时出错: {e}")
    exit()

# 输出部分结果（用于演示）
print("计算结果示例:")
print(f"t(λ) 数组的前5个元素: {results['t_lambda'][:5]}")
print(f"r(λ) 数组的前5个元素: {results['r_lambda'][:5]}")
print(f"z(λ) 数组的前5个元素: {results['z_lambda'][:5]}")
print(f"φ(λ) 数组的前5个元素: {results['phi_lambda'][:5]}")
print(f"角频率 Upsilon_t: {results['Upsilon_t']}")
print(f"角频率 Upsilon_r: {results['Upsilon_r']}")
print(f"角频率 Upsilon_z: {results['Upsilon_z']}")
print(f"角频率 Upsilon_phi: {results['Upsilon_phi']}")

# 创建 lambda 数组用于绘图
lambda_array = kerr.lambda_array

# 创建 t(λ), r(λ), z(λ), φ(λ) 的演化图
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
