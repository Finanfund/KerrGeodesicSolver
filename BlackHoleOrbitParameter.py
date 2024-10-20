import numpy as np

class BlackHoleOrbitParameter:
    """
    该类用于求解旋转黑洞引力场中的轨道参数。

    继承了数值方法来求解运动方程，包括求解运动常数和计算附加物理参数。

    方法说明：

    - `__init__` 初始化类，定义基本物理参数。
    - `delta` 计算黑洞度规函数 δ(r)。
    - `d` 计算辅助函数 d(r)。
    - `f` 计算辅助函数 f(r)。
    - `g` 计算辅助函数 g(r)。
    - `h` 计算辅助函数 h(r)。
    - `solve` 求解运动常数 (E, L, C)。
    - `compute_additional_parameters` 计算附加物理参数，包括 z2, r3, r4。
    - `_compute_horizon_radii` 计算外视界和内视界半径。
    - `parameter` 求解运动常数并获取所有计算的参数。

    参数：
    -------
    a : float
        黑洞自旋参数，范围 0 ≤ a ≤ 1。
    z1 : float
        最大偏离赤道面的参数，范围 0 ≤ z1 ≤ 1。
    r1 : float
        近日点，定义为黑洞轨道的最近点。
    r2 : float
        远日点，定义为黑洞轨道的最远点。
    M : float
        黑洞质量。

    属性：
    -------
    E : float
        求解得到的运动常数 E。
    L : float
        求解得到的运动常数 L。
    C : float
        求解得到的运动常数 C。
    z2 : float
        计算得到的参数，用于描述轨道的形状。
    r3, r4 : float
        轨道其他特征点的半径。
    r_plus : float
        外视界半径。
    r_minus : float
        内视界半径。
    """

    def __init__(self, a=0.5, z1=0.5, r1=4.0, r2=5.0, M=1.0):
        """初始化黑洞轨道参数类，定义基本物理参数。"""
        self.a = a          # 黑洞自旋参数
        self.z1 = z1        # 最大偏离赤道面的参数
        self.r1 = r1        # 近日点
        self.r2 = r2        # 远日点
        self.M = M          # 黑洞质量

    def delta(self, r):
        """计算黑洞度规函数 δ(r)。"""
        return r**2 - 2 * self.M * r + self.a**2

    def d(self, r):
        """计算辅助函数 d(r)。"""
        return self.delta(r) * (r**2 + self.a**2 * self.z1**2)

    def f(self, r):
        """计算辅助函数 f(r)。"""
        return r**4 + self.a**2 * (r * (r + 2) + self.z1**2 * self.delta(r))

    def g(self, r):
        """计算辅助函数 g(r)。"""
        return 2 * self.a * r

    def h(self, r):
        """计算辅助函数 h(r)。"""
        return r * (r - 2) + (self.z1**2 * self.delta(r)) / (1 - self.z1**2)


    def solve(self):
        """
        求解运动常数 (E, L, C)。

        使用新的公式计算 E, L, C。

        Raises:
        -------
        ValueError
            如果无法找到合理的 E, L, C 值，则抛出异常。
        """
        # 计算中间变量
        kappa = self.d(self.r2) * self.h(self.r1) - self.d(self.r1) * self.h(self.r2)
        epsilon = self.d(self.r2) * self.g(self.r1) - self.d(self.r1) * self.g(self.r2)
        rho = self.f(self.r2) * self.h(self.r1) - self.f(self.r1) * self.h(self.r2)
        eta = self.f(self.r2) * self.g(self.r1) - self.f(self.r1) * self.g(self.r2)
        sigma = self.g(self.r2) * self.h(self.r1) - self.g(self.r1) * self.h(self.r2)

        # 计算 E 的分子和分母
        numerator_E = (
            kappa * rho
            + 2 * epsilon * sigma
            - 2 * self.a * np.sqrt(
                (sigma / self.a**2)
                * (sigma * epsilon**2 + rho * epsilon * kappa - eta * kappa**2)
            )
        )
        denominator_E = rho**2 + 4 * eta * sigma

        # 计算 E
        self.E = np.sqrt(numerator_E / denominator_E)

        # 计算 L 的分子
        g_r2 = self.g(self.r2)
        h_r2 = self.h(self.r2)
        f_r2 = self.f(self.r2)
        d_r2 = self.d(self.r2)
        numerator_L = (
            -g_r2 * self.E
            + np.sqrt((g_r2**2 + h_r2 * f_r2) * self.E**2 - h_r2 * d_r2)
        )
        self.L = numerator_L / h_r2

        # 计算 C
        self.C = self.z1**2 * (
            self.a**2 * (1 - self.E**2)
            + self.L**2 / (1 - self.z1**2)
        )

        # 检查求解结果是否为有限数
        if not (np.isfinite(self.E) and np.isfinite(self.L) and np.isfinite(self.C)):
            raise ValueError("无法找到合理的 E, L, C 值。请调整输入参数。")

    def compute_additional_parameters(self):
        """
        计算附加物理参数，包括 z2, r3, r4。
        """
        # 计算 z2
        self.z2 = np.sqrt(
            self.a**2 * (1 - self.E**2)
            + self.L**2 / (1 - self.z1**2)
        )

        # 计算 r3
        term1 = 1 / (1 - self.E**2)
        term2 = (self.r1 + self.r2) / 2
        term3 = np.sqrt(
            (term2 - term1)**2
            - (self.a**2 * self.C) / (self.r1 * self.r2 * (1 - self.E**2))
        )
        self.r3 = term1 - term2 + term3

        # 利用 r3 计算 r4
        self.r4 = (self.a**2 * self.C) / (
            self.r1 * self.r2 * self.r3 * (1 - self.E**2)
        )

        # 重新排序 r1, r2, r3, r4
        r_values = np.sort([self.r1, self.r2, self.r3, self.r4])
        self.r3, self.r4, self.r1, self.r2 = r_values  # 按升序排序

    def _compute_horizon_radii(self):
        """
        计算外视界和内视界半径。

        Raises:
        -------
        ValueError
            如果自旋参数 a 的绝对值大于 1，则事件视界不存在。
        """
        if abs(self.a) > 1:
            raise ValueError("自旋参数a的绝对值不能大于1，否则事件视界不存在。")
        self.r_plus = self.M + np.sqrt(self.M**2 - self.a**2)
        self.r_minus = self.M - np.sqrt(self.M**2 - self.a**2)
        print(f"r_plus: {self.r_plus}")
        print(f"r_minus: {self.r_minus}")

    def parameter(self):
        """
        求解运动常数并获取所有计算的参数。

        返回：
        -------
        dict
            包含所有参数的字典。
        """
        # 计算事件视界半径
        self._compute_horizon_radii()

        # 检查近日点是否小于外视界半径
        if self.r1 < self.r_plus:
            raise ValueError("近日点小于外视界！请调整 r1 参数。")

        # 求解运动常数并计算附加参数
        self.solve()
        self.compute_additional_parameters()

        # 汇总所有参数
        parameters = {
            "a": self.a,
            "E": self.E,
            "L": self.L,
            "C": self.C,
            "r1": self.r1,
            "r2": self.r2,
            "r3": self.r3,
            "r4": self.r4,
            "z1": self.z1,
            "z2": self.z2
        }

        # 逐行输出计算结果
        for key, value in parameters.items():
            print(f"{key}: {value}")

        return parameters


if __name__ == "__main__":
    # 初始化并求解黑洞轨道参数
    orbit_params = BlackHoleOrbitParameter()
    parameters = orbit_params.parameter()

    # 将参数赋值到对应的变量
    E = parameters["E"]
    L = parameters["L"]
    C = parameters["C"]
    r1 = parameters["r1"]
    r2 = parameters["r2"]
    r3 = parameters["r3"]
    r4 = parameters["r4"]
    z1 = parameters["z1"]
    z2 = parameters["z2"]

    # 如果需要，可以在此处添加进一步的处理或使用这些参数
