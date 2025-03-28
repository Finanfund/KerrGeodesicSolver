import numpy as np
import warnings

class KerrParameters_1:
    """
    该类用于求解旋转黑洞引力场中的轨道参数。

    继承了数值方法来求解运动方程，包括求解运动常数和计算附加物理参数。

    方法说明：

    - `__init__` 初始化类，定义基本物理参数。
    - `solve` 求解运动常数 (E, L, C)。
    - `solve_r_z` 计算附加物理参数，包括 z2, r3, r4。
    - `solve_horizon` 计算外视界和内视界半径。
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

    def __init__(self, a=0.5, z1=0.7, r1=5.0, r2=4.0, M=1.0):
        """初始化黑洞轨道参数类，定义基本物理参数。"""
        self.a = a          # 黑洞自旋参数
        self.z1 = z1        # 最大偏离赤道面的参数
        self.r1 = r1        # 远日点
        self.r2 = r2        # 近日点
        self.M = M          # 黑洞质量

    def solve(self):
        """
        求解运动常数 (E, L, C)。

        Raises:
        -------
        ValueError
            如果无法找到合理的 E, L, C 值，则抛出异常。
        """
        def delta(r):
            return r**2 - 2 * self.M * r + self.a**2

        def d_func(r):
            return delta(r) * (r**2 + self.a**2 * self.z1**2)

        def f_func(r):
            return r**4 + self.a**2 * (r * (r + 2) + self.z1**2 * delta(r))

        def g_func(r):
            return 2 * self.a * r

        def h_func(r):
            if 1 - self.z1**2 == 0:
                raise ValueError("h_func 分民为零，无法计算 h(r)。")
            return r * (r - 2) + (self.z1**2 * delta(r)) / (1 - self.z1**2)

        # 计算中间变量
        kappa = d_func(self.r2) * h_func(self.r1) - d_func(self.r1) * h_func(self.r2)
        epsilon = d_func(self.r2) * g_func(self.r1) - d_func(self.r1) * g_func(self.r2)
        rho = f_func(self.r2) * h_func(self.r1) - f_func(self.r1) * h_func(self.r2)
        eta = f_func(self.r2) * g_func(self.r1) - f_func(self.r1) * g_func(self.r2)
        sigma = g_func(self.r2) * h_func(self.r1) - g_func(self.r1) * h_func(self.r2)

        if self.r1!=self.r2:
            
            # 计算 E 的分子和分母
            term_inside_sqrt = sigma * (sigma * epsilon**2 + rho * epsilon * kappa - eta * kappa**2)
            if term_inside_sqrt < 0:
                raise ValueError("平方根内的项为负，无法计算 E。")
            numerator_E = (
                kappa * rho
                + 2 * epsilon * sigma
                - 2 * np.sqrt(term_inside_sqrt)
            )
            if rho**2 + 4 * eta * sigma == 0:
                raise ValueError("E 的分母为零，无法计算 E。")
            denominator_E = rho**2 + 4 * eta * sigma

            # 计算 E
            self.E = np.sqrt(numerator_E / denominator_E)
        else:
            # Numerator
            term1 = (-3 + self.r1) * (-2 + self.r1)**2 * self.r1**7
            term2 = -2 * self.a * (-1 + self.z1**2)**2 * np.sqrt(
                (self.r1 * (self.a**2 + (-2 + self.r1) * self.r1)**2 * 
                (-self.r1**2 + self.a**2 * self.z1**2)**3 * 
                (self.r1**2 + self.a**2 * self.z1**2)**2) / 
                (-1 + self.z1**2)**3
            )
            term3 = self.a**8 * self.z1**6 * (-1 + 3 * self.r1 + (-1 + self.r1)**2 * self.z1**2)
            term4 = self.a**6 * self.r1**2 * self.z1**4 * (-5 + 3 * self.r1 + (1 + self.r1 * (-7 + 4 * self.r1)) * self.z1**2)
            term5 = self.a**4 * self.r1**3 * self.z1**2 * (
                self.r1 - 3 * self.r1**2 + (4 + self.r1 * (7 + 3 * self.r1 * (-5 + 2 * self.r1))) * self.z1**2
            )
            term6 = self.a**2 * self.r1**5 * (
                (5 - 3 * self.r1) * self.r1 + 
                (-8 + self.r1 * (23 + self.r1 * (-17 + 4 * self.r1))) * self.z1**2
            )
            numerator = term1 + term2 + term3 + term4 + term5 + term6

            # Denominator
            denom1 = (self.r1**2 + self.a**2 * self.z1**2)**2
            denom2 = ((-3 + self.r1)**2 * self.r1**4 + 
                    self.a**4 * self.z1**2 * (4 * self.r1 + (-1 + self.r1)**2 * self.z1**2) +
                    2 * self.a**2 * self.r1**2 * (-2 * self.r1 + (-3 + self.r1**2) * self.z1**2))
            denominator = denom1 * denom2
            self.E = np.sqrt(numerator / denominator)
        # 计算 L 的分子
        g_r2 = g_func(self.r2)
        h_r2 = h_func(self.r2)
        f_r2 = f_func(self.r2)
        d_r2 = d_func(self.r2)
        term_inside_sqrt_L = (g_r2**2 + h_r2 * f_r2) * self.E**2 - h_r2 * d_r2
        if term_inside_sqrt_L < 0:
            raise ValueError("L 的平方根内的项为负，无法计算 L。")
        numerator_L = (
            -g_r2 * self.E
            + np.sqrt(term_inside_sqrt_L)
        )
        if h_r2 == 0:
            raise ValueError("h(r2) 为零，无法计算 L。")
        self.L = numerator_L / h_r2

        # 计算 C
        if 1 - self.z1**2 == 0:
            raise ValueError("C 的分母为零，无法计算 C。")
        self.C = self.z1**2 * (
            self.a**2 * (1 - self.E**2)
            + self.L**2 / (1 - self.z1**2)
        )

        # 检查求解结果是否为有限数
        if not (np.isfinite(self.E) and np.isfinite(self.L) and np.isfinite(self.C)):
            raise ValueError("无法找到合理的 E, L, C 值。请调整输入参数。")

    def solve_r_z(self):
        """
        计算附加物理参数，包括 z2, r3, r4。
        """
        # 计算 z2
        if 1 - self.z1**2 == 0:
            raise ValueError("z2 的分母为零，无法计算 z2。")
        self.z2 = np.sqrt(
            self.a**2 * (1 - self.E**2)
            + self.L**2 / (1 - self.z1**2)
        )

        # 计算 r3
        if 1 - self.E**2 == 0:
            raise ValueError("r3 的分母为零，无法计算 r3。")
        term1 = 1 / (1 - self.E**2)
        term2 = (self.r1 + self.r2) / 2
        term_inside_sqrt = (term2 - term1)**2 - (self.a**2 * self.C) / (self.r1 * self.r2 * (1 - self.E**2))
        if term_inside_sqrt < 0:
            raise ValueError("r3 的平方根内的项为负，无法计算 r3。")
        term3 = np.sqrt(term_inside_sqrt)
        self.r3 = term1 - term2 + term3

        # 利用 r3 计算 r4
        denominator_r4 = self.r1 * self.r2 * self.r3 * (1 - self.E**2)
        # if denominator_r4 == 0:
        #     raise ValueError("计算 r3 为零，故 r4 的分母为零，无法计算 r4。")
        self.r4 = (self.a**2 * self.C) / denominator_r4

        # 重新排序 r1, r2, r3, r4
        r_values = np.sort([self.r1, self.r2, self.r3, self.r4])
        self.r4, self.r3, self.r2, self.r1 = r_values  # 按升序排序

    def solve_horizon(self):
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
        self.solve_horizon()

        # 检查近日点是否小于外视界半径
        if self.r1 < self.r_plus:
            print("警告: 近日点小于外视界！请调整 r1 参数。")

        # 求解运动常数并计算附加参数
        self.solve()
        self.solve_r_z()

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

class KerrParameters_2:
    """
    该类用于根据给定的运动常数 (E, L, C) 求解旋转黑洞引力场中的轨道参数。

    继承了数值方法来求解运动方程，包括计算轨道半径和描述轨道形状的参数。

    方法说明：

    - `__init__` 初始化类，定义基本物理参数。
    - `solve` 求解轨道半径 (r1, r2, r3, r4) 和最大偏离赤道面的参数 z1。
    - `solve_horizon` 计算外视界和内视界半径。
    - `parameter` 求解所有参数并获取计算结果。

    参数：
    -------
    a : float
        黑洞自旋参数，范围 0 ≤ a ≤ 1。
    P1 : float
        运动常数 E。
    P2 : float
        运动常数 C。
    P3 : float
        运动常数 L。
    M : float, 可选
        黑洞质量，默认为 1.0。

    属性：
    -------
    E : float
        给定的运动常数 E。
    L : float
        给定的运动常数 L。
    C : float
        给定的运动常数 C。
    z1 : float
        计算得到的最大偏离赤道面的参数。
    z2 : float
        计算得到的参数，用于描述轨道的形状。
    r1, r2, r3, r4 : float
        轨道四个特征点的半径。
    r_plus : float
        外视界半径。
    r_minus : float
        内视界半径。
    """

    def __init__(self, a=0.5, P1=0.9469832369813538, P2=11.731192036000717, P3=0.4875214698910026, M=1.0):
        """初始化黑洞轨道参数类，定义基本物理参数。"""
        self.a = a          # 黑洞自旋参数
        self.P1 = P1        # 运动常数 E
        self.P2 = P2        # 运动常数 C
        self.P3 = P3        # 运动常数 L
        self.M = M          # 黑洞质量

    def solve_quartic(self, a4, a3, a2, a1, a0):
        """
        求解四次方程 a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0 = 0 的四个根。

        参数：
        a4, a3, a2, a1, a0 : float
            四次方程的系数，其中 a4 != 0。

        返回：
        roots : numpy.ndarray
            四个根，可能是实数或复数，按从大到小排序。
        """
        if a4 == 0:
            raise ValueError("系数 a4 不能为零，这不是一个四次方程。")
        
        # 构造多项式系数列表，按降幂顺序
        coefficients = [a4, a3, a2, a1, a0]
        
        # 使用 numpy.roots 计算根
        roots = np.roots(coefficients)
        
        # 对根进行从大到小排序
        roots = np.sort(roots)[::-1]
        
        return roots

    def solve_R(self):
        """
        求解自定义四次方程：
        a^4 P1^2 - a^2 P2 - a^2 (a P1 - P3)^2 - 2 a^3 P1 P3 + a^2 P3^2 +
        (2 P2 + 2 (a P1 - P3)^2) r + (-a^2 + 2 a^2 P1^2 - P2 - (a P1 - P3)^2 - 2 a P1 P3) r^2 +
        2 r^3 + (-1 + P1^2) r^4 == 0

        返回：
        roots : numpy.ndarray
            四个根，可能是实数或复数，按从大到小排序。
        """
        # 计算多项式系数
        coef_r4 = -1 + self.P1**2
        coef_r3 = 2
        coef_r2 = -self.a**2 + 2 * self.a**2 * self.P1**2 - self.P2 - (self.a * self.P1 - self.P3)**2 - 2 * self.a * self.P1 * self.P3
        coef_r1 = 2 * self.P2 + 2 * (self.a * self.P1 - self.P3)**2
        coef_r0 = self.a**4 * self.P1**2 - self.a**2 * self.P2 - self.a**2 * (self.a * self.P1 - self.P3)**2 - 2 * self.a**3 * self.P1 * self.P3 + self.a**2 * self.P3**2
        
        # 使用 solve_quartic 求解方程
        roots = self.solve_quartic(coef_r4, coef_r3, coef_r2, coef_r1, coef_r0)
        
        return roots

    def solve_Z(self):
        """
        求解另一个自定义四次方程：
        P2 + (-a^2 (1 - P1^2) - P2 - P3^2) z^2 + a^2 (1 - P1^2) z^4 == 0

        返回：
        roots : numpy.ndarray
            四个根，可能是实数或复数，按从大到小排序。
        """
        if self.a == 0:
            # 计算特殊情况下的解
            root1 = np.sqrt(self.P2) / np.sqrt(self.P2 + self.P3**2)
            root2 = -np.sqrt(self.P2) / np.sqrt(self.P2 + self.P3**2)
            sorted_roots = np.sort([root1, root2])[::-1]
            return np.array([sorted_roots[0], sorted_roots[0], sorted_roots[1], sorted_roots[1]])
        else:
            # 计算多项式系数
            coef_z4 = self.a**2 * (1 - self.P1**2)
            coef_z3 = 0
            coef_z2 = -self.a**2 * (1 - self.P1**2) - self.P2 - self.P3**2
            coef_z1 = 0
            coef_z0 = self.P2
            
            # 使用 solve_quartic 求解方程
            roots = self.solve_quartic(coef_z4, coef_z3, coef_z2, coef_z1, coef_z0)
            roots = np.array([roots[0]*(self.a*np.sqrt(1-self.P1**2)), roots[1], roots[2], roots[3]*(self.a*np.sqrt(1-self.P1**2))])
        return roots

    def solve(self):
        """
        求解轨道半径 (r1, r2, r3, r4) 和最大偏离赤道面的参数 z1 和 z2。

        Raises:
        -------
        ValueError
            如果无法找到合理的 r1, r2, r3, r4 或 z1, z2 值，则抛出异常。
        """
        # 使用 solve_R 函数求解 r 的根
        roots_r = self.solve_R()
        # 筛选实数根
        real_roots_r = roots_r[np.isreal(roots_r)].real
        if len(real_roots_r) < 4:
            warnings.warn("无法找到四个实数根 r1, r2, r3, r4。请检查输入参数。")
        # 排序并赋值
        sorted_r = np.sort(real_roots_r)[::-1]
        self.r1, self.r2, self.r3, self.r4 = sorted_r

        # 使用 solve_Z 函数求解 z 的根
        roots_z = self.solve_Z()
        # 筛选实数根并取正值
        real_roots_z = roots_z[np.isreal(roots_z)].real
        real_roots_z = real_roots_z[real_roots_z >= 0]
        if len(real_roots_z) == 0:
            warnings.warn("无法找到有效的 z1 和 z2。请检查输入参数。")
        # 选择最大正实数根作为 z2，选择第二大正实数根作为 z1
        sorted_z = np.sort(real_roots_z)[::-1]
        self.z2, self.z1 = sorted_z[:2]
        real_roots_z = real_roots_z[real_roots_z != self.z2]
        if len(real_roots_z) == 0:
            warnings.warn("无法找到有效的 z1。请检查输入参数。")
        self.z1 = np.max(real_roots_z)

    def solve_horizon(self):
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
        求解所有参数并获取计算结果。

        返回：
        -------
        dict
            包含所有参数的字典。
        """
        # 计算事件视界半径
        self.solve_horizon()

        # 求解轨道半径和 z1, z2
        self.solve()

        # 检查轨道半径是否大于外视界半径
        if self.r1 < self.r_plus:
            warnings.warn("轨道半径 r1 小于外视界半径 r_plus！请调整输入参数。")

        # 汇总所有参数
        parameters = {
            "a": self.a,
            "E": self.P1,
            "L": self.P3,
            "C": self.P2,
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
    mode = 1
    if mode == 0 or mode == 1:
        # 初始化并求解黑洞轨道参数
        orbit_params = KerrParameters_1(a=0.,z1=0.5,r1=50,r2=50,M=1)
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
    if mode == 0 or mode == 2:
        a=0.5
        orbit_params_1 = KerrParameters_1(a,0.99,10,5)
        parameters_1 = orbit_params_1.parameter()
        E = parameters_1["E"]
        L = parameters_1["L"]
        C = parameters_1["C"]
        P1=E
        P2=C
        P3=L
        # 初始化并求解黑洞轨道参数
        orbit_params_2 = KerrParameters_2(a, P1, P2, P3)
        parameters_2 = orbit_params_2.parameter()