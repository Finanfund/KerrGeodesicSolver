import numpy as np
from scipy.special import ellipk, ellipe, ellipj, ellipeinc, elliprf, elliprj
from typing import Dict
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

class KerrGeodesic:
    def __init__(
        self,
        a: float = 0.5,
        E: float = 0.9399345988610105,
        L: float = 2.7112773585455683,
        C: float = 2.4576243221024896,
        r1: float = 4.0,
        r2: float = 10.0,
        r3: float = 0.042226249843339975,
        r4: float = 3.121774509959378,
        z1: float = 0.5,
        z2: float = 3.135362385873193,
        delta_lambda: float = 0.005,
        N: int = 1000000
    ) -> None:
        """
        初始化 KerrGeodesic 类并设置必要的参数。

        参数:
            a (float): 自旋参数
            E (float): 能量
            L (float): 角动量
            C (float): tilde_Upsilont_z 中使用的常数
            r1 (float): 径向参数 r1
            r2 (float): 径向参数 r2
            r3 (float): 径向参数 r3
            r4 (float): 径向参数 r4
            z1 (float): z 相关参数 z1
            z2 (float): z 相关参数 z2
            delta_lambda (float): 演化步长
            N (int): 演化步数
        """
        self.a = a
        self.E = E
        self.L = L
        self.C = C
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.z1 = z1
        self.z2 = z2
        self.delta_lambda = delta_lambda
        self.N = N

        # 初始化计算
        self.compute()

    def compute(self) -> None:
        """运行整个计算过程。"""
        self.horizon_radii()
        self.constants()
        self.k_values()
        self.elliptic_integrals()
        self.angular_frequencies()
        self.evolution_parameters()
        self.sn_functions()
        self.r_and_z()
        self.t_and_phi()

    def horizon_radii(self) -> None:
        """计算外视界和内视界半径。"""
        if abs(self.a) > 1:
            raise ValueError("自旋参数 a 的绝对值不能大于 1，否则事件视界不存在。")
        self.r_plus = 1 + np.sqrt(1 - self.a**2)
        self.r_minus = 1 - np.sqrt(1 - self.a**2)
        print(f"Horizons: r_plus={self.r_plus}, r_minus={self.r_minus}")

    def constants(self) -> None:
        """计算常数 h_r, h_plus, h_minus。"""
        self.h_r = (self.r1 - self.r2) / (self.r1 - self.r3)
        self.h_plus = self.h_r * (self.r3 - self.r_plus) / (self.r2 - self.r_plus)
        self.h_minus = self.h_r * (self.r3 - self.r_minus) / (self.r2 - self.r_minus)
        print(f"常数: h_r={self.h_r}, h_plus={self.h_plus}, h_minus={self.h_minus}")

    def k_values(self) -> None:
        """根据公式计算椭圆模数 k_r 和 k_z。"""
        self.k_r = np.sqrt(
            ((self.r1 - self.r2) * (self.r3 - self.r4)) / ((self.r1 - self.r3) * (self.r2 - self.r4))
        )
        self.k_z = np.sqrt(self.a**2 * (1 - self.E**2) * (self.z1**2) / (self.z2**2))
        print(f"椭圆函数: k_r={self.k_r}, k_z={self.k_z}")

    def elliptic_integrals(self) -> None:
        """计算完全椭圆积分。"""
        self.K_k_r = ellipk(self.k_r)  # K(k_r)
        self.E_k_r = ellipe(self.k_r)  # E(k_r)
        self.K_k_z = ellipk(self.k_z)  # K(k_z)
        self.E_k_z = ellipe(self.k_z)  # E(k_z)

    def incomplete_pi(self, n: float, phi: float, m: float) -> float:
        """
        计算不完全椭圆积分 Pi。

        参数:
            n (float): 参数 n
            phi (float): 参数 phi
            m (float): 模数 m

        返回:
            float: 计算结果
        """
        sqepsilon = 1e-15
        epsilon = sqepsilon**2

        # 参数验证
        if not isinstance(n, (int, float)):
            raise TypeError("参数 n 必须是整数或浮点数。")
        if not isinstance(phi, (int, float)):
            raise TypeError("参数 phi 必须是整数或浮点数。")
        if not isinstance(m, (int, float)):
            raise TypeError("参数 m 必须是整数或浮点数。")
        if not (0 <= m <= 1):
            raise ValueError("参数 m 必须在 [0, 1] 范围内。")

        def compute_terms(n_val: float, angle: float, m_val: float) -> float:
            sin_phi = np.sin(angle)
            cos_phi_sq = np.cos(angle) ** 2
            try:
                first_term = sin_phi * (1 + 0.1 * sqepsilon) * elliprf(
                    epsilon + (1 - epsilon) * cos_phi_sq,
                    1 - m_val * sin_phi**2,
                    1,
                )
                second_term = (n_val / 3) * sin_phi**3 * (1 + sqepsilon) * elliprj(
                    epsilon + (1 - epsilon) * cos_phi_sq,
                    1 - m_val * sin_phi**2,
                    1,
                    1 - n_val * sin_phi**2,
                )
            except Exception as e:
                raise ValueError(f"计算 elliprf 或 elliprj 时出错: {e}")
            return first_term + second_term

        original_phi = phi
        phi = np.mod(phi, np.pi)  # 直接使用 np.mod 将角度标准化

        k = 2 * (original_phi - phi) / np.pi
        combined_terms = compute_terms(n, phi, m)

        midpoint = np.pi / 2
        midpoint_terms = compute_terms(n, midpoint, m)

        if phi <= midpoint:
            return combined_terms + k * (midpoint_terms + sqepsilon)
        else:
            return (k + 2) * (midpoint_terms + sqepsilon) - combined_terms

    def angular_frequencies(self) -> None:
        """计算角频率 Upsilon_r、Upsilon_z、Upsilon_t、Upsilon_phi。"""
        # 计算 Upsilon_r 和 Upsilon_z
        self.Upsilon_r = (np.pi / (2 * self.K_k_r)) * np.sqrt(
            (1 - self.E**2) * (self.r1 - self.r3) * (self.r2 - self.r4)
        )
        self.Upsilon_z = (np.pi * self.z2) / (2 * self.K_k_z)
        # 此处做了对参考文献的修改
        print(f"角频率: Upsilon_r={self.Upsilon_r}")
        print(f"角频率: Upsilon_z={self.Upsilon_z}")

        # 计算 tilde_Upsilont_r 和 tilde_Upsilont_z
        self.tilde_Upsilont_r = self.tilde_Upsilont_r()
        self.tilde_Upsilont_z = self.tilde_Upsilont_z()
        self.Upsilon_t = self.tilde_Upsilont_r + self.tilde_Upsilont_z
        print(f"角频率: Upsilon_t={self.Upsilon_t}")

        # 计算 tilde_Upsilonphi_r 和 tilde_Upsilonphi_z
        self.tilde_Upsilonphi_r = self.tilde_Upsilonphi_r()
        self.tilde_Upsilonphi_z = self.tilde_Upsilonphi_z()
        self.Upsilon_phi = self.tilde_Upsilonphi_r + self.tilde_Upsilonphi_z
        print(f"角频率: Upsilon_phi={self.Upsilon_phi}")

    def tilde_Upsilont_r(self) -> float:
        """计算 tilde_Upsilont_r。"""
        Pi_h_r_pi = self.incomplete_pi(self.h_r, np.pi / 2, self.k_r)
        Pi_h_plus_pi = self.incomplete_pi(self.h_plus, np.pi / 2, self.k_r)
        Pi_h_minus_pi = self.incomplete_pi(self.h_minus, np.pi / 2, self.k_r)
        E_pi_k_r = ellipeinc(np.pi, self.k_r**2)  # E(pi | k_r)

        term1 = (4 + self.a**2) * self.E
        term2 = self.E * 0.5 * (
            (4 + self.r1 + self.r2 + self.r3) * self.r3
            - self.r1 * self.r2
            + (self.r1 - self.r3) * (self.r2 - self.r4) * (self.E_k_r / self.K_k_r)
            + (4 + self.r1 + self.r2 + self.r3 + self.r4) * (self.r2 - self.r3) * (Pi_h_r_pi / self.K_k_r)
        )
        term3 = self.E * (2 / (self.r_plus - self.r_minus)) * (
            (
                (4 - self.a * self.L / self.E) * self.r_plus
                - 2 * self.a**2
            ) / (self.r3 - self.r_plus)
            * (1 - (self.r2 - self.r3) / (self.r2 - self.r_plus) * (Pi_h_plus_pi / self.K_k_r))
            - (
                (4 - self.a * self.L / self.E) * self.r_minus
                - 2 * self.a**2
            ) / (self.r3 - self.r_minus)
            * (1 - (self.r2 - self.r3) / (self.r2 - self.r_minus) * (Pi_h_minus_pi / self.K_k_r))
        )
        return term1 + term2 + term3

    def tilde_Upsilont_z(self) -> float:
        """计算 tilde_Upsilont_z。"""
        if self.z1 == 0:
            return -self.a**2 * self.E
        if self.a**7<1e-10:
            return(
                -self.a**2 * self.E
                + (self.E * self.C / ((1 - self.E**2) * self.z1**2))
                * (((-4 + self.r1) * self.z1**2 * self.a**2) / (2 * self.r1**3)) + \
                (2 * (-2 + self.r1)**2 * self.z1**2 * np.sqrt(1 - self.z1**2) * self.a**3) / ((-3 + self.r1) * self.r1**(9/2)) + \
                (self.z1**2 * (512 - 176 * self.z1**2 + self.r1**2 * (144 - 67 * self.z1**2) + self.r1**3 * (-8 + self.z1**2) + \
                    40 * self.r1 * (-12 + 5 * self.z1**2)) * self.a**4) / (16 * (-3 + self.r1) * self.r1**6) + \
                ((-2 + self.r1) * self.z1**2 * np.sqrt(1 - self.z1**2) * \
                (self.r1 * (920 - 206 * self.z1**2) + self.r1**3 * (106 - 22 * self.z1**2) + self.r1**4 * (-8 + self.z1**2) + \
                    8 * (-80 + 17 * self.z1**2) + self.r1**2 * (-482 + 111 * self.z1**2)) * self.a**5) / \
                (2 * (-3 + self.r1)**3 * self.r1**(15/2)) + \
                (1 / (32 * (-3 + self.r1)**4 * self.r1**9)) * \
                (self.z1**2 * (self.r1**4 * (-68048 + 58788 * self.z1**2 - 8888 * self.z1**4) + \
                self.r1**7 * (16 - 4 * self.z1**2 + self.z1**4) - 12 * self.r1**6 * (72 - 53 * self.z1**2 + 7 * self.z1**4) - \
                64 * (2816 - 2104 * self.z1**2 + 341 * self.z1**4) + \
                2 * self.r1**5 * (5640 - 4714 * self.z1**2 + 673 * self.z1**4) + \
                16 * self.r1 * (26432 - 21028 * self.z1**2 + 3371 * self.z1**4) - \
                4 * self.r1**2 * (104928 - 87580 * self.z1**2 + 13891 * self.z1**4) + \
                self.r1**3 * (224240 - 192728 * self.z1**2 + 30069 * self.z1**4)) * self.a**6)
            )
        else:
            return (
                -self.a**2 * self.E
                + (self.E * self.C / ((1 - self.E**2) * self.z1**2))
                * (self.K_k_z - self.E_k_z )/ self.K_k_z
            )


    def tilde_Upsilonphi_r(self) -> float:
        """计算 tilde_Upsilonphi_r。"""
        # 计算不完全椭圆积分 Pi(h_plus | k_r) 和 Pi(h_minus | k_r)
        Pi_h_plus = self.incomplete_pi(self.h_plus, np.pi/2, self.k_r)
        Pi_h_minus = self.incomplete_pi(self.h_minus, np.pi/2, self.k_r)
        
        # 计算完全椭圆积分 K(k_r)
        K_kr = ellipk(self.k_r)
        
        # 计算 r_plus 对应的项
        numerator_plus = 2 * self.E * self.r_plus - self.a * self.L
        denominator_plus = self.r3 - self.r_plus
        factor_plus = numerator_plus / denominator_plus
        correction_plus = (self.r2 - self.r3) / (self.r2 - self.r_plus) * (Pi_h_plus / K_kr)
        term_plus = factor_plus * (1 - correction_plus)
        
        # 计算 r_minus 对应的项
        numerator_minus = 2 * self.E * self.r_minus - self.a * self.L
        denominator_minus = self.r3 - self.r_minus
        factor_minus = numerator_minus / denominator_minus
        correction_minus = (self.r2 - self.r3) / (self.r2 - self.r_minus) * (Pi_h_minus / K_kr)
        term_minus = factor_minus * (1 - correction_minus)
        
        # 计算 tilde_Upsilonphi_r
        tilde_Upsilonphi_r = (self.a / (self.r_plus - self.r_minus)) * (term_plus - term_minus)
        
        return tilde_Upsilonphi_r

    def tilde_Upsilonphi_z(self) -> float:
        """计算 tilde_Upsilonphi_z。"""
        Pi_z1_pi = self.incomplete_pi(self.z1**2, np.pi / 2, self.k_z)
        return (self.L / self.K_k_z) * Pi_z1_pi

    def evolution_parameters(self) -> None:
        """设置演化参数。"""
        self.lambda_array = np.linspace(0, self.N * self.delta_lambda, self.N)
        self.q_r = self.Upsilon_r * self.lambda_array
        self.q_z = self.Upsilon_z * self.lambda_array
        self.q_t = self.Upsilon_t * self.lambda_array
        self.q_phi = self.Upsilon_phi * self.lambda_array
        # print("已设置演化参数。")

    def sn_functions(self) -> None:
        """计算雅可比椭圆函数 sn, cn, dn。"""
        # 计算 u_r 和 u_z
        self.u_r = (self.K_k_r / np.pi) * self.q_r
        self.sn_r, self.cn_r, self.dn_r, self.ph_r = ellipj(self.u_r, self.k_r)
        # print("径向雅可比椭圆函数 sn_r 计算完成。")

        self.u_z = (2 * self.K_k_z / np.pi) * self.q_z
        self.sn_z, self.cn_z, self.dn_z, self.ph_z = ellipj(self.u_z, self.k_z)
        # print("z 向雅可比椭圆函数 sn_z 计算完成。")

    def r_and_z(self) -> None:
        """计算 r(lambda) 和 z(lambda)。"""
        # 计算 r(lambda)
        numerator_r = self.r3 * (self.r1 - self.r2) * self.sn_r**2 - self.r2 * (self.r1 - self.r3)
        denominator_r = (self.r1 - self.r2) * self.sn_r**2 - (self.r1 - self.r3)
        # 避免除以零
        denominator_r = np.where(np.abs(denominator_r) < 1e-10, 1e-10, denominator_r)
        self.r_lambda = numerator_r / denominator_r

        # 计算 z(lambda)
        self.z_lambda = self.z1 * self.sn_z

    def t_and_phi(self) -> None:
        """计算 t(lambda) 和 phi(lambda)。"""
        # 并行计算 t_r, t_z, phi_r, phi_z
        with ProcessPoolExecutor() as executor:
            futures = {
                "t_r": executor.submit(self.t_r, self.q_r),
                "t_z": executor.submit(self.t_z, self.q_z),
                "phi_r": executor.submit(self.phi_r, self.q_r),
                "phi_z": executor.submit(self.phi_z, self.q_z)
            }

            # 收集结果
            results = {}
            for key, future in futures.items():
                results[key] = future.result()

        # 计算 t(lambda) 和 phi(lambda)
        self.t_lambda = self.q_t + results["t_r"] + results["t_z"]
        self.phi_lambda = self.q_phi + results["phi_r"] + results["phi_z"]
        # print("已计算 t(lambda) 和 phi(lambda)。")

    def t_r(self, q_r: np.ndarray) -> np.ndarray:
        """
        从径向部分计算 t_r(q_r)。

        参数:
            q_r (np.ndarray): 演化参数 q_r

        返回:
            np.ndarray: t_r 的值
        """
        # 计算雅可比幅角 am_r
        _, _, _, am_r = ellipj(self.K_k_r * q_r / np.pi, self.k_r)

        # 计算 tildet_r(am_r) 和 tildet_r(pi) 并行化
        with ProcessPoolExecutor() as executor:
            futures = {
                "am_r": executor.submit(self.vectorized_tilde_t_r, am_r),
                "pi": executor.submit(self.tilde_t_r, np.pi)
            }
            tildet_r_am_r = futures["am_r"].result()
            tildet_r_pi = futures["pi"].result()

        # 根据公式计算 t_r(q_r)
        t_r = tildet_r_am_r - (tildet_r_pi / (2 * np.pi)) * q_r
        return t_r

    def t_z(self, q_z: np.ndarray) -> np.ndarray:
        """
        从 z 部分计算 t_z(q_z)。

        参数:
            q_z (np.ndarray): 演化参数 q_z

        返回:
            np.ndarray: t_z 的值
        """
        # 计算雅可比幅角 am_z
        _, _, _, am_z = ellipj(self.K_k_z * 2 * q_z / np.pi, self.k_z)

        # 计算 tildet_z(am_z) 和 tildet_z(pi) 并行化
        with ProcessPoolExecutor() as executor:
            futures = {
                "am_z": executor.submit(self.vectorized_tilde_t_z, am_z),
                "pi": executor.submit(self.tilde_t_z, np.pi)
            }
            tildet_z_am_z = futures["am_z"].result()
            tildet_z_pi = futures["pi"].result()

        # 根据公式计算 t_z(q_z)
        t_z = tildet_z_am_z - (tildet_z_pi / np.pi) * q_z
        return t_z

    def phi_r(self, q_r: np.ndarray) -> np.ndarray:
        """
        从径向部分计算 phi_r(q_r)。

        参数:
            q_r (np.ndarray): 演化参数 q_r

        返回:
            np.ndarray: phi_r 的值
        """
        # 计算雅可比幅角 am_r
        _, _, _, am_r = ellipj(self.K_k_r * q_r / np.pi, self.k_r)

        # 计算 tildephi_r(am_r) 和 tildephi_r(pi) 并行化
        with ProcessPoolExecutor() as executor:
            futures = {
                "am_r": executor.submit(self.vectorized_tilde_phi_r, am_r),
                "pi": executor.submit(self.tilde_phi_r, np.pi)
            }
            tildephi_r_am_r = futures["am_r"].result()
            tildephi_r_pi = futures["pi"].result()

        # 根据公式计算 phi_r(q_r)
        phi_r = tildephi_r_am_r - (tildephi_r_pi / (2 * np.pi)) * q_r
        return phi_r

    def phi_z(self, q_z: np.ndarray) -> np.ndarray:
        """
        从 z 部分计算 phi_z(q_z)。

        参数:
            q_z (np.ndarray): 演化参数 q_z

        返回:
            np.ndarray: phi_z 的值
        """
        # 计算雅可比幅角 am_z
        _, _, _, am_z = ellipj(self.K_k_z * 2 * q_z / np.pi, self.k_z)

        # 计算 tildephi_z(am_z) 和 tildephi_z(pi) 并行化
        with ProcessPoolExecutor() as executor:
            futures = {
                "am_z": executor.submit(self.vectorized_tilde_phi_z, am_z),
                "pi": executor.submit(self.tilde_phi_z, np.pi)
            }
            tildephi_z_am_z = futures["am_z"].result()
            tildephi_z_pi = futures["pi"].result()

        # 根据公式计算 phi_z(q_z)
        phi_z = tildephi_z_am_z - (tildephi_z_pi / np.pi) * q_z
        return phi_z

    def vectorized_tilde_t_r(self, xi_r_array: np.ndarray) -> np.ndarray:
        """向量化计算 tilde_t_r(xi_r)"""
        return np.array([self.tilde_t_r(xi) for xi in xi_r_array])

    def vectorized_tilde_t_z(self, xi_z_array: np.ndarray) -> np.ndarray:
        """向量化计算 tilde_t_z(xi_z)"""
        return np.array([self.tilde_t_z(xi) for xi in xi_z_array])

    def vectorized_tilde_phi_r(self, xi_r_array: np.ndarray) -> np.ndarray:
        """向量化计算 tilde_phi_r(xi_r)"""
        return np.array([self.tilde_phi_r(xi) for xi in xi_r_array])

    def vectorized_tilde_phi_z(self, xi_z_array: np.ndarray) -> np.ndarray:
        """向量化计算 tilde_phi_z(xi_z)"""
        return np.array([self.tilde_phi_z(xi) for xi in xi_z_array])

    def tilde_t_r(self, xi_r: float) -> float:
        """
        Calculate the value of tildet_r(xi_r) (scalar version).

        Parameters:
            xi_r (float): Parameter xi_r

        Returns:
            float: The value of tildet_r
        """
        # Precompute common terms to simplify the expression
        numerator = self.E * (self.r2 - self.r3)
        denominator = np.sqrt(
            (1 - self.E**2) * (self.r1 - self.r3) * (self.r2 - self.r4)
        )
        common_factor = numerator / denominator

        term1 = (4 + self.r1 + self.r2 + self.r3 + self.r4) * self.incomplete_pi(self.h_r, xi_r, self.k_r)

        r_diff_plus = self.r2 - self.r_plus
        r_diff_minus = self.r2 - self.r_minus
        r3_diff_plus = self.r3 - self.r_plus
        r3_diff_minus = self.r3 - self.r_minus

        factor_plus = (self.r_plus * (4 - self.a * self.L / self.E) - 2 * self.a**2) / (r_diff_plus * r3_diff_plus)
        factor_minus = (self.r_minus * (4 - self.a * self.L / self.E) - 2 * self.a**2) / (r_diff_minus * r3_diff_minus)

        term2 = 4 / (self.r_plus - self.r_minus) * (
            factor_plus * self.incomplete_pi(self.h_plus, xi_r, self.k_r) -
            factor_minus * self.incomplete_pi(self.h_minus, xi_r, self.k_r)
        )

        ellipe_inc = ellipeinc(xi_r, self.k_r)
        term3 = self.h_r * np.sin(xi_r) * np.cos(xi_r) * np.sqrt(1 - self.k_r * np.sin(xi_r)**2)
        term4 = (1 - self.h_r * np.sin(xi_r)**2)
        term5 = (self.r1 - self.r3) * (self.r2 - self.r4) / (self.r2 - self.r3) * (
            ellipe_inc - term3 / term4
        )

        result = common_factor * (term1 - term2 + term5)

        return result

    def tilde_t_z(self, xi_z: float) -> float:
        """
        计算 tildet_z(xi_z) 的值（标量版本）。

        参数:
            xi_z (float): 参数 xi_z

        返回:
            float: tildet_z 的值
        """
        E_xi_z = ellipeinc(xi_z, self.k_z)
        return -self.E / (1 - self.E**2) * self.z2 * E_xi_z

    def tilde_phi_r(self, xi_r: float) -> float:
        """
        计算 tildephi_r(xi_r) 的值（标量版本）。

        参数:
            xi_r (float): 参数 xi_r

        返回:
            float: tildephi_r 的值
        """
        Pi_h_plus_pi = self.incomplete_pi(self.h_plus, xi_r, self.k_r)
        Pi_h_minus_pi = self.incomplete_pi(self.h_minus, xi_r, self.k_r)

        factor = -2 * self.a * self.E * (self.r2 - self.r3) / (
            (self.r_plus - self.r_minus)
            * np.sqrt((1 - self.E**2) * (self.r1 - self.r3) * (self.r2 - self.r4))
        )
        term = (
            ((2 * self.r_plus - self.a * self.L / self.E) / ((self.r2 - self.r_plus) * (self.r3 - self.r_plus)))
            * Pi_h_plus_pi
            - ((2 * self.r_minus - self.a * self.L / self.E) / ((self.r2 - self.r_minus) * (self.r3 - self.r_minus)))
            * Pi_h_minus_pi
        )
        return factor * term

    def tilde_phi_z(self, xi_z: float) -> float:
        """
        计算 tildephi_z(xi_z) 的值（标量版本）。

        参数:
            xi_z (float): 参数 xi_z

        返回:
            float: tildephi_z 的值
        """
        Pi_z1_pi = self.incomplete_pi(self.z1**2, xi_z, self.k_z)
        return (self.L / self.z2) * Pi_z1_pi

    def get_results(self) -> Dict[str, np.ndarray]:
        """
        获取计算得到的 t(lambda)、r(lambda)、z(lambda) 和 phi(lambda) 数组，
        以及角频率 Upsilon_t、Upsilon_r、Upsilon_z、Upsilon_phi。

        返回:
            Dict[str, np.ndarray]: 计算结果的字典
        """
        return {
            't_lambda': self.t_lambda,
            'r_lambda': self.r_lambda,
            'z_lambda': self.z_lambda,
            'phi_lambda': self.phi_lambda,
            'Upsilon_t': self.Upsilon_t,
            'Upsilon_r': self.Upsilon_r,
            'Upsilon_z': self.Upsilon_z,
            'Upsilon_phi': self.Upsilon_phi
        }

import time
# 示例用法
if __name__ == "__main__":

    start_time = time.perf_counter()  # 开始计时
    # 初始化 KerrGeodesic 对象
    kerr = KerrGeodesic()

    # 获取结果
    results = kerr.get_results()

    # 输出部分结果（用于演示）
    print(f"t(λ) 前5个元素: {results['t_lambda'][:5]}")
    print(f"r(λ) 前5个元素: {results['r_lambda'][:5]}")
    print(f"z(λ) 前5个元素: {results['z_lambda'][:5]}")
    print(f"φ(λ) 前5个元素: {results['phi_lambda'][:5]}")
    print(f"角频率 Upsilon_t: {results['Upsilon_t']}")
    print(f"角频率 Upsilon_r: {results['Upsilon_r']}")
    print(f"角频率 Upsilon_z: {results['Upsilon_z']}")
    print(f"角频率 Upsilon_phi: {results['Upsilon_phi']}")

    end_time = time.perf_counter()  # 结束计时
    print(f"运行时间: {end_time - start_time:.2f} 秒")

    # 创建 lambda 数组用于绘图
    lambda_array = kerr.lambda_array

    # 创建 t(λ), r(λ), z(λ), φ(λ) 的演化图
    plt.figure(figsize=(10, 6))

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
